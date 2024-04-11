from flax import linen as nn
# from flax.core import FrozenDict
from flax.training.train_state import TrainState

import jax
from jax import Array, jit
import jax.numpy as jnp
from jax import random
from jax import config as jax_config

jax_config.update("jax_platform_name", "cpu")
jax_config.update("jax_enable_x64", True)

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import optax
from typing import Dict, Tuple
import skopt

def get_parameters():
    parameters = {
        "tmin": 0.0,
        "tmax": 10.0,
        "m": 1.0,
        "l": 1.0,
        "g": 9.8,
        "torq_max": 1.5,
        "target": -1.0
    }
    return parameters

class PointResampler:

  def __init__(self, train_x_bc,
  tmin, tmax):

    assert tmin <= tmax, "tmin should be lesser than tmax"
    self.tmin, self.tmax = tmin,tmax
    self.diam = tmax - tmin
    self.sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
    # initializing the sampler
    self.period = 100
    self.train_x_bc = train_x_bc
    self.space = [(0.0, 1.0)]
    self.epoch_counter = 0

  def get_train_samples(self, n_samples):
    if self.epoch_counter % self.period == 0:
      self.epoch_counter = 0
      x = np.asarray(self.sampler.generate(self.space,
      n_samples)[0:])
      self.train_x = self.tmin + self.diam * x
      self.train_x_all = np.vstack((self.train_x_bc, self.train_x))
      self.epoch_counter += 1
    
    return self.train_x_all

class Pendulum(nn.Module):
    """
    Physics Informed Neural network for pendulum
    """

    num_hidden: int = 64  # Number of hidden units per intermediate layer
    num_outputs: int = 2 # theta, torq_norm


    @nn.compact
    def __call__(self, t: Array) -> Array:
        """
        Evaluate theta and torque
        Args:
            t: time
        Returns:
            u: output of shape (2, )
        """
        # implement the neural network layers
        layers = [nn.Dense(self.num_hidden), nn.tanh, \
                  nn.Dense(self.num_hidden), nn.tanh, \
                  nn.Dense(self.num_hidden), nn.tanh, \
                  nn.Dense(self.num_outputs)]
        u = t
        for layer in layers:
            u = layer(u)
        return u
@jit
def mse_loss_fn(pred: Array, target: Array) -> Array:
    loss = jnp.mean((pred - target)**2)
    return loss

def initialize_train_states(rng: Array) -> TrainState:
    """
    Initialize the train state of the neural network.
    Args:
        rng: PRNG key for pseudo-random number generation.
    Returns:
        pendulum_train_state: the current state of the training of the neural network.
    """
    # initialize the neural network object
    pendulum = Pendulum()

    # initialize parameters of the neural networks by passing a dummy input through the network
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (1,))  # dummy input
    pendulum_params = pendulum.init(init_rng, inp)['params']


    # initialize the adam optimizer for the neural network
    pendulum_tx = optax.adam(learning_rate=1.5e-2)

    # create the TrainState object for the neural network
    pendulum_train_state = TrainState.create(
        apply_fn=pendulum.apply,
        params=pendulum_params,
        tx=pendulum_tx,
    )

    return pendulum_train_state

@jit
def train_step(state: TrainState, ts: Array) -> TrainState:
    """
    Trains the neural network for one step.
    Args:
        state: the current state of the training of the neural network
        ts: array of timesteps
    Returns:
        state: updated training state
    """
    parameters = get_parameters()
    tmin = parameters["tmin"]
    tmax = parameters["tmax"]
    m = parameters["m"]
    l = parameters["l"]
    g = parameters["g"]
    torq_max = parameters["torq_max"]
    target = parameters["target"]

    def phi(pendulum_params: Dict, t: Array) -> Array:
        return Pendulum().apply({"params": pendulum_params}, t)[0]

    def tau(pendulum_params: Dict, t: Array) -> Array:
        return Pendulum().apply({"params": pendulum_params}, t)[1]

    def physics_loss(pendulum_params: Dict, ts: Array) -> Array:

        # computing derivatives
        phi_t = jax.jacfwd(phi,argnums=1)
        phi_tt = jax.jacfwd(phi_t,argnums=1)

        # PINN outputs
        phi_pred = jax.vmap(phi,(None,0),0)(pendulum_params,ts)
        phi_tt_pred = jax.vmap(phi_tt,(None,0),0)(pendulum_params,ts) # hessian with respect to time
        tau_pred = jax.vmap(tau,(None,0),0)(pendulum_params,ts)

        # reshape
        phi_tt_pred = jnp.reshape(phi_tt_pred,phi_pred.shape)

        # physics loss for domain point
        F_pred = (m*l*l*phi_tt_pred - (torq_max*jnp.tanh(tau_pred) - m*g*l*jnp.sin(phi_pred)))
        F = jnp.zeros_like(F_pred)

        return mse_loss_fn(F_pred,F)

    def constraints_loss(pendulum_params: Dict) -> Array:

        # define functions for gradient and hessian
        # phi_t = jax.grad(phi,argnums=1)
        phi_t = jax.jacfwd(phi,argnums=1)

        # PINN outputs
        # ic1 for phi
        phi_pred_tmin = phi(pendulum_params,jnp.array([tmin]))
        # Neumann ic3 for phi dot
        phi_t_pred_tmin = -1.0*phi_t(pendulum_params,jnp.array([tmin])) # gradient with respect to time
        # ic2 for tau
        tau_pred_tmin = tau(pendulum_params,jnp.array([tmin]))

        # reshape
        phi_pred_tmin = jnp.reshape(phi_pred_tmin,phi_pred_tmin.shape)

        phi_tmin = jnp.zeros_like(phi_pred_tmin)
        phi_t_tmin = jnp.zeros_like(phi_t_pred_tmin)
        tau_tmin = jnp.zeros_like(tau_pred_tmin)

        return mse_loss_fn(phi_pred_tmin,phi_tmin),  mse_loss_fn(tau_pred_tmin,tau_tmin), mse_loss_fn(phi_t_pred_tmin,phi_t_tmin)

    def goal_loss(pendulum_params: Dict) -> Array:
        # goal for optimization
        cos_phi_pred_tmax = jnp.cos(phi(pendulum_params,jnp.array([tmax])))
        cos_phi_tmax = jnp.array([target])
        return mse_loss_fn(cos_phi_pred_tmax,cos_phi_tmax)


    def loss_fn(pendulum_params: Dict, ts: Array) -> Tuple[Array, Array]:
        # remove boundary points
        L_phys = physics_loss(pendulum_params,ts)
        ic1,ic2,ic3 = constraints_loss(pendulum_params)
        L_goal = goal_loss(pendulum_params)
        L = L_phys + 10*ic1 + ic2 + ic3 + L_goal

        return L, [L_phys,ic1,ic2,ic3,L_goal]


    # compute loss and gradients with respect to the parameters of the neural network
    pendulum_params = state.params
    (loss, component_loss), grad = jax.value_and_grad(loss_fn, argnums=0,has_aux=True)(pendulum_params,ts)


    # optimize the neural network parameters with gradient descent
    state = state.apply_gradients(
        grads=grad
    )
    return state, loss, component_loss

def train_epoch(state: TrainState, ts: Array, batch_size: int, 
                epoch: int, rng: Array) -> Tuple[Dict[str, TrainState], float, Dict[str, float]]:
    ts_size = int(ts.shape[0])
    steps_per_epoch = ts_size // batch_size

    perms = jax.random.permutation(rng, ts_size)  # get a randomized index array
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape(
        (steps_per_epoch, batch_size)
    )  # index array, where each row is a batch
    batch_loss = []
    component_losses = []
    for perm in perms:
        batch = ts[perm]
        state, loss, component_loss = train_step(state, batch)
        component_losses.append(component_loss)
        batch_loss.append(loss)


    return state, batch_loss, min(batch_loss), component_losses

def run_pinn_training(rng: Array, ts: Array, num_epochs: int, batch_size: int):
    rng, init_train_states_rng = jax.random.split(rng, 2)

    # initialize the train states
    state = initialize_train_states(init_train_states_rng)

    # initialize the lists for the training history
    train_loss_history = []
    state_history = []
    component_losses_history = []
    
    print(f"Training the PINN for {num_epochs} epochs...")

    best_loss = 1e12
    best_state = state
    for epoch in range(1, num_epochs + 1):
        
        rng, epoch_rng = jax.random.split(rng,num=2)

        state, train_loss, min_loss, component_losses = train_epoch(state,ts,batch_size,epoch,epoch_rng)
        if min_loss < best_loss:
          best_loss = min_loss
          best_state = state

        train_loss_history.append(train_loss)
        component_losses_history.append(component_losses)
        state_history.append(state)
        print(f"Epoch {epoch}", jnp.array(component_losses).ravel())


    train_loss_history = jnp.array(train_loss_history)
    return train_loss_history, state_history, component_losses_history, best_loss, best_state

if __name__ == "__main__":
    rng = random.PRNGKey(seed=0)
    num_points = 1000
    num_epochs = 5000
    batch_size = num_points

    parameters = get_parameters()
    tmin = parameters["tmin"]
    tmax = parameters["tmax"]
    torq_max = parameters["torq_max"]
    # load the dataset
    ts = PointResampler(jnp.array([[tmin],[tmax]]),tmin,tmax).get_train_samples(num_points)
    ts = ts[2:] # remove the first two points for loss computation(boundary conditions)

    
    train_loss_history, state_history, component_losses_history, best_loss, best_state = run_pinn_training(rng,ts,num_epochs,batch_size)
    print("Best loss: ",best_loss)

    # plotting log loss
    plt.figure(figsize=(8, 4))
    plt.semilogy(train_loss_history.mean(axis=1))
    plt.title('Train Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # plotting pendulum position and torque
    plt.figure(figsize=(8, 4))
    t = jnp.linspace(tmin, tmax, 101)
    t = t.reshape(t.shape[0], 1)

    u = Pendulum().apply({"params": best_state.params}, t)

    # plotting pendulum position and torque
    plt.plot(t, u[:, 0], label='Position')
    plt.plot(t, torq_max*jnp.tanh(u[:, 1]), label='Torque')
    plt.title('Pendulum Position and Torque')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
