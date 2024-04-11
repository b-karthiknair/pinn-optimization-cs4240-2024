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
from typing import Dict, Tuple, List
import skopt

LR = 0.001
# get physical parameters
def get_parameters():
    parameters = {
        "x0":0.0,
        "x1":1.0,
        "y0":1.0,
        "y1":0.0,
        "g":9.8,
        "tmin":0.0,
        "tmax":1.0
    }
    return parameters

class PointResampler:
  def __init__(self, tmin, tmax):
    assert tmin <= tmax, "tmin should be lesser than tmax"
    self.tmin, self.tmax = tmin,tmax
    self.diam = tmax - tmin
    self.sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
    # initializing the sampler
    self.period = 100
    self.space = [(0.0, 1.0)]
    self.epoch_counter = 0

  def get_train_samples(self, n_samples):
    if self.epoch_counter % self.period == 0:
      self.epoch_counter = 0
      x = np.asarray(self.sampler.generate(self.space,
      n_samples)[0:])
      self.train_x = self.tmin + self.diam * x
      self.epoch_counter += 1
    
    return self.train_x

class Snell_NN(nn.Module):
    """
    Physics Informed Neural network for the Snell's law
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
        hidden_layers = [self.num_hidden] * 3
        activation = nn.tanh
        kernel_init = jax.nn.initializers.glorot_normal()
        output_activation=nn.sigmoid

        # implement the neural network layers
        layers = []
        # hidden layers
        for hidden_units in hidden_layers:
            layers.append(nn.Dense(hidden_units, kernel_init=kernel_init))
            layers.append(activation)
        # output layer
        layers.append(nn.Dense(self.num_outputs))
        layers.append(output_activation)
        
        u = t
        for layer in layers:
            u = layer(u)
        return u
    
@jit
def mse_loss_fn(pred: Array, target: Array) -> Array:
    loss = jnp.mean((pred - target)**2)
    return loss

def update_time(time:Array) -> Array:
    point_resampler=PointResampler(tmin,tmax)
    time_new=point_resampler.get_train_samples(time.shape[0])
    return time_new
def initialize_train_states(rng: Array, learning_rate: float,T:Array) -> TrainState:
    """
    Initialize the train state of the neural network.
    Args:
        rng: PRNG key for pseudo-random number generation.
    Returns:
        snell_train_state: the current state of the training of the neural network.
    """
    # initialize the neural network object
    model = Snell_NN()

    # initialize parameters of the neural networks by passing a dummy input through the network
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (1,))  # dummy input
    snell_params = model.init(init_rng, inp)['params']
    # initialize the adam optimizer for the neural network
    snell_tx = optax.adam(learning_rate=learning_rate)
    snell_params['T']=T
    # create the TrainState object for the neural network
    snell_train_state = TrainState.create(
        apply_fn=model.apply,
        params=snell_params,
        tx=snell_tx,
    )

    return snell_train_state

@jit
def train_step(state: TrainState, t: Array) -> Tuple[TrainState, Array, Array]:
    parameters = get_parameters()
    x0 = parameters["x0"]
    x1 = parameters["x1"]
    y0 = parameters["y0"]
    y1 = parameters["y1"]
    g = parameters["g"]
    #tmin = parameters["tmin"]
    #tmax = parameters["tmax"]

    # defining functions to get x, y, max time, and to compute the losses
    def x_position(params: Dict, t: Array) -> Array:
        return Snell_NN().apply({"params": params}, t)[0]
    
    def y_position(params: Dict, t: Array) -> Array:
        return Snell_NN().apply({"params": params}, t)[1]

    """def find_max_time(params:Dict,t:Array)-> Array:
        t_max_step=jnp.max(t)
        x_pos=jax.vmap(x_position,(None,0),0)(params,t)
        y_pos=jax.vmap(y_position,(None,0),0)(params,t)
        max_x_value = jnp.max(x_pos)
        max_y_value = jnp.max(y_pos)
        max_index = jnp.argmax(jnp.logical_and(x_pos == max_x_value, y_pos == max_y_value))
        t_max_step=t[max_index]
        return t_max_step"""

    def physics_loss_fn(params: Dict,t:Array) -> Array:
        x_t = jax.grad(x_position,argnums=1)
        total_time = params['T']
        y_t = jax.grad(y_position,argnums = 1)
        y_pos = jax.vmap(y_position,(None,0),0)(params,t)
        x_t_pred = jax.vmap(x_t,(None,0),0)(params,t)
        y_t_pred=jax.vmap(y_t,(None,0),0)(params,t)

        ones_arr=jnp.ones_like(t)
        g_y0=(g*y0*ones_arr)
        g_y=g*y_pos
        F_pred=g_y0-(g_y+0.5*((x_t_pred/total_time)**2+ (y_t_pred/total_time)**2)) 
        F = jnp.zeros_like(F_pred)
        return mse_loss_fn(F_pred, F)

    def constraint_loss_fn(params: Dict) -> Array:
        x_pos_pred_tmin = x_position(params,jnp.array([tmin]))
        y_pos_pred_tmin = y_position(params,jnp.array([tmin]))
        x_pos_pred_tmax = x_position(params,jnp.array([tmax]))
        y_pos_pred_tmax = y_position(params,jnp.array([tmax]))
        # converting to jnp arrays
        x_tmin, y_tmin, x_tmax, y_tmax = jnp.array([x0]), jnp.array([y0]), jnp.array([x1]), jnp.array([y1])
        # computing individual losses
        l1 = mse_loss_fn(x_pos_pred_tmin,x_tmin)
        l2 = mse_loss_fn(x_pos_pred_tmax,x_tmax) 
        l3 = mse_loss_fn(y_pos_pred_tmin,y_tmin)
        l4 = mse_loss_fn(y_pos_pred_tmax,y_tmax)
        constraint_loss = l1 + l2 + l3 + l4
        return constraint_loss 

    def goal_loss_fn(params:Dict) -> Array:
        #t_target=jnp.zeros_like(params['T'])
        return params['T']
    
    def loss_fn(params: Dict,t:Array) -> Tuple[Array, Array]:
        L_phy = physics_loss_fn(params,t)
        L_cons = constraint_loss_fn(params)
        L_goal = goal_loss_fn(params)

        w_phy , w_cons, w_goal= 1.0, 1.0, 0.01
        L = w_phy*L_phy + w_cons*L_cons + w_goal*L_goal
        
        return L,[L_phy,L_cons,L_goal]
    
    (loss, component_loss), grad = jax.value_and_grad(loss_fn, argnums=0,has_aux=True)(state.params,t)
    state = state.apply_gradients(grads=grad)
    return state, loss,component_loss

def train_epoch(
        state: TrainState,
        ts: Array,
        batch_size: int,
        epoch: int,
        rng: Array
    ) -> Tuple[TrainState,Array,Array]:

    ts_size = int(ts.shape[0])
    perms = jax.random.permutation(rng, ts_size)  # get a randomized index array
    batch_loss = []
    batch = ts[perms]
    state, loss,component_loss = train_step(state, batch)
    batch_loss.append(loss)
    return state, batch_loss,component_loss

def run_pinn_training(
        rng: Array,
        ts: Array,
        num_epochs: int,
        batch_size: int,
        verbose: bool = True
    ) -> Tuple[Array,List[TrainState],Array]:
    # number of training samples
    # num_train_samples = len(ts)
    # ts_size = int(ts.shape[0])
    # steps_per_epoch = ts_size // batch_size
    rng, init_train_states_rng = jax.random.split(rng, 2)
    # initialize the train states
    T=jnp.max(ts)
    state = initialize_train_states(init_train_states_rng, LR,T)

    # initialize the lists for the training history
    train_loss_history = []
    state_history = []
    component_loss_history=[]
    if verbose:
        print(f"Training the PINN for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        rng, epoch_rng, sample_rng,initialiser_rng = jax.random.split(rng, num=4)
        #ts = jax.random.uniform(sample_rng,ts.shape,minval=tmin+epsilon,maxval=tmax)
        #ts_endpoint=jax.random.uniform(initialiser_rng,(1,))
        # ts=point_resampler.get_train_samples(ts.shape)
        state, train_loss,component_loss = train_epoch(state,ts,batch_size,epoch,epoch_rng)
        train_loss_history.append(train_loss)
        state_history.append(state)
        component_loss_history.append(component_loss)
        ts=update_time(ts)

    train_loss_history = jnp.array(train_loss_history)

    return train_loss_history, state_history,component_loss_history

if __name__ == "__main__":
    parameters = get_parameters()
    tmin = parameters["tmin"]
    tmax = parameters["tmax"]

    # initialize the PRNG key at seed=0
    rng = random.PRNGKey(seed=0)
    num_points = 1000
    # load the dataset
    ts = jnp.linspace(tmin, tmax, num_points)
    # load the dataset
    ts = PointResampler(tmin,tmax).get_train_samples(num_points)
    ts = ts[2:] # remove the first two points for loss computation(boundary conditions)
    ts = ts.reshape(ts.shape[0],1)

    num_epochs = 3000
    batch_size = 1000
    train_loss_history, state_history,component_loss_history = run_pinn_training(rng,ts,num_epochs,batch_size)
    print(component_loss_history[-1])
    # plotting log loss
    plt.figure(figsize=(8, 4))
    plt.plot(jnp.log(train_loss_history.mean(axis=1)))
    plt.title('Log of Train Loss History')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.show()

    # plotting pendulum position and torque
    plt.figure(figsize=(8, 4))
    t = jnp.linspace(tmin, tmax, 10000)
    t = t.reshape(t.shape[0], 1)

    t = jnp.linspace(tmin, tmax, 101)
    t = t.reshape(t.shape[0],1)
    state = state_history[-1]
    u = Snell_NN().apply({"params": state.params}, t)
    r = 0.5729
    theta = np.linspace(0, np.arccos(1 - 1 / r), 101)
    xx = r * (theta - np.sin(theta))
    yy = 1 - r * (1 - np.cos(theta))
    
    plt.plot(u[:,0],u[:,1], 'b')
    plt.plot(xx, yy, 'k--')
    plt.show()
