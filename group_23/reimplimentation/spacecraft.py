
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

n_adam = 2000

LR = 1e-3 # for Adam

def get_parameters():
    parameters = {
        "tmin": 0.,
        "tmax" : 1.,  
        "bh_xygm": [
            [-0.5, -1.0, 0.5],
            [-0.2, 0.4, 1.0],
            [0.8, 0.3, 0.5],
        ],
        "x0": -1.,
        "x1": 1.,
        "y0" : -1.,
        "y1" : 1.,
        "m0" : 1.
    }
    return parameters
     # normalized time
    # each row corresponds to a celestial body - (x, y, gravitational constant)

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

class SpaceCraftNN(nn.Module):
    """
    Physics Informed Neural network for the spacecraft task 
    """
    num_hidden: int = 64  # Number of hidden units per intermediate layer
    num_outputs: int = 2 # x, u

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
        model_train_state: the current state of the training of the neural network.
    """
    # initialize the neural network object
    model = SpaceCraftNN()

    # initialize parameters of the neural networks by passing a dummy input through the network
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (1,))  # dummy input
    model_params = model.init(init_rng, inp)['params']

    # initialize the adam optimizer for the neural network
    model_tx = optax.adam(learning_rate=LR)

    # create the TrainState object for the neural network
    model_train_state = TrainState.create(
        apply_fn=model.apply,
        params=model_params,
        tx=model_tx,
    )
    return model_train_state

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
    params = get_parameters()
    # tmin = params["tmin"]
    # tmax = params["tmax"]
    bh_xygm = params["bh_xygm"]
    # x0 = params["x0"]
    # x1 = params["x1"]
    # y0 = params["y0"]
    # y1 = params["y1"]
    m0 = params["m0"]

    def x(model_params: Dict, t: Array) -> Array:
      return SpaceCraftNN().apply({"params": model_params}, t)[0]

    def y(model_params: Dict, t: Array) -> Array:
      return SpaceCraftNN().apply({"params": model_params}, t)[1]

    def physics_loss(model_params: Dict, ts: Array, T) -> Array:
        # computing derivatives
        x_TT = jax.hessian(x, argnums=1) # d2x/dt2
        y_TT = jax.hessian(y, argnums=1) # d2x/dt2
        loss_x = m0 * x_TT
        loss_y = m0 * y_TT

        x_TT_pred = jax.vmap(x_TT, (None, 0), 0)(model_params, ts) / T
        y_TT_pred = jax.vmap(y_TT, (None, 0), 0)(model_params, ts) / T

        # return mse_loss_fn(F_pred,F)

    def constraints_loss(model_params: Dict) -> Array:
       pass