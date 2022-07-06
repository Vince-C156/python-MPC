import numpy as np
import jaxlib
import jax
from jax import grad, jit, vmap, jacfwd, lax, random, vjp
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
from math import sin, pi, cos, pow, degrees, radians
from jax import jacrev as builtin_jacrev
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False
"""
MPC class
"""
class MPC:
    """
    functions:

    jacobian_linearization :

    pred_state :

    calc_MHE :

    optimize_convex :

    """
    def __init__(self, x : np.array, n_actions : int, dynamics : object, loss_function):
        #MPC( x=[pos, theta], dx=[dpos, dtheta], n_actions=2, system_equations=None, loss_function=None, A_raw= )
        print("CREATED MPC INSTANCE")
        """
        These memebers, system_equations and loss_function, handle the MPC's model dynamics
        and its optimization goals.

        dynamics : class initalized from utils class which represents the dynamics of a system.
        loss_function : a function which quantifies feedback based on action input

        """
        self.dynamics = dynamics
        self.loss_function = loss_function
        
        """
        The class members below represent the linearized system of equatoins

        x_dot : a list of the derivatives of each state variable
        x : a list of the variables representing the model's current state.
        u : Vector used to store the MPC's calculated control action to stabalize the system.
        n_actions : number of possible actions the model can preform

        cost : user initalized cost object from costs.py which outputs a control sequence u = -kx
        """
        self.x = x
        self.u = np.zeros((n_actions, 1))
        #self.A, self.B, self.C, self.D = None, None, None, None
        self.cost = loss_function
        #setting Bdot to a column vector of ones with the same shape as u if user does not specify

        """
        A boolean which is used to toggle between making t+1 predictions using conventional equation based modeling or
        model dynamics learned through a machine learning model.

        isdynamics_traditional : When true, the MPC tries to solve the dynamics of the model through the jacobian_linearization function to find predictions in t+1. When false the MPC
        uses a loaded reinforcement learning model that makes t+1 predictions through methods like the bellman equations
        """
        self.isdynamics_traditional = True
        
    def jacobian_linearization(self):
        """
        Finds constant matrix A, B, C, D for a linear system
        """

        self.dynamics.fixedptn_linearization()
  
    def set_cost(self, cost_obj):
        """
        Takes in a cost object from costs.py and sets self.cost to that instance.
        """

        self.cost = cost_obj


    def optimize_convex(self, T):
        """
        finds control sequence policy u = -kx T time steps ahead based on cost function set and sets 
        self.u to that sequence
        """
        assert self.cost != None, 'No cost object instance set'
        self.cost.x = self.x
        self.u = self.cost.min(T)
        return self.u

    def get_action(self):
        """
        returns calculated actuation for the most current timestamp
        """
        u = jnp.squeeze(self.u)
        print("RETURNING", u[0])
        return u[0]
        
    def calc_feasibility():
        pass
    
    def calc_controllability():
        pass
 
