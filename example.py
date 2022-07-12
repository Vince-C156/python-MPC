from utils import get_cartpole_dynamics
from MPC import MPC
import costs
import numpy as np
import cvxpy as cp
#testing pipeline
#(self, x : np.array, n_actions : int, dynamics : object, loss_function):

"""
Creating an instance of dynamics object with cartpole equations
"""
dynamics = get_cartpole_dynamics()

"""
Defining inital state starting values in x0

u_ranges is a list of tuples that each define the boundary of respective control action.
"""
x0 = np.array([0., 0., 0.1, 0.])
xr = np.array([5., 0., 0., 0.])
u_ranges = [(-3., 3.)]
"""
Defining a weight matrix Q to prioritize certain state corrections depending on the task.

Defining a R matrix which weighs the cost of any actuation
"""
Q = np.array([[20., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 40., 0.],
              [0., 0., 0., 1.]])


R = np.array([[1.]])
#R = 100.
print(Q.shape)
#LQR_obj = costs.LQR(x0, u_ranges, dynamics, Q=Q, R=np.array([[10.]]) )

"""
Define objective function

X = cp.Variable((states, T + 1))
U = cp.Variable((actions, T))


(expression, constraints, T window)

expression = 

self, x : np.array, n_actions : int, dynamics):
mpc.minimize_objective(n_actions, n_states, Q, R, T)
"""

mpc_model = MPC(x = x0, u_range = u_ranges, dynamics = dynamics)
mpc_model.jacobian_linearization()
mpc_model.minimize_objective(4, 1, Q, R, xr, 15)
optimal_u = mpc_model.get_action()
print(optimal_u)

#mpc_model.loss_function.spec()
#mpc_model.optimize_convex(4)
#optimal_u = mpc_model.get_action()
