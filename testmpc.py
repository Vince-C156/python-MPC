from utils import get_cartpole_dynamics
from MPC import MPC
import costs
import numpy as np

#testing pipeline
#(self, x : np.array, n_actions : int, dynamics : object, loss_function):
dynamics = get_cartpole_dynamics()
x = np.array([1., 1., 30., 2.])
u = [(-3., 3.)]
Q = np.array([[1., 0., 0., 0.],
              [0., 5., 0., 0.],
              [0., 0., 100., 0.],
              [0., 0., 0., 10.]])

print(Q.shape)
LQR_obj = costs.LQR(x, u, dynamics, Q=Q, R=np.array([[10.]]) )

mpc_model = MPC(x = x, n_actions = 1, dynamics = dynamics, loss_function = LQR_obj)
mpc_model.jacobian_linearization()
#mpc_model.loss_function.spec()
mpc_model.optimize_convex(4)
optimal_u = mpc_model.get_action()
