import costs
import numpy as np
from math import pi as pi_const
from utils import get_cartpole_dynamics
x = np.array([1., 0., 30., 0.])
u = np.array([0.])
Q = np.array([[1., 0., 0., 0.],
              [0., 5., 0., 0.],
              [0., 0., 10., 0.],
              [0., 0., 0., 10.]])
mydynamics = get_cartpole_dynamics()
myLQR = costs.LQR(x, u, mydynamics, Q=Q, R=np.array([0.1]))
myLQR.spec()

path = myLQR.cost2go(3)
print(path)


