import numpy as np
import jax
import jaxlib
from jax import jacfwd, vmap, grad, lax
import math
from math import sin, cos
from math import pi as pi_const
from inspect import signature
from utils import dynamics
from MPC import MPC

def f1(x1, x2, x3, x4, u1=1.):
    return x2

def f2(x1, x2, x3, x4, u=1.):
    #F = u
    g = 9.8
    M = 1.0
    m = 0.1
    l = 0.5
    L = m * l
    Fm = 10.0
    sm = M+m

    return (u / (m *(sm + L * lax.sin(x3) ** 2.0))) * (-x4**2.0*L*lax.sin(x3) - g*lax.cos(x3)-lax.sin(x3)) 


def f3(x1, x2, x3, x4, u=1.):
    return x4

def f4(x1, x2, x3, x4, u=1.):
    #F = u
    g = 9.8
    M = 1.0
    m = 0.1
    l = 0.5
    L = m * l
    Fm = 10.0
    sm = M+m

    return ( -u / (m*l*(sm + lax.sin(x3)**2)) * (lax.cos(x3) - x4**2*L*lax.sin(x3)*lax.cos(x3)) + (1 + sm)*(g*lax.sin(x3)) )

def main():
    print(signature(dynamics))
    x0=np.asarray([0., 0., pi_const, 0.])
    u0=np.asarray([0.])
    x1, x2, x3, x4 = x0[0], x0[1], x0[2], x0[3]
    cnst = {'F' : 1.0, 'g' : 9.8, 'M' : 1.0, 'm' : 0.1, 'l' : 0.5, 'Fm' : 10.0}
    state_def = {'pos' : x1, 'pos_dot' : x2, 'theta' : x3, 'theta_dot' : x4}
    myDynamics = dynamics(state_def, cnst, x0, u0, fn1=f1, fn2=f2, fn3=f3, fn4=f4)
    myMPC = MPC(x0, 1, myDynamics, Q=None)

    myMPC.jacobian_linearization()
    #print(signature(MPC()))

if __name__ == "__main__":
    main()
