import numpy as np
import jaxlib
import jax
import jax.numpy as jnp
from math import sin, cos
from matplotlib import pyplot as plt
import itertools
import scipy
import cvxpy as cp

class costs:

    list_all = ['LQR']
    def __init__(self):
        print("INITALIZED COSTS")

    def compare():
        pass



class LQR(costs):

    def __init__(self, x, u_range, dynamics, Q=None, R=None):

        for var in [x]:
            assert type(var) == np.ndarray, 'array types need to be ndarray'

        super().__init__()

        if type(Q) != np.ndarray:
            print('DEFAULTING Q TO IDENTITY MATRIX')
            self.Q = np.identity(x.shape[0])
        else:
            self.Q = Q

        self.x = x
        self.u_range = u_range
        self.R = R
        self.dynamics = dynamics

        #self.domain_actions = []

        if dynamics.A == None:
            print("No exisiting A and B of lineraization")
            print("Defaulting to initalize values now")
            dynamics.fixedptn_linearization()
            print('Matrix A')
            print(dynamics.A)
            print()
            print('Matrix B')
            print(dynamics.B)
        """
        self.U = cp.Variable(4)
        self.function = cp.Problem(cp.Minimize(cp.quad_form(xdot, Q) + cp.quadform(self.U, R)),
                                  [self.dynamics.A @ self.x + self.dynamics.B @ self.U == xdot,
                                   self.U >= -3., self.U <= 3.])
        """
        w, T = jnp.linalg.eig(dynamics.A)
        TA = jnp.matmul(T, dynamics.A)
        D = jnp.matmul(TA, jnp.linalg.inv(T))
        D_diag = jnp.real(jnp.diagonal(D))

        self.T = T
        self.D = D
        self.D_diag_real = D_diag
        print(f'eign values {w}')
        print(f'eign vec \n {T}')
        print(f'diagonal eign val {D_diag}')

    def step_cost(self, u):

        assert type(self.R) != None, 'No R value'

        K = self.find_K(p = np.array([-1.0, -1.1, -1.2, -1.3]))
        u = self.x * (K * -1)

        x_transQ = jnp.dot(self.x.T, self.Q)
        u_transR = u.T * self.R

        x_transQx = jnp.dot(x_transQ, self.x)
        u_transRu = jnp.dot(u_transR, u)

        J = x_transQx + u_transRu
        J = max(J[0])

        print('LOSS VAL IS :', J)
        return J
    
    def find_K(self, p: np.array):
        """
        function to find control input K assuming u = -kx
        
        x_dot = (A-BK)*x
        """
        system = self.dynamics

        M = scipy.signal.place_poles(system.A, system.B, p)
        K = M.gain_matrix
        
        print('ORIGINAL K', K)
        print('K MULT -1', K*-1)
        temp = system.A-(system.B*K)
        u = system.x0 * (K*-1.)
        print('K MATRIX', K)
        print('u as a result of -kx', u)
        print('(A-B*K) eign values', jnp.linalg.eig(temp)[0] )
        return K

    def update_dynamics(self, u):
        system = self.dynamics
        A = system.A
        B = system.B
        x = system.x0
        u = u

        Ax = A*x
        Bu = B*u
        """
        Ax = np.dot(A, x)
        Bu = np.dot(B, u)
        """
        x_dot = Ax + Bu
        print(f'XDOT {x_dot}')
        #self.dynamics.u0 = u
        self.dynamics.x0 = x_dot.T
        print("NEW X0", self.dynamics.x0)
        self.dynamics.fixedptn_linearization()

    def min(self, T):
        cost = 0
        actions = 1
        states = 4
        
        x = cp.Variable((states, T + 1))
        U = cp.Variable((actions, T))
        print("LQR X", self.x)
        print("=======================")
        print("LQR U", U)
        print("LQR R", self.R)
        constraints = []
        print(x[0])
        for t in range(T):
            cost += cp.quad_form(x[:, t + 1], self.Q) + cp.quad_form(U[:, t], self.R)
            #cost += cp.quad_form(U[:, t], self.R)
            constraints += [x[:, t + 1] == self.dynamics.A @ x[:, t] + self.dynamics.B @ U[:, t], U >= -3., U <= 3.]
        term_constr = self.dynamics.x0.T
        print(term_constr[0])
        constraints += [x[:, 0] == self.x.T]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        result = problem.solve()
        policy = U.value
        simulated_states = x.value
        print("POLICY :", policy)
        
        print("state evolution" , simulated_states.T)
        
        return policy

    def spec(self):
        print("======================")
        print("LINEAR SYSTEM")
        print("----------------------")
        print("A=")
        print(self.dynamics.A, '\n')
        print("x=")
        print(self.x, '\n')
        print("B=")
        print(self.dynamics.B, '\n')
        print("u=")
        print(self.u)
        print("======================\n")
        print("======================")
        print("EIGEN VALUES AND VECTORS")
        print("-----------------------")
        print("D=")
        print(self.D, '\n')
        print("D diagonal only reals=")
        print(self.D_diag_real, '\n')
        print("Eigen vectors or T =")
        print(self.T)
        print("==========================")
        print(self.domain_actions)
