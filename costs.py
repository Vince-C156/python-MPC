import numpy as np
import jaxlib
import jax
import jax.numpy as jnp
from math import sin, cos
from matplotlib import pyplot as plt
import itertools
import scipy

class costs:

    list_all = ['LQR']
    def __init__(self):
        print("INITALIZED COSTS")

    def compare():
        pass



class LQR(costs):

    def __init__(self, x, u, dynamics, Q=None, R=None):

        for var in [x, u]:
            assert type(var) == np.ndarray, 'array types need to be ndarray'

        super().__init__()

        if type(Q) != np.ndarray:
            print('DEFAULTING Q TO IDENTITY MATRIX')
            self.Q = np.identity(x.shape[0])
        else:
            self.Q = Q

        self.x = dynamics.x0
        self.u = dynamics.u0
        self.R = R
        self.dynamics = dynamics

        self.domain_actions = [np.asarray(i) for i in itertools.product([0.,1.], repeat=len(self.u) )]

        if dynamics.A == None:
            print("No exisiting A and B of lineraization")
            print("Defaulting to initalize values now")
            dynamics.fixedptn_linearization()
            print('Matrix A')
            print(dynamics.A)
            print()
            print('Matrix B')
            print(dynamics.B)

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
        print('u', u)
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

    def cost2go(self, t):
        if t == 0:
            return []
        
        all_costs = [[u, self.step_cost(u)] for u in self.domain_actions]
        minimum = min(all_costs, key = lambda pair : pair[1])
        u_dot = minimum[0].T
        self.update_dynamics(u_dot)
        return self.cost2go(t-1).append(minimum)

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
