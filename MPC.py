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
import cvxpy as cp
import gurobipy as gb

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

    J. L. Junkins, M. R. Akella, and R. D. Robinett, “Nonlinear adaptive control of spacecraft maneuvers,”
Journal of Guidance, Control, and Dynamics, vol. 20, no. 6, pp. 1104–1110, 1997.

    optimize_convex :

    """
    def __init__(self, x : np.array, u_range : list, dynamics):
        #MPC( x=[pos, theta], dx=[dpos, dtheta], n_actions=2, system_equations=None, loss_function=None, A_raw= )
        print("CREATED MPC INSTANCE")
        """
        These memebers, system_equations and loss_function, handle the MPC's model dynamics
        and its optimization goals.

        dynamics : class initalized from utils class which represents the dynamics of a system.
        loss_function : a function which quantifies feedback based on action input

        """
        self.dynamics = dynamics
        self.Q = None
        self.R = None
        self.objective = None
        self.control_law = None
        self.u_range = u_range
        """
        The class members below represent the linearized system of equatoins

        x_dot : a list of the derivatives of each state variable
        x : a list of the variables representing the model's current state.
        u : Vector used to store the MPC's calculated control action to stabalize the system.
        n_actions : number of possible actions the model can preform

        cost : user initalized cost object from costs.py which outputs a control sequence u = -kx
        """
        self.x = x
        #self.u = np.zeros((n_actions, 1))
        #self.A, self.B, self.C, self.D = None, None, None, None
        #self.cost = loss_function
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

    def minimize_objective(self, n_states, n_actions, Q, R, xr, T):
        """
        """
        cost = 0
        lbu0, ubu0 = self.u_range[0][0], self.u_range[0][1]
        X = cp.Variable((n_states, T + 1))
        U = cp.Variable((n_actions, T))
        print("Q MATRIX", Q)
        print("=======================")
        print("R MATRIX", R)
        constraints = []
        print("SELF X", self.x)
        x0 = self.x.T
        #x0 = np.expand_dims(self.x, axis=0)
        print("x0", x0)
        for t in range(T):
            cost += cp.quad_form(X[:, t + 1] - xr, Q) + cp.quad_form(U[:, t], R)
            #cost += cp.quad_form(X[:, t+1], Q)
            constraints += [X[:, t + 1] == self.dynamics.A @ X[:, t] + self.dynamics.B @ U[:, t], U[:, t] >= lbu0, U[:, t] <= ubu0]
        #U >= -3., U <= 3., cp.norm(U[:, t], "inf") <= 1]
        #term_constr = self.dynamics.x0.T
        #print(term_constr[0])
        constraints += [X[:, 0] == self.x, X[:, T] <= xr]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        #env = gb.Env("testenv.log")
        #env.setParam("SolutionNumber" , 1)
        #env.setParam("PoolSearchMode", 2)
        #gurobi_params = {"SolutionNumber" : 1}
        #result = problem.solve(solver=cp.OSQP, max_iter=500, verbose=True)
        result = problem.solve(solver=cp.GUROBI, verbose=True)

        if problem.status not in ["infeasible", "unbounded"]:
            print("Optimal value: %s" % problem.value)
            for variable in problem.variables():
                print("Variable %s: value %s" % (variable.name(), variable.value))
        control_law = U.value
        simulated_states = X.value
        print("Control law:", control_law.astype(np.float32))

        print("state evolution: \n" , simulated_states.astype(np.float32).T)

        self.control_law = control_law

        return control_law
 
        """
        constraints = []
        for t in range(T):
            cost += expression
            constraints += constr
        #constraints += [X[:, 0] == self.x.T]
        problem = cp.Problem(cp.minimize(cost), constraints)
        problem.param_dict['x0'] = self.x
        proble.solve()
        """

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
        u = jnp.squeeze(self.control_law)
        print("RETURNING", u[0])
        return u[0]
        
    def calc_feasibility():
        pass
    
    def calc_controllability():
        pass
 
