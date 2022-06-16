import numpy as np
import jax
from jax import grad, lax
from math import pi as pi_const

class dynamics:

    def __init__(self, x_mapping : dict, cnst : dict, x0=np.array([]), u0=np.array([]), **system_equations):
        print("Initalizing dynamics object")
        self.state_mapping = x_mapping
        self.x0 = x0.reshape(4,1)
        self.cnst = cnst
        self.u0 = u0.T
        self.system_mapping = system_equations

        self.A = None
        self.B = None
        self.C = None
        self.D = None

    def fixedptn_search():
        pass
    def fixedptn_linearization(self):
        
        n = len(self.system_mapping.values())
        #J = np.arrange(n**n).reshape(n,n)
        jac_A = []
        jac_B = []
        x1, x2, x3, x4 = self.x0[0][0], self.x0[1][0], self.x0[2][0], self.x0[3][0]
        print(f'x1:{x1}, x2:{x2}, x3:{x3}, x4:{x4}')
        u_list = [u for u in self.u0]
        for function in self.system_mapping.values():
            jac_row = np.array([[dfdx for dfdx in grad(function, (0, 1, 2, 3))(x1,x2,x3,x4)]])
            jac_A.append(jac_row)

        for idx, function in enumerate(self.system_mapping.values()):
            jac_row = grad(function, 4)(x1, x2, x3, x4, u_list[0])
            jac_B.append(jac_row)
            """
            for u in u_list:
                dfdu = jac_B.append(grad(function, 4)(x1, x2, x3, x4, u1))
            """
       
        """
        df2dx1, df2dx2, df2dx3, df2dx4 = grad(f2, (0, 1, 2, 3))(x1,x2,x3,x4)
        df3dx1, df3dx2, df3dx3, df3dx4 = grad(f3, (0, 1, 2, 3))(x1,x2,x3,x4)
        df4dx1, df4dx2, df4dx3, df4dx4 = grad(f4, (0, 1, 2, 3))(x1,x2,x3,x4)
        """
        jac_A = np.reshape(jac_A, (4,4))
        jac_B = np.asarray([jac_B]).T

        print("A")
        print("=====================================")
        print(jac_A)
        print("=====================================")
        print("B")
        print(jac_B)

        self.A = jac_A
        self.B = jac_B
        print("UPDATED A AND B SYSTEM MATRICES")



def get_cartpole_dynamics():

    def f1(x1, x2, x3, x4, u=0.):
        return x2

    def f2(x1, x2, x3, x4, u=0.):
        #F = u
        g = 9.8
        M = 3.0
        m = 1.0
        l = 0.5
        L = 2.0
        Fm = 10.0
        #sm = M+m
        sm = M / m
        return (1. / (sm + L*lax.sin(x3)**2.0) ) * (  (u/m)-(x4**2.0*L*lax.sin(x3))-(g*lax.cos(x3)*lax.sin(x3)) )
        #return (u / (m *(sm + L * lax.sin(x3) ** 2.0))) * (-x4**2.0*L*lax.sin(x3) - g*lax.cos(x3)-lax.sin(x3)) 


    def f3(x1, x2, x3, x4, u=0.):
        return x4

    def f4(x1, x2, x3, x4, u=0.):
        #F = u
        g = 9.8
        M = 3.0
        m = 1.0
        l = 0.5
        L = 2.0
        Fm = 10.0
        #sm = M+m
        sm = M / m

        return (1. / L*(sm+lax.sin(x3)**2.0)) * (  ((-1.0*u/m)*lax.cos(x3)) - (x4**2.0*L*lax.sin(x3)*lax.cos(x3)) + ( (1. + sm)*g*lax.sin(x3) )  )
        #return ( -u / (m*l*(sm + lax.sin(x3)**2)) * (lax.cos(x3) - x4**2*L*lax.sin(x3)*lax.cos(x3)) + (1 + sm)*(g*lax.sin(x3)) )


    x0=np.asarray([1., 0., 0., 0.])
    u0=np.asarray([0.])
    x1, x2, x3, x4 = x0[0], x0[1], x0[2], x0[3]
    cnst = {'F' : 1.0, 'g' : 9.8, 'M' : 1.0, 'm' : 0.1, 'l' : 0.5, 'Fm' : 10.0}
    state_def = {'pos' : x1, 'pos_dot' : x2, 'theta' : x3, 'theta_dot' : x4}
    myDynamics = dynamics(state_def, cnst, x0, u0, fn1=f1, fn2=f2, fn3=f3, fn4=f4)


    return myDynamics
