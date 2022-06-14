import numpy as np
import jax
from jax import grad, lax

class dynamics:

    def __init__(self, x_mapping : dict, cnst : dict, x0=np.array([]), u0=np.array([]), **system_equations):
        print("Initalizing dynamics object")
        self.state_mapping = x_mapping
        self.x0 = x0
        self.cnst = cnst
        self.u0 = u0
        self.system_mapping = system_equations

    def fixedptn_search():
        pass
    def fixedptn_linearization(self):
        
        n = len(self.system_mapping.values())
        #J = np.arrange(n**n).reshape(n,n)
        jac_A = []
        jac_B = []
        x1, x2, x3, x4 = self.x0[0], self.x0[1], self.x0[2], self.x0[3]
        u1, u2, u3, u4 = self.u0[0], self.u0[1], self.u0[2], self.u0[3]
        for function in self.system_mapping.values():
            jac_row = np.array([[dfdx for dfdx in grad(function, (0, 1, 2, 3))(x1,x2,x3,x4)]])
            jac_A.append(jac_row)

        for idx, function in enumerate(self.system_mapping.values()):
            jac_B.append(grad(function, 4)(x1, x2, x3, x4, u1))

       
        """
        df2dx1, df2dx2, df2dx3, df2dx4 = grad(f2, (0, 1, 2, 3))(x1,x2,x3,x4)
        df3dx1, df3dx2, df3dx3, df3dx4 = grad(f3, (0, 1, 2, 3))(x1,x2,x3,x4)
        df4dx1, df4dx2, df4dx3, df4dx4 = grad(f4, (0, 1, 2, 3))(x1,x2,x3,x4)
        """
        jac_A = np.reshape(jac_A, (4,4))
        jac_B = np.asarray(jac_B)

        print("A")
        print("=====================================")
        print(jac_A)
        print("=====================================")
        print("B")
        print(jac_B)
