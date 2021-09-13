import numpy as np


class ShapeFunction:
    """Class which contains functions to calculate the N matrix and the B matrix
    Inputs:
    1) (self.xi, self.eta) = natural coordinates
    2) vertex_coord = coordinates of the vertex points
    3) el_type = type of element used in meshing
       el_type = 0 (4 noded quadrilateral)
       el_type = 1 (8 noded quadrilateral)"""

    def __init__(self, xi, eta, el_type):
        self.xi = xi
        self.eta = eta
        self.el_type = el_type
        if self.el_type == 0:
            self.N = np.zeros((2, 8))
        elif self.el_type == 1:
            self.N = np.zeros((2, 16))
        self.J = np.zeros((2, 2))
        if self.el_type == 0:
            self.dpN = np.zeros((2, 4))
        elif self.el_type == 1:
            self.dpN = np.zeros((2, 8))

    def get_N_matrix(self):
        """Calculate the N matrix"""

        if self.el_type == 0:
            # shape functions for 4 noded quadrilateral
            N1 = (1 - self.xi) * (1 - self.eta) / 4
            N2 = (1 + self.xi) * (1 - self.eta) / 4
            N3 = (1 + self.xi) * (1 + self.eta) / 4
            N4 = (1 - self.xi) * (1 + self.eta) / 4

            self.N[0, ::2] = np.array([N1, N2, N3, N4])
            self.N[1, 1::2] = np.array([N1, N2, N3, N4])
            return self.N

        elif self.el_type == 1:
            # shape function for 8 noded quadrilateral element
            N1 = -(1 - self.xi) * (1 - self.eta)*(1 + self.xi + self.eta) / 4
            N2 = (1 + self.xi) * (1 - self.eta) * (-1 + self.xi - self.eta) / 4
            N3 = (1 + self.xi) * (1 + self.eta) * (-1 + self.xi + self.eta) / 4
            N4 = (1 - self.xi) * (1 + self.eta) * (-1 - self.xi + self.eta) / 4
            N5 = (1 - self.xi ** 2) * (1 - self.eta) / 2
            N6 = (1 + self.xi) * (1 - self.eta ** 2) / 2
            N7 = (1 - self.xi ** 2) * (1 + self.eta) / 2
            N8 = (1 - self.xi) * (1 - self.eta ** 2) / 2

            self.N[0, ::2] = np.array([N1, N2, N3, N4, N5, N6, N7, N8])
            self.N[1, 1::2] = np.array([N1, N2, N3, N4, N5, N6, N7, N8])
            return self.N

    def compute_Jacobian(self, vertex_coord):
        """Calculate the Jacobian matrix for 4 noded and 8 noded quadrilateral element"""

        # dpN = gradient of shape functions wrt natural coordinates (xi, eta)

        if self.el_type == 0:
            self.dpN[0, 0] = -(1 - self.eta) / 4
            self.dpN[1, 0] = -(1 - self.xi) / 4
            self.dpN[0, 1] = (1 - self.eta) / 4
            self.dpN[1, 1] = -(1 + self.xi) / 4
            self.dpN[0, 2] = (1 + self.eta) / 4
            self.dpN[1, 2] = (1 + self.xi) / 4
            self.dpN[0, 3] = -(1 + self.eta) / 4
            self.dpN[1, 3] = (1 - self.xi) / 4

        elif self.el_type == 1:
            self.dpN[0, 0] = -(1-self.eta)*(-2*self.xi-self.eta)/4
            self.dpN[1, 0] = -(1-self.eta)*(-self.xi-2*self.eta)/4
            self.dpN[0, 1] = (1 - self.eta) * (2*self.xi - self.eta) / 4
            self.dpN[1, 1] = -(1 - self.eta) * (-self.xi - 2 * self.eta) / 4
            self.dpN[0, 2] = (1 + self.eta) * (2*self.xi + self.eta) / 4
            self.dpN[1, 2] = (1 + self.eta) * (self.xi + 2 * self.eta) / 4
            self.dpN[0, 3] = (1 + self.eta) * (2*self.xi - self.eta) / 4
            self.dpN[1, 3] = (1 - self.eta) * (-self.xi + 2 * self.eta) / 4
            self.dpN[0, 4] = -2*self.xi * (1- self.eta) / 2
            self.dpN[1, 4] = (-1 + self.xi**2) / 2
            self.dpN[0, 5] = (1- self.eta**2)/2
            self.dpN[1, 5] = -2*self.eta*(1+self.xi) / 2
            self.dpN[0, 6] = -2*self.xi*(1+self.eta) / 2
            self.dpN[1, 6] = (1 - self.xi**2) / 2
            self.dpN[0, 7] = -(1 - self.eta**2) / 2
            self.dpN[1, 7] = (1 - self.xi)*(-2*self.eta) / 2

        self.J = np.dot(self.dpN, vertex_coord)

    def get_B_matrix(self):
        """Calculates the strain displacement matrix (B)"""

        # dN = spatial gradient of shape functions
        dN = np.dot(np.linalg.inv(self.J), self.dpN)
        B = np.zeros((3, 2*self.dpN.shape[1]))

        B[0, ::2] = dN[0, :]
        B[1, 1::2] = dN[1, :]
        B[2, ::2] = dN[1, :]
        B[2, 1::2] = dN[0, :]

        return B






