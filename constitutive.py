import numpy as np


def constitutive(E, nu, problem_type):
    """Calculate the constitutive matrix for the plane elasticity problem
    Inputs:
    1) E = Young's Modulus (N/m^2)
    2) nu = Poisson's ratio
    3) problem_type = type of problem, plane stress or plane strain
    problem_type = 0 (plane stress)
    problem_type = 1 (plane strain)

    Output:
    C = 3x3 constitutive matrix for the given material"""

    C = np.eye(3)
    if problem_type == 0:
        const = E/(1 - nu**2)
        C[0, 1] = nu
        C[2, 2] = (1-nu)/2
        C[1, 0] = nu

    else:
        const = E/((1+nu)*(1-2*nu))
        C[0, 0] = 1 - nu
        C[1, 1] = C[0, 0]
        C[0, 1] = nu
        C[1, 0] = C[0, 1]
        C[2, 2] = (1 - 2*nu)/2

    C = const * C
    return C
