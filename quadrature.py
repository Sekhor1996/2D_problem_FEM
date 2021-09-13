import numpy as np


def quadrature(ngp):
    """Values of the Gauss points and their respective weights
    Input:
    ngp = number of Gauss points in each direction in the element
    Outputs:
    1) xi = Gauss points
    2) weights = corresponding weights at the Gauss points"""

    if ngp == 2:
        xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.ones(2)
        return xi, weights

    elif ngp == 3:
        xi = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)])
        weights = np.array([5/9, 8/9, 5/9])
        return xi, weights

