import numpy as np


def boundary_data_generator(nx, ny, n_side):
    """Function to generate the data for boundary conditions"""

    n_ele = nx*ny
    bc_type = np.zeros((n_ele, n_side), dtype="int64")

    # Assign bottom boundary
    bc_type[:nx, 0] = 1
    # Assign right boundary
    bc_type[(nx-1):n_ele:nx, 1] = 2
    # Assign top boundary
    bc_type[nx*(ny-1):, 2] = 3
    # Assign left boundary
    bc_type[0::nx, 3] = -1

    return bc_type

