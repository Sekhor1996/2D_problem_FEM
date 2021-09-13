import numpy as np
from scipy.sparse import coo_matrix
from shape_function import *
from constitutive import *
from quadrature import *
# import time
from joblib import Parallel, delayed


def get_element_stiffness(ele, coord, connect, E, nu, el_type, problem_type, ngp):
    """Calculates the elemental stiffness matrix
        Inputs:
        1) ele = element number (starts from 0)
        2) coord = coordinate array
        3) connect = connectivity array
        4) E = Young's Modulus
        5) nu = Poisson's Ratio
        6) el_type = type of element used for meshing
            el_type = 0 (4 noded quadrilateral)
            el_type = 1 (8 noded quadrilateral)
        7) problem_type = type of 2D elasticity problem
            problem_type = 0 (plane stress)
            problem_type = 1 (plane strain)
        8) ngp = number of Gauss points in each direction used for integration

        Output:
        K_ele = element stiffness matrix"""

    # Constitutive matrix
    C = constitutive(E, nu, problem_type)
    # Nodes
    node = connect[ele, :]
    # Coordinates of the nodes
    vertex_coord = coord[node, :]
    # Element stiffness matrix
    K_ele = np.zeros((2 * vertex_coord.shape[0], 2 * vertex_coord.shape[0]))
    # Gauss points and weights
    gp, weights = quadrature(ngp)

    for i in range(ngp):
        for j in range(ngp):
            xi = gp[i]
            eta = gp[j]
            shape_func = ShapeFunction(xi, eta, el_type)
            shape_func.compute_Jacobian(vertex_coord)
            J = shape_func.J
            B = shape_func.get_B_matrix()
            K_ele += (B.T @ C @ B) * weights[i] * weights[j] * np.linalg.det(J)

    return K_ele


def global_stiffness(coord, connect, E, nu, el_type, problem_type, ngp):
    """Function to calculate the global stiffness matrix

    Inputs:
    1) coord = coordinate array
    2) connect = connectivity array
    3) E = Young's Modulus
    4) nu = Poisson's ratio
    5) el_type = type of element used for meshing
        el_type = 0 (4 noded quadrilateral)
        el_type = 1 (8 noded quadrilateral)
    6) problem_type = type of 2D elasticity problem
        problem_type = 0 (plane stress)
        problem_type = 1 (plane strain)
    7) ngp = number of Gauss points in each direction used for integration

    Output:
    K_global = global stiffness matrix"""

    n_ele = connect.shape[0]    # No. of elements
    node_perele = connect.shape[1]  # No. of nodes per element
    ndof_perele = 2*node_perele # No. of DOF per element

    # tic = time.perf_counter()
    K_par = Parallel(n_jobs=-1)(delayed(get_element_stiffness)(ele, coord, connect, E, nu, el_type, problem_type, ngp)
                               for ele in range(n_ele))

    # K_temp = np.zeros((n_ele, ndof_perele, ndof_perele))
    # for i in range(K_temp.shape[0]):
    #     K_temp[i, :, :] = K_par[i]

    K_temp = np.array(K_par)

    I = np.zeros((n_ele, K_temp.shape[1] ** 2))
    J = np.zeros((n_ele, K_temp.shape[1] ** 2))

    dof = np.zeros((n_ele, 2 * node_perele))
    for j in range(connect.shape[1]):
        dof[:, 2 * j] = 2 * connect[:, j]
        dof[:, 2 * j + 1] = 2 * connect[:, j] + 1

    for i in range(dof.shape[1]):
        I[:, i*ndof_perele:(i+1)*ndof_perele] = dof[:, i].reshape(dof.shape[0], 1)*np.ones((dof.shape[0], ndof_perele))
        J[:, i*ndof_perele:(i+1)*ndof_perele] = dof

    I = I.flatten()
    J = J.flatten()
    V = K_temp.flatten()
    # Forming the global stiffness matrix in COO format
    K_global = coo_matrix((V, (I, J)), shape=(2 * coord.shape[0], 2 * coord.shape[0]))
    # Converting the global stiffness matrix to LIL format to perform indexing and slicing operations
    K_global = K_global.tocsr()

    return K_global
