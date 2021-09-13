import numpy as np
import shape_function
import quadrature
from joblib import Parallel, delayed


def ele_body_force(ele, coord, connect, b, el_type, ngp):
    """Function to calculate the element body force vector
    Inputs:
    1) ele = element no.
    2) coord = coordinate matrix
    3) connect = connectivity matrix
    4) b = body force vector
    5) el_type = element type (0 ---> 4 noded quadrilateral, 1 ---> 8 noded quadrilateral)
    6) ngp = number of Gauss points"""

    # Nodes
    node = connect[ele, :]
    # Coordinates of the nodes
    vertex_coord = coord[node, :]
    # Element stiffness matrix
    fbody_ele = np.zeros((2 * vertex_coord.shape[0], 1))
    # Gauss points and weights
    gp, weights = quadrature.quadrature(ngp)

    for i in range(ngp):
        for j in range(ngp):
            xi = gp[i]
            eta = gp[j]
            shape_func = shape_function.ShapeFunction(xi, eta, el_type)
            N_matrix = shape_func.get_N_matrix()
            shape_func.compute_Jacobian(vertex_coord)
            J = shape_func.J
            fbody_ele += (N_matrix.T @ b) * weights[i] * weights[j] * np.linalg.det(J)

    return fbody_ele.reshape(2 * vertex_coord.shape[0])


def ele_f_nbc(ele, coord, connect, bc_type, T, el_type, ngp):
    """Function used for calculating the Natural Boundary term of each element
    Inputs:
    a) ele = element no.
    b) coord = coordinate matrix
    c) connect = connectivity matrix
    d) bc_type = boundary data
    e) T = matrix containing all the traction vectors along all the Neumann boundaries
    g) ngp = no. of Gauss points used for line integration"""

    ele_edges_nbc = list(bc_type[ele, :])
    nodes = connect[ele, :]
    vertex_coord = coord[nodes, :]
    corner_nodes = np.hstack((nodes[0:4], nodes[0]))
    f_nbc_ele = np.zeros((2 * vertex_coord.shape[0], 1))
    gp, weights = quadrature.quadrature(ngp)

    if len(ele_edges_nbc) != 0:
        for i in range(len(ele_edges_nbc)):
            side_no = ele_edges_nbc[i]
            t = T[side_no-1, :].reshape(2, 1)
            node1 = corner_nodes[side_no - 1]
            node2 = corner_nodes[side_no]
            p1 = coord[node1, :]
            p2 = coord[node2, :]
            detJ = np.linalg.norm(p2 - p1) / 2

            if side_no == 1:
                eta = -1
                for j in range(ngp):
                    xi = gp[j]
                    shape_func = shape_function.ShapeFunction(xi, eta, el_type)
                    N = shape_func.get_N_matrix()
                    f_nbc_ele += (N.T @ t) * weights[j] * detJ
            elif side_no == 2:
                xi = 1
                for j in range(ngp):
                    eta = gp[j]
                    shape_func = shape_function.ShapeFunction(xi, eta, el_type)
                    N = shape_func.get_N_matrix()
                    f_nbc_ele += (N.T @ t) * weights[j] * detJ
            elif side_no == 3:
                eta = 1
                for j in range(ngp):
                    xi = gp[j]
                    shape_func = shape_function.ShapeFunction(xi, eta, el_type)
                    N = shape_func.get_N_matrix()
                    f_nbc_ele += (N.T @ t) * weights[j] * detJ

    return f_nbc_ele.reshape(2 * vertex_coord.shape[0])


def f_global(coord, connect, b, T, bc_type, el_type, ngp2d, ngp1d):
    """Function to calculate the global load vector"""

    n_ele = connect.shape[0]  # No. of elements
    node_perele = connect.shape[1]  # No. of nodes per element
    ndof_perele = 2 * node_perele  # No. of DOF per element
    n_dof = n_ele * ndof_perele  # No. of DOFs

    f_body_par = Parallel(n_jobs=-1)(delayed(ele_body_force)(ele, coord, connect, b, el_type, ngp2d)
                                     for ele in range(n_ele))

    f_nbc_par = Parallel(n_jobs=-1)(delayed(ele_f_nbc)(ele, coord, connect, bc_type, T, el_type, ngp1d)
                                    for ele in range(n_ele))

    f_array = np.array(f_body_par) + np.array(f_nbc_par)
    print(f_array.shape)
    dof = np.zeros((n_ele, ndof_perele), dtype="int64")

    for j in range(connect.shape[1]):
        dof[:, 2 * j] = 2 * connect[:, j]
        dof[:, 2 * j + 1] = 2 * connect[:, j] + 1

    f_global = np.zeros((n_dof, 1))
    for ele in range(n_ele):
        dof_ele = dof[ele, :]
        # print(dof_ele)
        # for j in range(dof_ele.shape[0]):
        #     f_global[dof_ele[j], :] += f_array[ele, j]
        f_global[dof_ele, :] += f_array[ele, :].reshape(f_array.shape[1], 1)
        # print(f_global[dof_ele, :])

    return f_global

