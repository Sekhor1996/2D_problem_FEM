import numpy as np
from scipy.sparse.linalg import spsolve
import mesh_gen
import stiffness_matrix
import force_vector
import boundary_data_generation


def solve_fem(N, E, nu, ngp2d, ngp1d, el_type, problem_type, domain_coord, b, T):
    """Function to calculate the nodal displacements"""
    # Mesh parameters
    nx = N
    ny = N
    n_ele = nx * ny
    n_nodes = (nx + 1) * (ny + 1)
    n_dof = 2 * n_nodes

    # 2) Generate mesh data
    mesh_obj = mesh_gen.MeshGenerator(nx, ny, domain_coord.reshape(16, 1), el_type)
    # Coordinate matrix
    coord, _, _ = mesh_obj.coord_array()
    # Connectivity matrix
    connect = mesh_obj.connectivity()
    # Boundary data
    bc_type = boundary_data_generation.boundary_data_generator(nx, ny, 4)

    # Generate global stiffness matrix
    K_global = stiffness_matrix.global_stiffness(coord, connect, E, nu, el_type, problem_type, ngp2d)
    # plt.spy(K_global)
    # plt.show()

    # Generate global force vector
    f_global = force_vector.f_global(coord, connect, b, T, bc_type, el_type, ngp2d, ngp1d)
    # print(f_global)

    # Application of Dirichlet BC
    nodes_2d = np.arange(n_nodes).reshape(nx + 1, ny + 1)
    nodes_d = nodes_2d[:, 0]
    dof_prescribed = np.zeros((2 * nodes_d.shape[0]), dtype="int64")

    # Prescribed displacements
    u_prescribed = np.zeros((dof_prescribed.shape[0]))
    for i in range(nodes_d.shape[0]):
        dof_prescribed[2 * i] = 2 * nodes_d[i]
        dof_prescribed[2 * i + 1] = 2 * nodes_d[i] + 1

    nodes_left = list(set(nodes_2d.flatten()) - set(nodes_d))

    dof_left = np.zeros((2 * len(nodes_left)), dtype="int64")
    for i in range(len(nodes_left)):
        dof_left[2 * i] = 2 * nodes_left[i]
        dof_left[2 * i + 1] = 2 * nodes_left[i] + 1

    K_reduced = ((K_global[dof_left]).tocsc()[:, dof_left]).tocsr()
    f_reduced = f_global[dof_left]

    # K_II = ((K_global[dof_left]).tocsc()[:, dof_left]).tocsr()
    # K_IO = ((K_global[dof_left]).tocsc()[:, dof_prescribed]).tocsr()

    # Solving the system of equations
    u_reduced = spsolve(K_reduced, f_reduced)
    # u_I = -spsolve(K_II, K_IO.dot(u_prescribed))
    # print(u_I)
    # Nodal displacement vector
    u_node = np.zeros(n_dof)
    u_node[dof_prescribed] = u_prescribed
    u_node[dof_left] = u_reduced

    # Node at which displacement is computed
    node = nodes_2d[int(ny/2), -1]
    dof = 2*node+1
    return u_node[dof]
