import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import displacement_solver
import mesh_gen


# --------------------------------------------------------------------------------------------------------------
# 1) Inputs
# a) Problem parameters
# Domain geometry
domain_coord = np.array([[0, 0], [48, 44], [48, 60], [0, 44], [24, 22], [48, 52], [24, 52], [0, 22]])
# Body force components
b = np.array([[0], [0]])
# Traction components
q = 1/16
T = np.array([[0, 0], [0, q], [0, 0]])
# b) Young's modulus
E = 1.0
# c) Poisson's ratio
nu = 1/3
# d) Problem type (0 ----> plane stress, 1 -----> plane strain)
problem_type = 0
# e) Element used for meshing (0 ---> 4 node quadrilateral, 1 -----> 8 node quadrilateral)
el_type = 1
# f) No. of Gauss points required for integration
ngp2d = 3
ngp1d = 2

# Testing
nx = 4
ny = 4
mesh = mesh_gen.MeshGenerator(nx, ny, domain_coord.reshape(16, 1), el_type)
coord = mesh.coord_array()
connect = mesh.connectivity()
print(connect)


# g) Mesh sizes to be tested
N = [2, 4, 8, 16, 32]
# ----------------------------------------------------------------------------------------------------------------

u_list = Parallel(n_jobs=-1, verbose=100)(
    delayed(displacement_solver.solve_fem)(N[i], E, nu, ngp2d, ngp1d, el_type, problem_type, domain_coord, b, T)
    for i in range(len(N)))

print(u_list)

# # Plot mesh convergence plot of displacements
# fig, ax = plt.subplots()
# ax.plot(N, u_list, ls="-", markersize=5, marker="o", mfc="black", mec="none")
# ax.set_xlabel("Mesh size")
# ax.set_ylabel("$u_y$")
# ax.set_title("Convergence plot using 4-noded quadrilateral elements")
# # plt.savefig("displacement_convergence_Q4.svg")
# # plt.show()


