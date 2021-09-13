import numpy as np
import math


def intersection(line1, line2):
    """Function to find the intersection points of two lines in 2D"""

    p1 = line1[0]
    p2 = line1[1]
    p3 = line2[0]
    p4 = line2[1]

    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


def equidist_points(a, b, points):
    """Function to calculate the coordinates of equidistant points between two end points"""

    point_x = np.linspace(a[0, :], b[0, :], points+1).reshape(points+1)
    point_y = np.linspace(a[1, :], b[1, :], points+1).reshape(points+1)
    point_mat = np.zeros((points+1, 2))
    point_mat[:, 0] = point_x
    point_mat[:, 1] = point_y
    return point_mat


class MeshGenerator:
    """Class used to construct the mesh of the element
    Inputs:
    nx = no. of elements in x direction
    ny = no. of elements in y direction
    points_coord = coordinates of all the 8 nodes (16 x 1)
    el_type = Element used for meshing (0-->4 node quadrilateral, 1-->8 node quadrilateral)"""

    def __init__(self, nx, ny, points_coord, el_type):
        self.nx = nx
        self.ny = ny
        self.points_coord = points_coord
        self.el_type = el_type

    def coord_array(self):
        """Function to generate the coordinate matrix

        Output:
        coordinate_xy = coordinate matrix (n_nodes x 2)
        outer_node = outer nodes of the element (list)
        inner_node = inner nodes of the element (list)"""

        # Array of vertex points of the elements
        x1 = self.points_coord[:2, :]
        x2 = self.points_coord[2:4, :]
        x3 = self.points_coord[4:6, :]
        x4 = self.points_coord[6:8, :]

        # Coordinate data
        side1 = equidist_points(x1, x2, self.nx)
        side2 = equidist_points(x2, x3, self.ny)
        side3 = equidist_points(x4, x3, self.nx)
        side4 = equidist_points(x1, x4, self.ny)

        if self.el_type == 0:
            coordinate_xy = np.zeros(((self.nx+1)*(self.ny+1), 2))
            k = 0
            for i in range(1, self.ny):
                k = k + self.nx + 1
                for j in range(1, self.nx):
                    coordinate_xy[k+j, :] = intersection((side1[j, :], side3[j, :]), (side2[i, :], side4[i, :]))

            for i in range(self.nx+1):
                coordinate_xy[i, :] = side1[i, :]

            k = 0
            for i in range(1, self.ny):
                k = k + self.nx + 1
                coordinate_xy[k, :] = side4[i, :]

                coordinate_xy[k+self.nx, :] = side2[i, :]

            coordinate_xy[self.ny*(self.nx+1):, :] = side3
            return coordinate_xy

        elif self.el_type == 1:
            # Number of nodes
            n_nodes = (self.nx + 1)*(self.ny + 1) + self.nx*(self.ny + 1) + (self.nx + 1)*self.ny
            # Initializing coordinate matrix
            coordinate_xy = np.zeros(((self.nx+1)*(self.ny+1), 2))
            # Computing vertex nodes of each element
            k = 0
            for i in range(1, self.ny):
                k = k + self.nx + 1
                for j in range(1, self.nx):
                    coordinate_xy[k + j, :] = intersection((side1[j, :], side3[j, :]), (side2[i, :], side4[i, :]))

            for i in range(self.nx + 1):
                coordinate_xy[i, :] = side1[i, :]

            k = 0
            for i in range(1, self.ny):
                k = k + self.nx + 1
                coordinate_xy[k, :] = side4[i, :]

                coordinate_xy[k + self.nx, :] = side2[i, :]

            coordinate_xy[self.ny * (self.nx + 1):, :] = side3
            # Computing mid-side nodes of each element
            node_grid = np.arange((self.nx+1)*(self.ny+1)).reshape(self.ny+1, self.nx+1)
            mid_node_h = np.zeros((self.nx*(self.ny + 1), 2))
            mid_node_v = np.zeros(((self.nx + 1)*self.ny, 2))

            k = 0
            for i in range(self.ny+1):
                for j in range(self.nx):
                    nd1 = node_grid[i, j]
                    nd2 = node_grid[i, j+1]
                    mid_node_h[k, :] = (coordinate_xy[nd1, :] + coordinate_xy[nd2, :])/2
                    k = k + 1

            k = 0
            for i in range(self.nx+1):
                for j in range(self.ny):
                    nd1 = node_grid[j, i]
                    nd2 = node_grid[j+1, i]
                    mid_node_v[k, :] = (coordinate_xy[nd1, :] + coordinate_xy[nd2, :])/2
                    k = k + 1

            coordinate_xy = np.vstack((coordinate_xy, mid_node_h, mid_node_v))
            return coordinate_xy

    def connectivity(self):
        """Generates the connectivity matrix of the element"""

        if self.el_type == 0:
            # Connectivity matrix
            nel = self.nx * self.ny
            connect = np.zeros((nel, 4), dtype='int64')
            el = 0
            for i in range(self.ny):
                for j in range(self.nx):
                    nd1 = j + (self.nx+1)*i
                    nd2 = nd1 + 1
                    nd3 = nd2 + (self.nx+1)
                    nd4 = nd3 - 1
                    connect[el, :] = np.array([nd1, nd2, nd3, nd4])
                    el = el + 1
            return connect

        elif self.el_type == 1:
            nel = self.nx * self.ny
            connect_vnodes = np.zeros((nel, 4), dtype="int64")
            connect_mnodes = np.zeros((nel, 4), dtype="int64")
            el = 0
            for i in range(self.ny):
                for j in range(self.nx):
                    nd1 = j + (self.nx + 1) * i
                    nd2 = nd1 + 1
                    nd3 = nd2 + (self.nx + 1)
                    nd4 = nd3 - 1
                    connect_vnodes[el, :] = np.array([nd1, nd2, nd3, nd4])
                    el = el + 1

            el = 0
            k1 = (self.nx + 1)*(self.ny + 1)
            k2 = k1 + self.nx*(self.ny + 1)
            for i in range(self.ny):
                for j in range(self.nx):
                    nd1 = k1 + j
                    nd3 = nd1 + self.nx
                    nd4 = k2 + j*self.ny
                    nd2 = nd4 + self.ny
                    connect_mnodes[el, :] = np.array([nd1, nd2, nd3, nd4])
                    el = el + 1
                k1 = k1 + self.nx
                k2 = k2 + 1

            connect = np.hstack((connect_vnodes, connect_mnodes))
            return connect
