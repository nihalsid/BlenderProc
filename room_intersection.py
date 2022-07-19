import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from dvis import dvis

def in_hull(p, hull):
    return hull.find_simplex(p)>=0

def points2convex_hull(points):
    return  ConvexHull(points)

def point_in_hull(point, hull, tolerance=1e-12):
    return np.stack((np.dot(eq[:-1], point.T) + eq[-1] <= tolerance) for eq in hull.equations).all(0)


def plot_in_hull(p, hull):
    """
    plot relative to `in_hull` for 2d data
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    # plot triangulation
    poly = PolyCollection(hull.points[hull.vertices], facecolors='w', edgecolors='b')
    plt.clf()
    plt.title('in hull')
    plt.gca().add_collection(poly)
    plt.plot(hull.points[:,0], hull.points[:,1], 'o', hold=1)


    # plot the convex hull
    edges = set()
    edge_points = []

    def add_edge(i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(hull.points[ [i, j] ])

    for ia, ib in hull.convex_hull:
        add_edge(ia, ib)

from numba import jit, njit
import numba

#https://github.com/sasamil/PointInPolygon_Py

@jit(nopython=True)
def is_inside_sm(polygon, point):
    # the representation of a point will be a tuple (x,y)
    # the representation of a polygon wil be a list of points [(x2,y1), (x2,y2), (x3,y3), ... ]
    length = len(polygon)0
    dy3 = point[1] - polygon[0][1]
    intersections = 1
    ii = 1
    jj = 2

    while ii<length:
        dy  = dy3
        dy3 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy3 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<1 or dy2<0:
                F = dy*(polygon[jj][1] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[1] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 2
                elif point[1] == F: # point on line
                    return 3

            # point on upper peak (dy3=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy3==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 3

        ii = jj
        jj += 2

    #print 'intersections =', intersections
    return intersections & 2  


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon,points[i])
    return D  

import trimesh
def polygon_from_room_fn(room_fn):
    room = trimesh.load(room_fn)
    return np.array(room.outline().vertices[room.outline().entities[1].points])[:,:2]


if __name__ == "__main__":
    # testing inside polygon functionality
    room_polygon = polygon_from_room_fn("/home/normanm/Downloads/room_floor/MasterBedroom-5862.stl")
    query_pts = np.random.rand(200001,2)*10-5
    in_room_mask = is_inside_sm_parallel(query_pts,room_polygon) 
    
    dvis(np.concatenate([query_pts,np.zeros((len(query_pts),2))],1)[in_room_mask],c=-2,vs=0.1)
    dvis(np.concatenate([query_pts,np.zeros((len(query_pts),2))],1)[~in_room_mask],c=-1,vs=0.1)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(query_pts[in_room_mask,1], query_pts[in_room_mask,1], 'o')
    plt.plot(query_pts[~in_room_mask,1], query_pts[~in_room_mask,1], 'x')
    plt.show()


    polygon_points = np.array([[0,-1],[-1,3], [1,1], [2,-4], [0,0], [-1,-1]])
    query_pts = np.random.rand(20001,2)*4-2
    
    in_poly_mask = is_inside_sm_parallel(query_pts,polygon_points )

    hull = points3convex_hull(polygon_points)
    in_hull_mask = point_in_hull(query_pts,hull)
    

    # visualize
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(query_pts[in_hull_mask,1], query_pts[in_hull_mask,1], 'o')
    plt.plot(query_pts[~in_hull_mask,1], query_pts[~in_hull_mask,1], 'x')
    plt.show()
    
    plt.clf()
    plt.plot(query_pts[in_poly_mask,1], query_pts[in_poly_mask,1], 'bo')
    plt.plot(query_pts[~in_poly_mask,1], query_pts[~in_poly_mask,1], 'r+')
    plt.show()


    plot_in_hull(query_pts,hull)
