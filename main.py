import scipy as sp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
# ------------------------------------------------------------
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.widgets import Button
from scipy.interpolate import Rbf
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import shapefile

# ------------------------------------------------------------
def get_data_plot(data):
    N = 50
    bPlotted = sp.zeros((N,N,50))
    data_plot = sp.zeros(data.shape)
    data_plot_len = 0
    xmin = data[:,0].min()
    xmax = data[:,0].max()
    ymin = data[:,1].min()
    ymax = data[:,1].max()
    zmin = data[:,2].min()
    zmax = data[:,2].max()
    dx = (xmax-xmin)/(bPlotted.shape[0]-1)
    dy = (ymax-ymin)/(bPlotted.shape[1]-1)
    dz = (zmax-zmin)/(bPlotted.shape[2]-1)
    for i in range(data.shape[0]):
        x = data[i,0];
        y = data[i,1];
        z = data[i,2];
        nx = int((x-xmin)/dx)
        ny = int((y-ymin)/dy)
        nz = int((z-zmin)/dz)
        if not bPlotted[nx,ny,nz]:
            bPlotted[nx,ny,nz] = 1
            data_plot[data_plot_len,:] = data[i,:]
            data_plot_len = data_plot_len+1

    return data_plot[0:data_plot_len,:]

# ------------------------------------------------------------
# read shape files
spatial_data = ['./data/spatial/aoi_a_aoi',
        './data/spatial/soilmu_a_aoi',
        ]
sf = shapefile.Reader(spatial_data[1])
shapes = sf.shapes()
records = sf.records()

rec_dict = {}
for i in range(0, len(records)):
    key = records[i][2]
    if key not in rec_dict:
        rec_dict[key] = len(rec_dict)

# ------------------------------------------------------------
print("Loading data ...")
# lon, lat
xmin, xmax, ymin, ymax = [-90.069432, -96.684901, 43.557019, 40.332776]
data = sp.load("data/data.npy")
xmin = min(data[:,0])
xmax = max(data[:,0])
ymin = min(data[:,1])
ymax = max(data[:,1])
zmin = min(data[:,2])
zmax = max(data[:,2])

points_all = sp.empty((0,2))
for i in range(0,len(shapes)):
    if (shapes[i].parts[0]):
        print("Parts != 0 detected")
    if (shapes[i].shapeType != 5):
        print("ShapeType != 5 detected")
    verts = sp.array(shapes[i].points)
    if ((verts[:,0] > xmin).all() and (verts[:,0] < xmax).all() and (verts[:,1] > ymin).all() and (verts[:,1] < ymax).all()):
        points_all = sp.concatenate((points_all,sp.array(shapes[i].points)),axis=0)

grid_x, grid_y = sp.mgrid[xmin:xmax:100j,ymin:ymax:100j]

file_grid_z = "data/grid_z_100.npy"
if os.path.isfile(file_grid_z):
    print("Loading mesh data ...")
    grid_z = sp.load(file_grid_z)
else:
    print("Interpolating mesh data ...")
    grid_z = griddata(data[:,0:2],data[:,2],(grid_x,grid_y), method='cubic')
    sp.save(file_grid_z, grid_z)

file_data_plot = "data/data_plot.npy"
if os.path.isfile(file_data_plot):
    print("Loading plot data ...")
    data_plot = sp.load("data/data_plot.npy")
else:
    print("Calc plot data ...")
    data_plot = get_data_plot(data)
    sp.save(file_data_plot, data_plot)

file_vertsz = "data/vertsz.npy"
if os.path.isfile(file_vertsz):
    print("Loading vertsz ...")
    vertsz = sp.load(file_vertsz)
else:
    print("Interpolating ...")
    vertsz = griddata(data[:,0:2],data[:,2],points_all,method='cubic')
#    X = np.array(data)
#    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X[:,0:2])
#    Y = np.array(points_all)
#    distances, indices = nbrs.kneighbors(Y)
#    vertsz = sp.zeros(indices.shape[0])
#    for i in range(0,indices.shape[0]):
#        vertsz[i] = (X[indices[i,0],2]+X[indices[i,1],2]+X[indices[i,2],2])/3;
    sp.save(file_vertsz, vertsz)

cmap = plt.get_cmap('rainbow')
colors = []
#ss = ['b','g','r','c','m','y','k']
fig = plt.figure()
fig.hold(True)
ax = fig.add_subplot(111, projection='3d')

"""
for i in range(0,len(shapes)):
    if (shapes[i].parts[0]):
        print("Parts != 0 detected")
    if (shapes[i].shapeType != 5):
        print("ShapeType != 5 detected")
    points = sp.array(shapes[i].points)
    verts = sp.zeros((points.shape[0],3))
    verts[:,0:2] = points
    verts[:,2] = zmin
    key = records[i][2]
    color = cmap(rec_dict[key]/len(rec_dict))
    p = Poly3DCollection([verts])
    p.set_color(color)
    ax.add_collection3d(p)
"""
pos = 0
for i in range(0,len(shapes)):
    if (shapes[i].parts[0]):
        print("Parts != 0 detected")
    if (shapes[i].shapeType != 5):
        print("ShapeType != 5 detected")
    key = records[i][2]
    """
    if(rec_dict[key] != 2):
        continue
    """
    points = sp.array(shapes[i].points)
    verts = sp.zeros((points.shape[0],3))
    verts[:,0:2] = points
    if ((verts[:,0] > xmin).all() and (verts[:,0] < xmax).all() and (verts[:,1] > ymin).all() and (verts[:,1] < ymax).all()):
        verts[:,2] = vertsz[pos:pos+points.shape[0]]
        j = 0
        while (j < len(verts)):
            if (sp.isnan(verts[j,2])):
                verts = sp.delete(verts,(j),axis=0)
            else:
                j += 1
        if (len(verts)<=0):
            continue
        pos = pos+points.shape[0]
        color = cmap(rec_dict[key]*1.0/len(rec_dict))
        ax.plot(verts[:,0], verts[:,1], verts[:,2], "b.")
        p = Poly3DCollection([verts])
        p.set_color(color)
        ax.add_collection3d(p)

#verts = sp.array(list(zip(x, y,z)))
ax.plot_wireframe(grid_x,grid_y,grid_z,color="black")
#ax.plot(points_all[:,0], points_all[:,1], np.ones(points_all.shape[0])*250, "r.")
#ax.plot(data_plot[:,0], data_plot[:,1], data_plot[:,2], "g.")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim3d(xmin,xmax)
ax.set_ylim3d(ymin,ymax)
ax.set_zlim3d(zmin,zmax)
plt.show(block=True)
