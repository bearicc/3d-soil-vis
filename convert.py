import scipy as sp
import numpy as np
import pandas as pd
from pyproj import *
import os

filename = ["data/05044572.xyzi",
            "data/05044574.xyzi",
            "data/05064572.xyzi",
            "data/05064574.xyzi",
    ]

data_list = []
for i in range(len(filename)):
    print("Load data "+filename[i]+" ...")
    data_list.append(pd.read_csv(filename[i],delimiter=",",names=["x","y","z","i"]).values)


# convert UTM NAD83 Zone 15N to Lat/Lon
p = Proj(init='epsg:26915')

data_write_list = []
for i in range(len(filename)):
    print("Convert data "+filename[i]+" ...")
    lon,lat = p(data_list[i][:,0],data_list[i][:,1],inverse=True)
    data_write = sp.zeros((data_list[i].shape[0],3))
    data_write[:,0] = lon
    data_write[:,1] = lat
    data_write[:,2] = data_list[i][:,2]
    #data_write[:,3] = data_list[i][:,3]
    data_write_list.append(data_write)
    #sp.save(os.path.splitext(filename[i])[0],data_write)
    #sp.savetxt(os.path.splitext(filename[i])[0]+".llzi",data_write,fmt="%.8f",delimiter=",")

data_write = data_write_list[0]
for i in range(1,len(filename)):
    data_write = sp.concatenate((data_write,data_write_list[i]),axis=0)
sp.save("data/data",data_write)
