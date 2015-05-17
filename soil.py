import scipy as sp
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import shapefile

data = ['./data/spatial/aoi_a_aoi',
        './data/spatial/soilmu_a_aoi',
        ]
sf = shapefile.Reader(data[1])
shapes = sf.shapes()
records = sf.records()

rec_dict = {}
for i in range(0, len(records)):
    key = records[i][2]
    if key not in rec_dict:
        rec_dict[key] = len(rec_dict)

cmap = plt.get_cmap('rainbow')
colors = []
patches = []

plt.ion()
fig, ax = plt.subplots()
ss = ['b','g','r','c','m','y','k']
for i in range(0, len(shapes)):
    if (shapes[i].parts[0]):
        print("Parts != 0 detected")
    points = sp.array(shapes[i].points)
    polygon = Polygon(points, True)
    patches.append(polygon)
    key = records[i][2]
    colors.append(rec_dict[key]/len(rec_dict))
    #plt.plot(points[:,0],points[:,1],color=cmap(rec_dict[key]/len(rec_dict)))

p = PatchCollection(patches, cmap=matplotlib.cm.jet,  alpha=0.4)
ax.add_collection(p)
p.set_array(sp.array(colors))
plt.colorbar(p)
ax.autoscale()
plt.show()
