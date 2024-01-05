from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import numpy as np
from numpy import matlib
import gudhi as gd
import networkx as nx
import math
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from GeneralisedFormanRicci.frc import gen_graph
from scipy.sparse import *
from scipy import *
from scipy.io import savemat
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
import matplotlib.transforms as transforms
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
from pyGH.GH import uGH
import multiprocessing as mp 
import os 
import sys 
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
simplefilter("ignore", ClusterWarning)
#import umap.umap_ as umap

def convertpdb(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        contents = f.readlines()
    
    #recordname = []

    #atomNum = []
    atomName = []
    #altLoc = []
    #resName = []

    #chainID = []
    #resNum = []
    X = []
    Y = []
    Z = []

    #occupancy = []
    #betaFactor = []
    element = []
    #charge = []
    
    
    for i in range(len(contents)):
        thisLine = contents[i]

        if thisLine[0:4]=='ATOM' or thisLine[0:6]=='HETATM':
            #recordname = np.append(recordname,thisLine[:6].strip())
            #atomNum = np.append(atomNum, float(thisLine[6:11]))
            atomName = np.append(atomName, thisLine[12:16])
            #altLoc = np.append(altLoc,thisLine[16])
            #resName = np.append(resName, thisLine[17:20].strip())
            #chainID = np.append(chainID, thisLine[21])
            #resNum = np.append(resNum, float(thisLine[23:26]))
            X = np.append(X, float(thisLine[30:38]))
            Y = np.append(Y, float(thisLine[38:46]))
            Z = np.append(Z, float(thisLine[46:54]))
            #occupancy = np.append(occupancy, float(thisLine[55:60]))
            #betaFactor = np.append(betaFactor, float(thisLine[61:66]))
            element = np.append(element,thisLine[12:14])

    #print(atomName)
    a = {'PRO': [{'atom': atomName, 'typ': element, 'pos': np.transpose([X,Y,Z])}]}
    np.savez(filename[:-4]+".npz", **a)

coords = []
flist = glob.glob('./data/*f9[6-9][0-9].txt')
flist = sorted(flist)
for ll in range(len(flist)):
    print(flist[ll])
    file = open(flist[ll])
    contents = file.readlines()
    for i in range(len(contents)):
        contents[i] = contents[i].rstrip("\n").split(",")
        contents[i] = [float(s) for s in contents[i]]
    
    coords.append(np.reshape(contents, (len(contents)*3,)))

print(np.shape(coords))

frd = 10
frs = 40 + 10

#values = PCA(n_components=2).fit_transform(feat)
#print(values.explained_variance_ratio_)
values = TSNE(n_components=2, verbose=2).fit_transform(coords)

#values = umap.UMAP(random_state=42).fit_transform(feat)
plt.figure(figsize=(5,5), dpi=200)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

plt.scatter(values[:(frs-frd),0], values[:(frs-frd),1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[(frs-frd):2*(frs-frd),0], values[(frs-frd):2*(frs-frd),1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[2*(frs-frd):3*(frs-frd),0], values[2*(frs-frd):3*(frs-frd),1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(values[3*(frs-frd):4*(frs-frd),0], values[3*(frs-frd):4*(frs-frd),1], marker='.', color='tab:red', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[4*(frs-frd):5*(frs-frd),0], values[4*(frs-frd):5*(frs-frd),1], marker='.', color='tab:purple', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[5*(frs-frd):6*(frs-frd),0], values[5*(frs-frd):6*(frs-frd),1], marker='.', color='tab:brown', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(values[6*(frs-frd):7*(frs-frd),0], values[6*(frs-frd):7*(frs-frd),1],  marker='.',color='tab:pink', alpha=0.75,  linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[7*(frs-frd):8*(frs-frd),0], values[7*(frs-frd):8*(frs-frd),1],  marker='.',color='tab:gray', alpha=0.75,  linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[8*(frs-frd):9*(frs-frd),0], values[8*(frs-frd):9*(frs-frd),1],  marker='.',color='tab:olive', alpha=0.75,  linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(np.min(values[:, 1])-10, np.max(values[:,1])+50)
#plt.xlim(-100, 100)
#plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.savefig("tsne_9types_coords.png", dpi=200)
#plt.show()
