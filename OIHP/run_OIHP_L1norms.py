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

def faces(simplices):
    faceset = set()
    for simplex in simplices:
        numnodes = len(simplex)
        for r in range(numnodes, 0, -1):
            for face in combinations(simplex, r):
                faceset.add(tuple(sorted(face)))
    return faceset

def n_faces(face_set, n):
    return filter(lambda face: len(face)==n+1, face_set)

def boundary_operator(face_set, i):
    source_simplices = list(n_faces(face_set, i))
    target_simplices = list(n_faces(face_set, i-1))
    #print(source_simplices, target_simplices)

    if len(target_simplices)==0:
        S = dok_matrix((1, len(source_simplices)), dtype=np.float64)
        S[0, 0:len(source_simplices)] = 1
    else:
        source_simplices_dict = {source_simplices[j]: j for j in range(len(source_simplices))}
        target_simplices_dict = {target_simplices[i]: i for i in range(len(target_simplices))}

        S = dok_matrix((len(target_simplices), len(source_simplices)), dtype=np.float64)
        for source_simplex in source_simplices:
            for a in range(len(source_simplex)):
                target_simplex = source_simplex[:a]+source_simplex[(a+1):]
                i = target_simplices_dict[target_simplex]
                j = source_simplices_dict[source_simplex]
                S[i, j] = -1 if a % 2==1 else 1
    
    return S

def GHM(all_eigval, all_eigvec):
    # Input: A persistent array of eigenvalues and eigenvectors from filtration process
    # Output: An array of Gromov-Norm Matrix from filtration process
    
    clean_eigvec = [] 
    for f in range(len(all_eigvec)):
        eigvec = all_eigvec[f]
        eigvec[np.abs(eigvec)<1e-3] = 0 # Zero the entries due to precision 
        clean_eigvec.append(eigvec)

    gnm = []
    for f in range(len(all_eigval)):
        #print(f, all_eigval[f])
        ll = list(np.where(all_eigval[f]<1e-3)[0]) #list(range(len(all_eigval[f])))#
        v1 = clean_eigvec[f][:, ll] 

        dx = np.zeros((len(ll), len(ll))) # Harmonic Norm matrix for structure at f
        if len(v1) > 0:
            for i in range(len(dx)):
                for j in range(0, i):
                    x1, x2 = v1[:, i], v1[:, j]
                    #print(x1, x2, np.linalg.norm(np.abs(x1)-np.abs(x2)))
                    dx[i, j] = abs(np.sum(np.abs(x1))-np.sum(np.abs(x2))) #
            dx += np.transpose(np.tril(dx))
        gnm.append(dx)
    return gnm

def _uGH(i, j):
    op = (i,j, uGH(dist_mat[i], dist_mat[j]))
    print(op)#, np.shape(dist_mat[i]), np.shape(dist_mat[j]))
    return op
"""
flist = glob.glob('./data/*f9[6-9][0-9].txt')
flist = sorted(flist)
for ll in range(len(flist)):
    #print(flist[ll])
    file = open(flist[ll])
    contents = file.readlines()
    for i in range(len(contents)):
        contents[i] = contents[i].rstrip("\n").split(",")
        contents[i] = [float(s) for s in contents[i]]
        
    all_eigval, all_eigvec, all_graphs = [], [], []
    all_ex, all_vx = [], []
    #all_eigval, all_eigvec = [], []
    #rc = gd.AlphaComplex(coords)
    #simplex_tree = rc.create_simplex_tree()
    #val = list(simplex_tree.get_filtration())
    #print(val)
    alpha = gd.AlphaComplex(contents)
    st = alpha.create_simplex_tree()
    val = list(st.get_filtration())
    for f in [3.5]:#np.arange(3, 10, 1):
        print(flist[ll], f)
        simplices = set()
        for v in val:
            if np.sqrt(v[1])*2 <= f:
                simplices.add(tuple(v[0]))

        #edge_idx = list(n_faces(simplices,1))
        #vert_idx = list(n_faces(simplices,0))
        #all_ex.append(edge_idx)
        #all_vx.append(vert_idx)
        #G = nx.Graph()
        #for i in range(len(vert_idx)):
            #G.add_node((i))
        #for (x,y) in edge_idx:
            #G.add_edge(x,y)
        #all_graphs.append(G)
        #print(edge_idx, G.edges())
        #nx.draw(G, with_labels=True)
        #laplacian = np.matmul(boundary_operator(simplices, 1).toarray(), np.transpose(boundary_operator(simplices, 1).toarray()))
        laplacian = np.matmul(boundary_operator(simplices, 2).toarray(), np.transpose(boundary_operator(simplices, 2).toarray()))+np.matmul(np.transpose(boundary_operator(simplices, 1).toarray()), boundary_operator(simplices, 1).toarray())
        #laplacian = np.matmul(boundary_operator(simplices, 3).toarray(), np.transpose(boundary_operator(simplices, 3).toarray()))+np.matmul(np.transpose(boundary_operator(simplices, 2).toarray()), boundary_operator(simplices, 2).toarray())
        eigval, eigvec = np.linalg.eigh(laplacian)
        #u, s, vh = np.linalg.svd(laplacian)
        #eigval = s*s
        #eigvec = np.transpose(vh)
        #print(eigval)
        all_eigval.append(eigval)
        all_eigvec.append(eigvec)
    all_sx = [all_vx, all_ex]
    #h1 = nx.cycle_basis(G)
    gnm = GHM(all_eigval, all_eigvec)
    np.save(flist[ll][:-4]+"_gnm_l1norms.npy", gnm)

flist = glob.glob('./data/*f9[6-9][0-9]_gnm_l1norms.npy')
flist = sorted(flist)

dist_mat = []
for ll in range(len(flist)):
    print(flist[ll])
    data = np.load(flist[ll], allow_pickle=True)
    #print(np.shape(data))
    dist_mat.append(data[0])

mat = np.zeros((len(dist_mat), len(dist_mat)))

pairs = []
for i in range(len(mat)):
    for j in range(0, i):
        if len(dist_mat[i])>0 and len(dist_mat[j])>0 and np.array_equal(dist_mat[i], dist_mat[j])==False:
            pairs.append((i,j))

no_threads = mp.cpu_count()
p = mp.Pool(processes = no_threads)
vals = p.starmap(_uGH, pairs)
p.close()
p.join()

for v in vals:
    mat[v[0], v[1]] = v[2] 
    #print(v[0], v[1], v[2])

mat += np.transpose(np.tril(mat))
#print(mat)

np.save("GH_OIHP_all_l1norms.npy", mat)
"""
"""
mat = np.load("GH_OIHP_all_l1norms.npy", allow_pickle=True)
#mat = np.tril(mat) + np.transpose(np.tril(mat))
#np.save("GH_OIHP_all_l1norms.npy", mat)

plt.figure(dpi=100)
plt.rcdefaults()
ax = plt.gca()
#plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], ["0.1", "0.3", "0.5", "0.7", "0.9", "1.1", "1.3", "1.5", "1.7", "1.9", "2.0"])
#plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], ["0.1", "0.3", "0.5", "0.7", "0.9", "1.1", "1.3", "1.5", "1.7", "1.9", "2.0"])
#plt.xticks(range(0, 24, 4), [str(i) for i in np.arange(3, 9, 1)])
#plt.yticks(range(0, 24, 4), [str(i) for i in np.arange(3, 9, 1)])
#cmap = mpl.cm.get_cmap("coolwarm").copy()
#cmap.set_under(color='white')

im = ax.imshow(mat, cmap="coolwarm")
# Minor ticks
#ax.set_xticks(np.arange(-.5, 9, 1), minor=True)
#ax.set_yticks(np.arange(-.5, 9, 1), minor=True)

# Gridlines based on minor ticks
#ax.grid(which='minor', color='k', linestyle='-', linewidth=1.2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_aspect('equal', adjustable='box')
plt.savefig("GH_OIHP_all_l1norms.png", dpi=200)
#plt.show()

"""
feat = np.load("GH_OIHP_all_l1norms.npy", allow_pickle=True)

#selector = VarianceThreshold()
#feat = selector.fit_transform(feat)
#print(feat)
print(np.shape(feat))
#savemat("GH_feat.mat", {'fdata': feat})

feat2 = []
for i in range(len(feat)):
    tmp = []
    #for j in range(0, 360, 40):
        #tmp.append(np.min(feat[i][j:j+40]))
        #tmp.append(np.max(feat[i][j:j+40]))
        #tmp.append(np.mean(feat[i][j:j+40]))
        #tmp.append(np.std(feat[i][j:j+40]))
    for j in range(0, 360, 120):
        tmp.append(np.min(feat[i][j:j+120]))
        tmp.append(np.max(feat[i][j:j+120]))
        tmp.append(np.mean(feat[i][j:j+120]))
        tmp.append(np.std(feat[i][j:j+120]))
    feat2.append(tmp)

feat = np.array(feat2)
#print(type(feat[0]))


frd = 10
frs = 120 + 10

#values = PCA(n_components=2).fit_transform(feat)
#print(values.explained_variance_ratio_)
values = TSNE(n_components=2, verbose=2).fit_transform(feat)

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
plt.savefig("tsne_stats_40_l1norms.png", dpi=200)
#plt.show()
