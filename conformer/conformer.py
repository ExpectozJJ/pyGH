import numpy as np 
from GH import uGH
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                    #for k in range(len(x1)):
                        #if x1[k] > 0 and x2[k] < 0:
                            #x2 = -x2
                            #break
                        #elif x1[k] < 0 and x2[k] > 0:
                            #x1 = -x1
                            #break
                    dx[i, j] = abs(np.sum(np.abs(x1))-np.sum(np.abs(x2))) # np.sum(np.abs(x1-x2))# CHANGE THIS LINE HERE TO CHANGE THE NORM
            dx += np.transpose(np.tril(dx))
        gnm.append(dx)
    return gnm

def WM(all_eigval, all_eigvec, all_M):
    # Input: A persistent array of eigenvalues and eigenvectors from filtration process
    #        M is a square matrix consisting of cost entries for transport between simplex i and simplex j
    # Output: An array of Gromov-Norm Matrix from filtration process
    
    clean_eigvec = [] 
    for f in range(len(all_eigvec)):
        eigvec = all_eigvec[f]
        eigvec[np.abs(eigvec)<1e-3] = 0 # Zero the entries due to precision 
        clean_eigvec.append(eigvec)

    wm = []
    for f in range(len(all_eigval)):
        print(f, len(all_eigval[f]))
        ll = list(np.where(all_eigval[f]<1e-3)[0]) #list(range(len(all_eigval[f])))#
        v1 = clean_eigvec[f][:, ll] 

        dx = np.zeros((len(ll), len(ll))) # Harmonic Norm matrix for structure at f
        if len(v1) > 0:
            for i in range(len(dx)):
                for j in range(0, i):
                    x1, x2 = v1[:, i]**2, v1[:, j]**2
                    #print(x1, x2, np.linalg.norm(np.abs(x1)-np.abs(x2)))
                    if np.sum(x1) < 1:
                        x1[np.argmax(x1)] += 1-np.sum(x1)
                        
                    if np.sum(x2) < 1: 
                        x2[np.argmax(x2)] += 1-np.sum(x2)
                    dx[i, j] = ot.emd2(x1, x2, all_M[f])
                    #print(i, j, dx[i, j])
            dx += np.transpose(np.tril(dx))
        wm.append(dx)
    return wm

file = open("./catalyst_feat/0.xyz", 'r')

contents = file.readlines()
data = []
for i in range(2, len(contents)):
    line = contents[i].split()
    data.append([float(line[1]), float(line[2]), float(line[3])])
#print(data)

all_eigval, all_eigvec, all_graphs = [], [], []
all_ex, all_vx = [], []
#all_eigval, all_eigvec = [], []
#rc = gd.AlphaComplex(coords)
#simplex_tree = rc.create_simplex_tree()
#val = list(simplex_tree.get_filtration())
#print(val)
for f in np.round(np.arange(0, 7.1, 0.2), 1):
    print(f)
    rc = gd.RipsComplex(data, max_edge_length=f)
    simplex_tree = rc.create_simplex_tree(max_dimension=2)
    val = list(simplex_tree.get_filtration())
    simplices = set()
    for v in val:
        #print(v)
        #if np.sqrt(v[1])*2 <= f:
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

#gnm = GHM(all_eigval, all_eigvec)
gnm = WM(all_eigval, all_eigvec, all_M)

mat = np.zeros((len(all_eigval), len(all_eigval)))

for i in range(len(all_eigval)):
    for j in range(0, i):
        #print(i, j)
        if len(gnm[i])>0 and len(gnm[j])>0 and np.array_equal(gnm[i], gnm[j])==False:
            #print(gnm[i], gnm[j])
            U = uGH(gnm[i], gnm[j])
            #print(U)
            mat[i, j] = U

mat += np.transpose(np.tril(mat))

#np.unique(mat)

plt.figure(dpi=200)
plt.rcdefaults()
ax = plt.gca()
#plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], ["0.1", "0.3", "0.5", "0.7", "0.9", "1.1", "1.3", "1.5", "1.7", "1.9", "2.0"])
#plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], ["0.1", "0.3", "0.5", "0.7", "0.9", "1.1", "1.3", "1.5", "1.7", "1.9", "2.0"])
#plt.xticks(range(0, 110, 5), [str(i) for i in np.arange(0, 22, 1)])
#plt.yticks(range(0, 110, 5), [str(i) for i in np.arange(0, 22, 1)])
plt.xticks([])
plt.yticks([])
im = ax.imshow(mat[:71, :71], cmap='coolwarm')
# Minor ticks
#ax.set_xticks(np.arange(-.5, 20, 1), minor=True)
#ax.set_yticks(np.arange(-.5, 20, 1), minor=True)

for i in range(len(mat)):
    for j in range(len(mat)):
        ax.text(i,j,round(mat[i,j],3),ha='center',va='center')
        #ax.text(i,j,mat[i,j],ha='center',va='center')

# Gridlines based on minor ticks
#ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_aspect('equal', adjustable='box')
plt.savefig("GH_cat_wasserstein.png", dpi=200)
plt.show()