import numpy as np
from numpy import matlib
import math
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import matplotlib.transforms as transforms
from scipy.cluster.hierarchy import ClusterWarning
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

def uGH(ux, uy):
    ux, uy = np.array(ux, dtype=float), np.array(uy, dtype=float)
    ux = max_subdominant_ultra(ux)
    uy = max_subdominant_ultra(uy)
    spec = np.unique(np.concatenate((np.unique(ux), np.unique(uy))))
    spec = np.sort(spec)[::-1]
    #print(spec)
    
    d = 0
    ns = len(spec)
    
    for c in range(ns):
        t = spec[c]
        
        #print(ux, t)
        subux = quotientUMFPS(ux, t)
        subuy = quotientUMFPS(uy, t)
        #print(subux, subuy)
        if not is_iso(subux, subuy):
            #print(spec[max([c-1,1])])
            return spec[max([c-1,0])]
    
    return d

def max_subdominant_ultra(A):
    n = len(A)
    for k in range(n):
        #print(np.transpose(matlib.repmat(A[:, k], n, 1)))
        #print(matlib.repmat(A[k, :], n, 1))
        #print(np.maximum(np.transpose(matlib.repmat(A[:, k], n, 1)), matlib.repmat(A[k, :], n,1)))
        A = np.minimum(A, np.maximum(np.transpose(matlib.repmat(A[:, k], n, 1)), matlib.repmat(A[k, :], n,1)))
    return A

def quotientUMFPS(uX, t):
    # This function implements a Farthest Point Sampling type of algorithm.
    #print(uX)
    uXt_big = np.copy(uX)
    uXt_big[uX<=t] = 0
    #print(t)
    #print("uXt_big: ", uXt_big)
    I = [0]
    #print(uX)
    dI = uXt_big[0, :]
    #print("initial dI: ", dI)
    
    for k in range(len(uX)):
        mx = np.max(dI)
        #print(mx)
        if mx == 0:
            break 
        J = list(np.where(dI==mx)[0])
        I.append(J[0])
        #print(J)
        #print(I, J, [dI, uXt_big[J[0], :]])
        dI = np.min([dI, uXt_big[J[0], :]], axis=0) # Minimum along every column 
        #print(dI)
    
    #print(uX, I)
    uXt = uX[np.ix_(I, I)]
    return uXt

def is_iso(u1, u2):
    if len(u1) == 1:
        b = (len(u2)==1)
        return b
    
    if list(u1) == [[0.]]: u1 = 0
    if list(u2) == [[0.]]: u2 = 0
    #print(np.tril(u1)-np.transpose(np.triu(u1)))
    #print(np.shape(u1), np.shape(u2))
    lk1 = linkage(squareform(u1), 'single')
    lk2 = linkage(squareform(u2), 'single')
    
    EPS = 1e-12
    
    d1 = np.max(lk1[:, 2])
    d2 = np.max(lk2[:, 2])
    
    if abs(d1-d2) > EPS:
        return 0
    
    #print(d1-1e-10, d2-1e-10)
    t1 = fcluster(lk1, d1-1e-10, criterion='distance')
    t2 = fcluster(lk2, d2-1e-10, criterion='distance')
    
    n = np.max(t1)
    if n!=np.max(t2):
        return 0
    
    em = np.zeros((n, n))
    for i in range(n):
        #print(t1, i)
        indi = list(np.where(t1==i+1)[0])
        subu1 = u1[np.ix_(indi, indi)]
        for j in range(n):
            indj = list(np.where(t2==j+1)[0])
            #print(indj)
            subu2 = u2[np.ix_(indj, indj)]
            #print(subu1, subu2, indi, indj, em)
            em[i,j] = is_iso(subu1, subu2)
            
    return contains_perm(em)

def contains_perm(em):
    # Determines if (0-1) matrix em contrains a permutation matrix
    
    n = len(em)
    
    # Graph g have maxflow = n iff there is a perm matrix
    g = np.zeros((2*n+2, 2*n+2), dtype=int)
    for i in range(n):
        g[0, 2+i] = 1
        g[2+i+n, 1] = 1
        #print(2+i, 2+i+n)
    
    for i in range(n):
        for j in range(n):
            g[i+2, j+n+2] = em[i,j]
            #print(i+2, j+n+2)
    
    #print(g)
    m = maximum_flow(csr_matrix(g), 0, 1).flow_value
    #print(m)
    b = abs(m-n) < 0.001
    return b

def plot_uGH(ux, uy):
    U = uGH(ux, uy)
    ux, uy = np.array(ux), np.array(uy)
    diam = max(np.max(ux), np.max(uy))

    lnky = linkage(squareform(uy), 'single')
    lnkx = linkage(squareform(ux), 'single')

    hierarchy.set_link_color_palette(['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple'])
    fig, axes = plt.subplots(2, 1, figsize=(3, 8), dpi=200)
    dn1 = dendrogram(lnkx, ax=axes[0], above_threshold_color='tab:green',
                            orientation='right', color_threshold = U + 1e-16)
    axes[0].set_title('X')
    axes[0].set_xlim([0, diam+.5])
    axes[0].axvline(U, color='r', linestyle='--')
    trans = transforms.blended_transform_factory(axes[0].get_xticklabels()[0].get_transform(), axes[0].transData)
    axes[0].text(U, axes[0].get_ylim()[1], "{:.4f}".format(U), color="red", transform=trans, 
            ha="center", va="bottom")
    dn2 = dendrogram(lnky, ax=axes[1],
                            above_threshold_color='tab:green',
                            orientation='right', color_threshold = U + 1e-16)
    axes[1].set_title('Y')
    hierarchy.set_link_color_palette(None)  # reset to default after use
    axes[1].set_xlim([0, diam+.5])
    axes[1].axvline(U, color='r', linestyle='--')
    trans = transforms.blended_transform_factory(
        axes[1].get_xticklabels()[1].get_transform(), axes[1].transData)
    axes[1].text(U, axes[1].get_ylim()[1], "{:.4f}".format(U), color="red", transform=trans, 
            ha="center", va="bottom")
    plt.savefig("dendo.png", dpi=200)
    #plt.show()


ux = np.array([[ 0, 5.0225,    5.4539,    4.8977,    5.3575],
    [5.0225,         0,    5.2971,    5.4132,    5.2084],
    [5.4539,    5.2971,         0,    5.2856,    4.5969],
    [4.8977,    5.4132,    5.2856,         0,    5.6365],
    [5.3575,    5.2084,    4.5969,    5.6365,         0]])
uy = np.array([[0, 4.89878009143743,	4.73993109215105],
[4.89878009143743,	0,	5.0687],
[4.73993109215105,	5.0687,	0]])

plot_uGH(ux,uy)