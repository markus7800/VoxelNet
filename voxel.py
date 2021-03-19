import numpy as np
import matplotlib.pyplot as plt

def bravais_lattice(s, A, n=101, d=2):
    xi = np.linspace(0,1,n)
    
    if d == 2:
        x1, x2 = np.meshgrid(xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1))) # (2,n*n)
        
    R = A.dot(X) # (2,n*n)
        
    SR = np.apply_along_axis(s, 0, R) # apply field column-wise, (1,n*n)
    
    SR = SR.reshape(n,n)
    
    return (xi, SR)

def reciprocal_lattice(s, A, n=101, g0=-16, g1=16, d=2):
    
    B = 2*np.pi * np.linalg.inv(A).T
    
    xi = np.linspace(0,1,n)
    mi = np.arange(g0,g1+1)
    m = len(mi)
    
    if d == 2:
        x1, x2 = np.meshgrid(xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1))) # (2,n*n)
        
        m1, m2 = np.meshgrid(mi, mi)
        M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1))) # (2,m*m)
    else:
        raise ValueError("Not implemented")

        
    R = A.dot(X) # (2,n*n)
    G = B.dot(M) # (2,m*m)
    
    XR = R.T.dot(G) # (n*n, m*m) (=2 * np.pi * X.T.dot(M))
    
    E = np.exp(-1j * XR) # (n*n, m*m)

    SR = np.apply_along_axis(s, 0, R) # apply field column-wise, (n*n,)

    delta = (xi[1] - xi[0])**d
    SG = delta * SR.dot(E) # (m*m,)
    
    SG = SG.reshape(m,m)
    
    return (B, mi, SG)


def plot_2D_realspace_lattice(A, xi, SR):
    n = len(xi)
    x1, x2 = np.meshgrid(xi, xi)
    X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1))) # (2,n*n)
    R = A.dot(X)
    r1 = R[0,:].reshape(n,n)
    r2 = R[1,:].reshape(n,n)
    
    l1 = np.abs(r1).max()
    l2 = np.abs(r2).max()

    fig = plt.figure(figsize=(4 * l1/l2 + 1,4))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.pcolormesh(r1, r2, SR, shading='auto')    
    plt.colorbar()
    plt.show()
    
    
def plot_2D_bravais_lattice(A, xi, SR):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.pcolormesh(xi, xi, SR, shading='auto')
    
    plt.xlabel(f"a1 = ({A[0,0]:.2f}, {A[1,0]:.2f})")
    plt.ylabel(f"a2 = ({A[0,1]:.2f}, {A[1,1]:.2f})")
    
    plt.colorbar()
    plt.show()
    
    
def plot_2D_reciprocal_lattice(B, mi, SG, xlims=None, ylims=None):
    absSG = np.abs(SG)
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.pcolormesh(mi, mi, absSG, shading='auto')
    plt.colorbar()
    
    plt.xlabel(f"b1 = ({B[0,0]:.2f}, {B[1,0]:.2f})")
    plt.ylabel(f"b2 = ({B[0,1]:.2f}, {B[1,1]:.2f})")
    
    plt.show()
    
    plt.figure(figsize=(4,4))
    m1, m2 = np.meshgrid(mi, mi)
    M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1)))
    G = B.dot(M)
    
    max_sg = absSG.max()
    colors = [(1.,0.,0.,v) for v in absSG.reshape(-1) / max_sg]
    plt.scatter(G[0,:], G[1,:], c=colors)
    if xlims:
        plt.xlim(xlims)
    if ylims:
        plt.ylim(ylims)
    
    plt.show()