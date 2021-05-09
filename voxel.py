import numpy as np
import matplotlib.pyplot as plt
import time

def get_nD_gaussian(A, mus, sigma, K):
    """
    Returns field where Gaussians with standard deviation sigma are placed at centres mus repeated according to A.
    
    Mathematically,
    s(x) = sum_i sum_(k in Z^d) N(x, mu[i] + A k, sigma) where N is the pdf of a Gaussian and Z^d are integer tuples
    
    As Gaussians have infinite range, K specifies how often the centres are repeated for the approximation.
    For example, if we have a two dimensional crystal with atom at [0,0] and A = [1,0; 0,1]
    then the atom should be repeated at each integer pair [i,j] in Z^2.
    In this case, K=3 specifies that we sum over [-3,3]^2.
    
    This is the current bottleneck for 3 dimensional voxelization.
    
    Parameters
    ----------
        A: d x d matrix
            basis of unit cell
            
        mus: np.array with shape (n, d)
            coordinates corresponding to n atoms within unit cell (atoms repeat according to A)
        
        sigma: Float
            standard deviations for the Guassians
            
        K: Int
            see above
    """
    
    new_mus = []
    d = A.shape[0]
    
    if d == 2:
        for kx in np.arange(-K,K+1):
            for ky in np.arange(-K,K+1):
                for mu in mus:
                    Ak = A.dot([kx,ky])
                    new_mus.append(Ak+mu)
    if d == 3:
        for kx in np.arange(-K,K+1):
            for ky in np.arange(-K,K+1):
                for kz in np.arange(-K,K+1):
                    for mu in mus:
                        Ak = A.dot([kx,ky,kz])
                        new_mus.append(Ak+mu)
    
    new_mus = np.array(new_mus)
    nmus, d = new_mus.shape
    
    mus3 = new_mus.T.reshape(d,1,nmus)
    
    def s(X):
        d2, n = X.shape
        assert(d == d2)
        
        X_centered = X.reshape(d,n,1) - mus3 # (d, n, nmus)
        
        XX = np.sum(X_centered * X_centered, axis=0) # (n, nmus)
        
        EXX = np.exp(- XX / (2 * sigma**2)) # (n, nmus)
        
        res = np.sum(EXX, axis=1) # (n,)

        return res / (2*np.pi * sigma**2)
        
    return s

def bravais_lattice(s, A, n=101):
    """
    Evaluates the field s over the unit cell spanned by the d x d matrix A.
    
    Each axis is discretised with n points.
    
    Returns
    -------
        R: np.array with shape (d, n ** d)
            vectors in unit cell A([0,1]^d)

        SR: np.array with shape (n ** d, )
            field quantities evaluated at the vectors of R
    """
    
    d = A.shape[0]

    xi = np.linspace(0,1,n)
    
    if d == 2:
        x1, x2 = np.meshgrid(xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1))) # (2,n*n)
        
    if d == 3:
        x1, x2, x3 = np.meshgrid(xi, xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1), x3.reshape(1,-1))) # (3,n*n*n)
        
    R = A.dot(X) # (d,n^d)
        
    SR = s(R) # (n^d, )
    
    SR = SR.reshape(n,n)
    
    return (R, SR)

    
def plot_2D_realspace_lattice(A, R, SR):
    """
    Plots the results of bravais_lattice(s, A) in cartesian space.
    """
    
    n = np.int(np.sqrt(R.shape[1]))
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

    
def reciprocal_lattice(s, A, mx=None, my=None, mz=None, n=101, verbose=False):
    """
    Evaluates the field quantity s over the reciprocal space via the Fourier Transform.
    
    h(g) = 1/vol(V) int_V s(r) exp(-i g . r) dr
    
    Here the Fourier Transform is realised via numerical integration.
    
    Let B = 2pi inv(A.T) be d x d matrix corresponding to the basis of the reciprocal space.
    h is only evaluated at integer multiples of the basis vectors,  B Z^d
    
    Paramters
    ---------
        s: function mapping np.array to np.array
            field quantity
            
        A: d x d matrix
            basis of unit cell
        
        mx, my, mz: np.array
            integer ranges, h is evaluated over B (mx x my x mz)
            mz is only required for the 3 dimensional case
            
        n: Int
            number of points per axis for numerical integration
            thus we evaluate and sum over n ** d points
        
        verbose: Bool
            if True the computation time is printed
            
    Returns
    -------
        B: d x d matrix
            basis of reciprocal space
        
        G: np.array with shape (d, len(mx) * len(my) * len(mz))
            reciprocal vectors
            
        SG: np.array with shape (len(mx) * len(my) * len(mz), )
            field quantities evaluated at the vectors of G
    
    """
    
    d = A.shape[0]
    
    B = 2*np.pi * np.linalg.inv(A).T
    
    xi = np.linspace(0,1,n)
    
    
    if d == 2:
        x1, x2 = np.meshgrid(xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1))) # (2,n*n) n = len(xi)
        
        m1, m2 = np.meshgrid(mx, my)
        M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1))) # (2,m) m = len(mx) * len(my)
    elif d == 3:
        x1, x2, x3 = np.meshgrid(xi, xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1), x3.reshape(1,-1))) # (2,n*n*n) n = len(xi)
        
        m1, m2, m3 = np.meshgrid(mx, my, mz)
        M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1), m3.reshape(1,-1))) # (3,m) m = len(mx) * len(my) * len(mz)

        
    R = A.dot(X) # (d,n^d)
    G = B.dot(M) # (d,m)
    
    t0 = time.time()
    
    SR = s(R) # (n^d, )
    
    t1 = time.time()
    
    if verbose: print(f"Evaluated field at {R.shape[1]} points in {t1-t0:.2f} seconds.")
    
    XR = R.T.dot(G) # (n^d, m) (= 2 * np.pi * X.T.dot(M))
    
    E = np.exp(-1j * XR) # (n^d, m)


    delta = (xi[1] - xi[0])**d
    SG = delta * SR.dot(E) # (m,)
    
    t2 = time.time()
    if verbose: print(f"Performed Fourier Transform in {t2-t1:.2f} seconds.")

            
    return (B, G, SG)

from numpy.fft import fftn
def reciprocal_lattice_fft(s, A, mx=None, my=None, mz=None, n=101, verbose=False):
    """
    Same as reciprocal_lattice but uses numpy's fft
    """
    d = A.shape[0]
    
    B = 2*np.pi * np.linalg.inv(A).T
    
    xi = np.linspace(0,1,n)
    x_shape = (0,)
    
    if d == 2:
        x1, x2 = np.meshgrid(xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1))) # (2,n*n) n = len(xi)
        x_shape = (n,n)
        
        m1, m2 = np.meshgrid(mx, my)
        M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1))) # (2,m) m = len(mx) * len(my)
        
    elif d == 3:
        x1, x2, x3 = np.meshgrid(xi, xi, xi)
        X = np.vstack((x1.reshape(1,-1), x2.reshape(1,-1), x3.reshape(1,-1))) # (2,n*n*n) n = len(xi)
        x_shape = (n,n,n)
        
        m1, m2, m3 = np.meshgrid(mx, my, mz)
        M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1), m3.reshape(1,-1))) # (3,m) m = len(mx) * len(my) * len(mz)

        
    R = A.dot(X) # (d,n^d)
    G = B.dot(M) # (d,m)

    t0 = time.time()
    SR = s(R) # (n^d, )
    t1 = time.time()
    
    print(f"Evaluated field at {SR.shape[0]} points in {t1-t0} seconds.")

    delta = (xi[1] - xi[0])**d
    SG_ftt = delta * fftn(SR.reshape(x_shape)) # (n,n[,n])
    
    SG = np.zeros((G.shape[1],), dtype=np.complex64)
    for i in range(G.shape[1]):
        index = M[:,i].astype(np.int)
        index[index < 0] += n
        SG[i] = SG_ftt[tuple(index)]
        
        
    t2 = time.time()
    if verbose: print(f"Performed Fourier Transform in {t2-t1:.2f} seconds.")
        
    
    return (B, G, SG)

def reciprocal_lattice_gaussian(A, mus, sigma, mx=None, my=None, mz=None):
    """
    Evaluates the Gaussian field quantity
    
    s(x) = sum_i sum_(k in Z^d) N(x, mu[i] + A k, sigma), where the inner sum is over integer vectors
    
    over the reciprocal space via the Fourier Transform.
    
    h(g) = 1/vol(V) int_V s(r) exp(-i g . r) dr
    
    The field quanitity is evaluated based on theoretical derivation.
    
    Let B = 2pi inv(A.T) be d x d matrix corresponding to the basis of the reciprocal space.
    h is only evaluated at integer multiples of the basis vectors,  B Z^d
    
    Paramters
    ---------
        A: d x d matrix
            basis of unit cell
            
        mus: np.array with shape (n, d)
            coordinates corresponding to n atoms within unit cell (atoms repeat according to A)
        
        sigma: Float
            standard deviations for the Guassians
        
        mx, my, mz: np.array
            integer ranges, h is evaluated over B (mx x my x mz)
            mz is only required for the 3 dimensional case
            
    Returns
    -------
        B: d x d matrix
            basis of reciprocal space
        
        G: np.array with shape (d, len(mx) * len(my) * len(mz))
            reciprocal vectors
            
        SG: np.array with shape (len(mx) * len(my) * len(mz), )
            field quantities evaluated at the vectors of G
    
    """
    
    d = A.shape[0]
    
    B = 2*np.pi * np.linalg.inv(A).T
    
    if d == 2:
        m1, m2 = np.meshgrid(mx, my)
        M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1))) # (2,m) m = len(mx) * len(mz)
    elif d == 3:
        m1, m2, m3 = np.meshgrid(mx, my, mz)
        M = np.vstack((m1.reshape(1,-1), m2.reshape(1,-1), m3.reshape(1,-1))) # (3,m) m = len(mx) * len(my) * len(mz)

    G = B.dot(M) # (d,m)
    
    GG = np.linalg.norm(G, axis=0) ** 2 # (m,)
    
    expGG = np.exp(-sigma**2 / 2 * GG) # (m,)

    muG = mus.dot(G) # (n,d) * (d, m) = (n, m)
    
    expmuG = np.exp(-1j * muG) # (n, m)
    
    SG = 1/np.linalg.det(A) * expGG * np.sum(expmuG, axis=0) #(m*d,)

    return (B, G, SG)


def plot_2D_reciprocal_lattice(B, mx, my, G, SG, xlims=None, ylims=None, L=None, N=None):
    """"
    Plot results of reciprocal_lattice_ functions.
    
    Left plot:
    all evaluated absolute field quantities in the basis
    
    Right plot:
    evaluated absolute field quantities in cartesian space
    if L and N is specified this plot will crop to the square [-pi/L*N, pi/L*N]^2 = [-pi/delta_L, pi/delta_L]^2
    otherwise you can specify xlims and ylims
    """
    
    absSG = np.abs(SG)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.set_aspect('equal')
    cmesh = ax1.pcolormesh(mx, my, absSG.reshape((len(my), len(mx))), shading='auto')
    plt.colorbar(cmesh, ax=ax1)
    
    ax1.set_xlabel(f"b1 = ({B[0,0]:.2f}, {B[1,0]:.2f})")
    ax1.set_ylabel(f"b2 = ({B[0,1]:.2f}, {B[1,1]:.2f})")

    
    max_sg = absSG.max()
    colors = [(1.,0.,0.,v) for v in np.maximum(absSG.reshape(-1) / max_sg, 0.1)]
    ax2.scatter(G[0,:], G[1,:], c=colors)
    if L and N:   
        e = 2*np.pi/L * N / 2
        ticks = np.arange(-e, e, 2*np.pi/L)
        plt.xticks(ticks, rotation=90)
        ax2.set_yticks(ticks)
        ax2.set_xlim((-e,e))
        ax2.set_ylim((-e,e))
        ax2.set_aspect('equal')
        ax2.title.set_text(f"L = {L} Å, N = {N}")
        ax2.grid(True)
    else:
        if xlims:
            ax2.set_xlim(xlims)
        if ylims:
            ax2.set_ylim(ylims)
    
    plt.show()

    
from mpl_toolkits.mplot3d import Axes3D   
def plot_3D_reciprocal_lattice(B, G, SG, xlims=None, ylims=None, zlims=None, L=None, N=None):
    """
    three dimensional pendant of plot_2D_reciprocal_lattice
    """
    
    absSG = np.abs(SG)
        
    fig = plt.figure(figsize=(4,4))
    ax = Axes3D(fig)
    
    max_sg = absSG.max()
    colors = [(1.,0.,0.,v) for v in np.maximum(absSG.reshape(-1) / max_sg, 0.1)]
    
    if L and N:   
        e = 2*np.pi/L * N / 2
        ax.set_xlim((-e,e))
        ax.set_ylim((-e,e))
        ax.set_zlim((-e,e))
        plt.suptitle(f"L = {L} Å, N = {N}")
        plt.grid(True)
        
        for i in range(len(colors)):
            if not np.all((-e <= G[:,i]) & (G[:,i] <= e)):
                colors[i] = (0.,0.,0.,0.)
    else:         
        if xlims:
            ax.set_xlim(xlims)
        if ylims:
            ax.set_ylim(ylims)
        if zlims:
            ax.set_zlim(zlims)
    
    ax.scatter(G[0,:], G[1,:], G[2,:], c=colors)
    
    plt.show()


def get_mesh_coords(A, width):
    """
    Find integer ranges M = [mx1, mx2] x [my1, my2] ( x [mz1, mz2])
    such that G = B . M contains all points that fall into [-width / 2, width / 2]^d
    """
    d = A.shape[0]

    # find inverse of B (= 1/2pi A.T) at corner points of the cube [-width / 2, width / 2]^d
    extreme_point = np.full((d,), width/(2 * 2*np.pi))
    
    if d == 2:
        extreme_points = np.array([[1,1], [-1,1], [1,-1], [-1,-1]]) * extreme_point # (4,2)
        r_extreme_points = A.T.dot(extreme_points.T) # (2,4)

        lower = np.min(np.floor(r_extreme_points), axis=1)
        upper = np.max(np.ceil(r_extreme_points), axis=1)+1

        return np.arange(lower[0], upper[0]), np.arange(lower[1], upper[1])
    if d == 3:
        extreme_points = np.array([[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1],
                                   [-1,-1,-1], [1,-1,-1], [-1,1,-1],[-1,-1,1]]) * extreme_point # (8,3)
        r_extreme_points = A.T.dot(extreme_points.T) # (3,8)

        lower = np.min(np.floor(r_extreme_points), axis=1)
        upper = np.max(np.ceil(r_extreme_points), axis=1)+1

        return np.arange(lower[0], upper[0]), np.arange(lower[1], upper[1]), np.arange(lower[2], upper[2])
    

def make_voxel_grid(G, SG, L, N, rot=None):
    """
    Discretise the evaluations of the field over the reciprocal space
    in the cube [-pi/L*N, pi/L*N]^d = [-pi/delta_L, pi/delta_L]^d
    where delta_L = L/N and N is the number of voxels along each axis
    
    This transforming G linearly such that [-pi/delta_L, pi/delta_L]^d is mapped to [0, N]^d.
    Then it the resulting vectors are rounded down to form the final grid.
    """
    
    d, n = G.shape
    absSG = np.abs(SG)
    voxel_width = 2*np.pi / L
    grid_width = voxel_width * N # = 2*pi/delta_L
    
    if rot is None:
        rot = np.eye(d) # no rotation
    
    grid_shape = (0,)
    if d == 2:
        grid_shape = (N,N)
    if d == 3:
        grid_shape = (N,N,N)
        
    grid = np.zeros(grid_shape)
    
    RG = rot.dot(G)
    
    I = np.floor((RG + grid_width/2) / voxel_width).astype(np.int)
    
    
    in_grid = np.all((0 <= I) & (I < N), axis=0)
    I = I[:,in_grid]
    absSG = absSG[in_grid]
    
    for i in range(I.shape[1]):
        index = I[:,i]
        if d == 2:
            grid[index[1], index[0]] += absSG[i] # rows = y-axis, cols = x-axis
        if d == 3:
            grid[index[1], index[0], index[2]] += absSG[i]

    return grid

def cartisan_2D_to_spherical(X):
    r = np.linalg.norm(X, axis=0)
    theta = np.apply_along_axis(lambda x: np.arctan2(x[1],x[0]), 0, X)
    return np.vstack((r, theta))

def cartisan_3D_to_spherical(xyz):
    pts = np.zeros(xyz.shape)
    xy = xyz[0,:]**2 + xyz[1,:]**2
    pts[0,:] = np.sqrt(xy + xyz[2,:]**2)
    pts[1,:] = np.arctan2(np.sqrt(xy), xyz[2,:]) # for elevation angle defined from Z-axis down
    #pts[1,:] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    pts[2,:] = np.arctan2(xyz[1,:], xyz[0,:])
    return pts

def make_spherical_voxel_grid(G, SG, L, N, rot=None):
    """
    Discretise the evaluations of the field over the reciprocal space
    in the circle around origin with radius pi/L * N * sqrt(d) =  pi/delta_L * sqrt(d)
    where delta_L = L/N and N is the number of voxels along each axis
    
    Thus transforming G linearly such that its spherical coordinates are mapped to [0, N]^d.
    Then it the resulting vectors are rounded down to form the final grid.
    """
    
    d, n = G.shape
    absSG = np.abs(SG)
    voxel_width = np.pi / L * np.sqrt(d)
    radius = voxel_width * N # = 2*pi/delta_L * sqrt(d)
    
    if rot is None:
        rot = np.eye(d) # no rotation
    
    RG = rot.dot(G)
    
    grid_shape = (0,)
    if d == 2:
        grid_shape = (N,N)
        
        RG_spherical = cartisan_2D_to_spherical(RG)
        
        r = RG_spherical[0,:] # in [0, infty]
        phi = RG_spherical[1,:] # in [-pi, pi]

        I = np.floor(np.vstack((
            r / voxel_width,
            (phi + np.pi) / (2*np.pi/N)
        ))).astype(np.int)
    if d == 3:
        grid_shape = (N,N,N)
        
        RG_spherical = cartisan_3D_to_spherical(RG)
        
        r = RG_spherical[0,:] # in [0, infty]
        theta = RG_spherical[1,:] # in [0, pi]
        phi = RG_spherical[2,:] # in [-pi, pi]

        I = np.floor(np.vstack((
            r / voxel_width,
            theta / (np.pi/N),
            (phi + np.pi) / (2*np.pi/N)
        ))).astype(np.int)
    
    
    grid = np.zeros(grid_shape)

    in_grid = np.all((0 <= I) & (I < N), axis=0)
    I = I[:,in_grid]
    absSG = absSG[in_grid]
    
    for i in range(I.shape[1]):
        index = I[:,i]
        if d == 2:
            grid[index[0], index[1]] += absSG[i] # rows = y-axis, cols = x-axis
        if d == 3:
            grid[index[0], index[1], index[2]] += absSG[i]

    return grid

def get_2D_rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s],
        [s, c]
    ])

def get_random_2D_rotation_matrix():
    get_2D_rotation_matrix(2*np.pi*np.random.rand())

from scipy.spatial.transform import Rotation
def get_random_3D_rotation_matrix():
    return Rotation.random().as_matrix()