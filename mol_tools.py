import numpy as np

# https://en.wikipedia.org/wiki/Fractional_coordinates#In_crystallography
def calc_basis(geometry):
    a = geometry[0]
    b = geometry[1]
    c = geometry[2]
    alpha = geometry[3] / 180 * np.pi
    beta = geometry[4] / 180 * np.pi
    gamma = geometry[5] / 180 * np.pi
    
    cosa = np.cos(alpha)
    cosb = np.cos(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    
    omega = a*b*c*np.sqrt(1 - cosa*cosa - cosb*cosb - cosg*cosg + 2*cosa*cosb*cosg)
    
    A = np.array([
        [ a, b * cosg,                      c * cosb],
        [0., b * sing, c * (cosa - cosb*cosg) / sing],
        [0.,       0.,        omega / (a * b * sing)]
    ])
    
    A = np.round(A, decimals=14)
    
    return A


def calc_cartesian_positions(A, positions_fractional):
    positions_cartesian = np.apply_along_axis(
        lambda p: A.dot(p), 1, positions_fractional
    )
    return positions_cartesian

# C,D,H,W
def make_descriptor(mol, sigma, L, N, elements=None):
    if elements is None:
        elements = mol.species # for testing
    elements = np.array(elements)
        
    descriptor = np.zeros((len(elements), N, N, N))
    
    A = calc_basis(mol.geometry)
    coords = calc_cartesian_positions(A, mol.positions_fractional)
    
    
    mx, my, mz = get_mesh_coords(A, L, N)
        
    # atom coordinations are order accoring to composition and species
    cs = np.cumsum(mol.composition)
    cs = np.insert(cs, 0, 0) # insert 0 at beginning
    
    print(elements)
        
    for i, element in enumerate(mol.species):
        print(element)
        element_coords = coords[cs[i]:cs[i+1]]
        print(cs[i], cs[i+1])
        print(element_coords, element_coords.shape)
        B, G, SG = reciprocal_lattice_gaussian(A, element_coords, sigma, mx, my, mz)
        element_descriptor = adapt_to_voxel_grid(G, SG, L, N) # (N,N,N)
        
        j = np.where(elements == element)[0][0]
        descriptor[j,:,:,:] = element_descriptor
    
    return descriptor
