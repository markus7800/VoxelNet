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

def get_element_list(mol):
    elements = []
    for e, n in zip(mol.species, mol.composition):
        elements = elements + [e]*n
    return elements

import ase
import nglview
def show_molecule(mol, calc_coords=True):
    A = calc_basis(mol.geometry)
    if calc_coords:
        coords = calc_cartesian_positions(A, mol.positions_fractional)
    else:
        coords = mol.positions_cartesian
        
    elements = get_element_list(mol)
    ase_mol = ase.Atoms(symbols=elements, positions=coords, pbc=False, cell=A)
    view = nglview.show_ase(ase_mol)
    view.background="black"
    display(view)
    

def elements_coords(entry, augment=True):
    elements = []
    for e, n in zip(entry.species, entry.composition):
        elements = elements + [e]*n
        
        
    pfs_list = list(entry.positions_fractional)
    
    if augment:
        for element, coords in zip(elements, pfs_list):
            for i, c in enumerate(coords):
                if c == 0:
                    new_coords = coords.copy()
                    new_coords[i] = 1.

                    already_in = False
                    for other_p in pfs_list:
                        if all(other_p == new_coords):
                            already_in = True
                            break

                    if not already_in:
                        pfs_list.append(new_coords)
                        elements.append(element)
    
    pfs = np.array(pfs_list)
    
    A = calc_basis(entry.geometry)
    cartesian_coords = calc_cartesian_positions(A, pfs)
    
    return elements, cartesian_coords
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_3D_crystal(mol):
    A = calc_basis(mol.geometry)

    fig = plt.figure()
    ax = Axes3D(fig)

    colors = ["red", "green", "blue", "orange", "purple", "black"]


    elements, coords = elements_coords(mol, augment=True)



    color_dict = {}
    for i, element in enumerate(mol.species):
        color_dict[element] = colors[i]


    cs = []
    for element in elements:
        cs.append(color_dict[element])


    corners = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]
        ])
    import itertools
    for corner1, corner2 in itertools.product(corners, repeat=2):
        if np.sum(np.abs(corner1 - corner2)) == 1:
            cc = list(zip(A.dot(corner1), A.dot(corner2)))
            ax.plot(cc[0], cc[1], cc[2], color="black")

    ax.scatter(coords[:,0], coords[:,1], coords[:,2], alpha=1, c=cs, s=25)
    for i in range(3):
        ax.plot([0, A[0,i]], [0,A[1,i]], [0,A[2,i]])

    legend_handels = []
    for i, n in enumerate(mol.composition):
        legend_element = plt.Line2D([0], [0], marker='o', color="w",
                                    markerfacecolor = colors[i], label=mol.species[i], markersize=10)
        legend_handels.append(legend_element)

    plt.suptitle(mol.compound)
    ax.legend(handles=legend_handels)
    plt.show()