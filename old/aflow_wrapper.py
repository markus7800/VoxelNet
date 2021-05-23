from aflow import * 
import pandas as pd
import numpy as np

def aflow_fetch(species):
    species_query_string = ",".join(sorted(species))
    print(f"Searching for {species_query_string} ...")
    result = search(batch_size=20
            ).filter(K.species == species_query_string
            ).filter(K.nspecies == len(species)
            ).orderby(K.enthalpy_atom
            ).select(K.compound, K.composition, K.species, K.natoms,
                      K.geometry, K.natoms, K.positions_fractional, K.positions_cartesian,
                      K.enthalpy_atom)
    
    df = pd.DataFrame(columns=[
    "auid", "aurl",
    "compound", "composition", "species", "natoms",
    "geometry", "positions_fractional", "positions_cartesian",
    "enthalpy_atom"
    ])
    
    counter = 1
    for entry in result:
        print(f"{counter}. Found compound", entry.compound, "with auid", entry.auid)
        row = {
            "auid": entry.auid,
            "aurl": entry.aurl,
            
            "compound": entry.compound,
            "composition": entry.composition,
            "species": entry.species,
            "natoms": entry.natoms,
            
            "geometry": entry.geometry,
            "positions_fractional": entry.positions_fractional,
            "positions_cartesian": entry.positions_cartesian,
            
            "enthalpy_atom": entry.enthalpy_atom
        }
        df = df.append(row, ignore_index=True)
        counter += 1
        
    print("Done.")
        
    return df

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


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_3D_crystal(crystal):
    A = calc_basis(crystal.geometry)

    fig = plt.figure()
    ax = Axes3D(fig)
    
    colors = ["red", "green", "blue", "orange", "purple", "black"]
    
    cs = []
    legend_handels = []
    
    for i, n in enumerate(crystal.composition):
        cs = cs + [colors[i]]*n
        legend_element = plt.Line2D([0], [0], marker='o', color="w",
                                    markerfacecolor= colors[i], label=crystal.species[i], markersize=10)
        legend_handels.append(legend_element)

    poss = calc_cartesian_positions(A, crystal.positions_fractional)
    ax.scatter(poss[:,0], poss[:,1], poss[:,2], alpha=1, c=cs, s=25)
    for i in range(3):
        ax.plot([0, A[0,i]], [0,A[1,i]], [0,A[2,i]])
     
    
    
    plt.suptitle(crystal.compound)
    ax.legend(handles=legend_handels)
    plt.show()