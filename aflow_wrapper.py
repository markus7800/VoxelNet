from aflow import * 
import pandas as pd
import numpy as np

def aflow_fetch(species):
    species_query_string = ",".join(sorted(species))
    print(f"Searching for {species_query_string} ...")
    result = search(batch_size=20
            ).filter(K.species == species_query_string
            ).filter(K.nspecies == len(species)
            ).orderby(K.enthalpy_atom)
    
    df = pd.DataFrame(columns=[
    "catalog", "compound", "composition", "species", "natoms",
    "geometry", "positions_fractional",
    "enthalpy_atom"
    ])
    
    counter = 1
    for entry in result:
        print(f"{counter}. Found compound", entry.compound)
        row = {
            "catalog": entry.catalog.rstrip("\n"),
            "compound": entry.compound,
            "composition": entry.composition,
            "natoms": entry.natoms,
            "species": entry.species,
            "geometry": entry.geometry,
            "positions_fractional": entry.positions_fractional,
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
    #for p in positions_fractional:
    #    print(A.dot(p))
    positions_cartesian = np.apply_along_axis(
        lambda p: A.dot(p), 1, positions_fractional
    )
    return positions_cartesian