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

