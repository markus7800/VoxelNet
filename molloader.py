import numpy as np
from voxel import *
from mol_tools import *

import torch

def single_channel_reciprocal_data(mol, sigma, L, N, per_atom):
    """
    Computes reciprocal vectors and coefficients from molecular information fetched from AFLOW.
    All atoms are used to make a perpare data for a single channel NxNxN descriptor.
    The standard deviations for the Gaussians is sigma and the Fourier coefficients for
    all reciprocal vectors lying in [-pi/L*N, pi/L*N]^3 are computed.
    
    Returns a triple consisting of:
    - Compound name : String
    - (reciprocal vectors, coefficients) : (3xn shaped array, vector of length n)
    - enthalpy for mol (if per_atom is true then enthalpy per atom)
  
    """
    width = 2 * np.pi / L * N
    adj_width = width*np.sqrt(3)
    A = calc_basis(mol.geometry)
    coords = calc_cartesian_positions(A, mol.positions_fractional)
    mx, my, mz = get_mesh_coords(A, adj_width)
    B, G, SG = reciprocal_lattice_gaussian(A, coords, sigma, mx, my, mz)
    
    y = mol.enthalpy_atom
    if not per_atom:
        y = y * mol.natoms
    
    return mol.compound, (G, SG), y
    
def prepare_single_channel_data(df, sigma, L, N, per_atom):
    """
    Computes reciprocal vectors and coefficents for all rows (compounds) of the dataframe df.
    For more details see single_channel_reciprocal_data.
    """
    names = []
    reciprocal_data = []
    ys = []
    for index, mol in df.iterrows():
        name, rd, y = single_channel_reciprocal_data(mol, sigma, L, N, per_atom)

        names.append(name)
        reciprocal_data.append(rd)
        ys.append(y)
                   
    return names, reciprocal_data, ys
        

def multi_channel_reciprocal_data(mol, sigma, L, N, elements, per_atom, reduce_data=True):
    """
    Computes reciprocal vectors and coefficients from molecular information fetched from AFLOW.
    The atoms are split by element to prepare data for a multichannel MxNxNxN descriptor.
    The standard deviations for the Gaussians is sigma and the Fourier coefficients for
    all reciprocal vectors lying in [-pi/L*N, pi/L*N]^3 are computed.
    If reduce_data is set to true than the reciprocal data will be reduced to vectors lying
    in the sphere with radius pi/L*N *sqrt(3) to safe RAM. This is all the data needed for
    the both the cartesian and spherical descriptor.
    
    Returns a triple consisting of:
    - Compound name : String
    - [(j, element, reciprocal vectors, coefficients)] : list of tuples (int, string, dxn shaped array, vector of length n))
        List of reciprocal data grouped by element; j is the index for element in the elements vector
    - enthalpy for mol (if per_atom is true then enthalpy per atom)
  
    """
    
    width = 2 * np.pi / L * N
    adj_width = width*np.sqrt(3)
    
    A = calc_basis(mol.geometry)
    coords = calc_cartesian_positions(A, mol.positions_fractional)
    mx, my, mz = get_mesh_coords(A, adj_width)

    # atom coordinations are order according to composition and species    
    cs = np.cumsum(mol.composition)
    cs = np.insert(cs, 0, 0) # insert 0 at beginning

    rd = [] # collect G and SG for all elements
    for i, element in enumerate(mol.species):
        element_coords = coords[cs[i]:cs[i+1]] # extract coordinates corresponding to atoms of one element
        B, G, SG = reciprocal_lattice_gaussian(A, element_coords, sigma, mx, my, mz)
        
        if reduce_data:
            # only keep data that lies in relevant circle
            in_grid = np.linalg.norm(G,axis=0) <= adj_width/2 + 2*width/N # add 2 times voxel width to be sure (rounding)
            G = G[:,in_grid].astype(np.float32)
            SG = np.abs(SG[in_grid]).astype(np.float32) # also convert complex to real here to half the size
            
        j = np.where(elements == element)[0][0] # get channel for element
        rd.append((j, element, (G, SG)))
        
    y = mol.enthalpy_atom
    if not per_atom:
        y = y * mol.natoms
    
    return mol.compound, rd, y


def multi_channel_reciprocal_data(df, sigma, L, N, elements, per_atom, reduce_data=True):
    """
    Computes reciprocal vectors and coefficents for all rows (compounds) of the dataframe df.
    For more details see single_channel_reciprocal_data.
    """ 
    
    names = []
    reciprocal_data = []
    ys = []
    counter = 0
    total = df.shape[0]
    
    for index, mol in df.iterrows():
        name, rd, y = multi_channel_reciprocal_data(mol, sigma, L, N, elements, per_atom, reduce_data)
        
        names.append(name)
        reciprocal_data.append(rd)
        ys.append(y)
        
        counter += 1
        print(f"Calculated {counter}/{total} compounds.      ", end="\r")
        
    return names, reciprocal_data, ys


class MolLoader(object):
    """
    Generator for preparing, batch-splitting, augmenting AFLOW data.
    
    Functionality:
    - Preparing dataframe of AFLOW data
        Here usual parameters sigma, L, N.
        If nchannel = 1 and elements = None then single channel descriptors.
        Otherwise, atoms will be splitted by elements vector (all elements occuring
        in compounds of df) into len(elements) channels.
        if per_atom is set to true then the enthalpy per atom is the target variable,
        otherwise total enthalpy.
        If reduce_data is set to true only the required reciprocal data is stored
        (more RAM efficient, no reason to set to false)
    - Splitting into batches
        batch_size controls number of compounds per batch
        if shuffle is set to true the compounds will be shuffled before splitting into batches
    - Augmenting reciprocal data and making the voxel descriptor
        Two descriptors are supported: cartesian and spherical.
        The reciprocal data needed for the descriptors is prepared in the init.
        Data augmentation is performed at generation (iteration) step (next).
        rotate_randomly, reflect_randomly specify the augmentation.
        The transformations (rotation + reflection) are applied to the reciprocal vectors
        and the (Mx)NxNxN descriptors have to be computed at generation step.
        An alternative would be to compute the (Mx)NxNxN descriptors once and apply the 
        transformations directly to these tensors, but this would require interpolation.
    - Copying to GPU
        if decice is set to cuda the batches will be copied to GPU
    """
    def __init__(self, df, sigma, L, N, batch_size, nchannel=1, elements=None,
                 shuffle=False, rotate_randomly=False, reflect_randomly=False,
                 device=torch.device('cpu'), reduce_data=True, per_atom=True, mode='cartesian'):
        
        # prepare/precompute reciprocal data
        if elements is None:
            names, reciprocal_data, ys = prepare_single_channel_data(df, sigma=sigma, L=L, N=N, per_atom=per_atom)
        else:
            names, reciprocal_data, ys = prepare_multi_channel_data(df, sigma=sigma, L=L, N=N, per_atom=per_atom,
                                                                    elements=elements, reduce_data=reduce_data)
            nchannel = len(elements)

        
        self.names = names
        self.reciprocal_data = reciprocal_data
        self.ys = ys
        
        self.batch_size = batch_size
        self.rotate_randomly = rotate_randomly
        self.reflect_randomly = reflect_randomly
        self.device = device
        self.mode = mode
        
        if self.mode != 'cartesian' and self.mode != 'spherical':
            raise ValueError(f"{self.mode} must be either cartesian or spherical")
        
        self.L = L
        self.N = N
        self.nchannel = nchannel
        
        self.shuffle = shuffle
        self.current = 0
        self.indices = np.arange(len(ys))
        self.N_data = len(self.indices)
        
        print(f"Initialised MolLoader with {self.N_data} compounds.")
        print(f"    sigma={sigma}, L={self.L}, N={self.N}, nchannel={self.nchannel}, mode={self.mode}, device={self.device}")
        print(f"    shuffle={self.shuffle}, rotate={self.rotate_randomly}, reflect={self.reflect_randomly}")
        
    def __iter__(self):
        return self
    
    def __next__(self):
         return self.next()

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))
    
    def reset(self, batch_size, shuffle, rotate_randomly, reflect_randomly):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rotate_randomly = rotate_randomly
        self.reflect_randomly = reflect_randomly
        self.current = 0
        self.indices = np.arange(len(self.ys))

    def next(self):
        # if first iteration shuffle compounds if shuffle is set to true
        if self.current == 0 and self.shuffle:
            np.random.shuffle(self.indices)
        
        if self.current < self.N_data:
            # compute index range for batch
            n1 = self.current
            n2 = min(self.current + self.batch_size, self.N_data)
            
            # init batch
            x_names = []
            x = np.zeros((n2-n1, self.nchannel, self.N, self.N, self.N), dtype="float32")
            y = np.zeros((n2-n1,1), dtype="float32")
            
            for i, j in enumerate(range(n1, n2)):
                # get index as in dataframe
                data_index = self.indices[j] # indices are shuffled if shuffle is set to true
                
                # target variable and compound name
                y[i] = self.ys[data_index]
                x_names.append(self.names[data_index])
                
                # generate 3x3 transformation matrix (rotations and/or reflections)
                R = np.eye(3)
                if self.rotate_randomly:
                    R = get_random_3D_rotation_matrix()
                if self.reflect_randomly:
                    if np.random.rand() < 0.5:
                        R = -R
                
                # make descriptor for compound single/multi channel, cartesian/spherical
                if self.nchannel == 1:
                    G, SG = self.reciprocal_data[data_index]
                    if self.mode == 'cartesian':
                        descriptor = make_voxel_grid(G, SG, self.L, self.N, rot=R)
                    elif self.mode == 'spherical':
                        descriptor = make_spherical_voxel_grid(G, SG, self.L, self.N, rot=R)
                    x[i, 0, :, :, :] = descriptor
                else:
                    for j, element, (G, SG) in self.reciprocal_data[data_index]:
                        if self.mode == 'cartesian':
                            descriptor = make_voxel_grid(G, SG, self.L, self.N, rot=R)
                        elif self.mode == 'spherical':
                            descriptor = make_spherical_voxel_grid(G, SG, self.L, self.N, rot=R)
                        x[i, j, :, :, :] = descriptor
                                           
            
            self.current = n2
            
            # copy to GPU
            x = torch.from_numpy(x).to(self.device, non_blocking=True)
            y = torch.from_numpy(y).to(self.device, non_blocking=True)
            
            return (x_names, x, y)
        
        self.current = 0
        # print("Stop Iteration")
        raise StopIteration()

        
        