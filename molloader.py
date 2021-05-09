import numpy as np
from voxel import *
from mol_tools import *

import torch

def single_channel_reciprocal_data(mol, sigma, L, N):
    width = 2 * np.pi / L * N
    adj_width = width*np.sqrt(3)
    A = calc_basis(mol.geometry)
    coords = calc_cartesian_positions(A, mol.positions_fractional)
    mx, my, mz = get_mesh_coords(A, adj_width)
    B, G, SG = reciprocal_lattice_gaussian(A, coords, sigma, mx, my, mz)
    return mol.compound, (G, SG), mol.enthalpy_atom
    
def prepare_single_channel_data(df, sigma, L, N):
    
    names = []
    reciprocal_data = []
    ys = []
    for index, mol in df.iterrows():
        name, rd, y = single_channel_reciprocal_data(mol, sigma, L, N)

        names.append(name)
        reciprocal_data.append(rd)
        ys.append(y)
                    
    return names, reciprocal_data, ys
        

def multi_channel_reciprocal_data(mol, sigma, L, N, elements, reduce_data=True):
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
            G = G[:,in_grid]
            SG = np.abs(SG[in_grid]) # also convert complex to real here to half the size
            
        j = np.where(elements == element)[0][0] # get channel for element
        rd.append((j, element, (G, SG)))
        
    return mol.compound, rd, mol.enthalpy_atom


def prepare_multi_channel_data(df, sigma, L, N, elements, reduce_data=True):
    names = []
    reciprocal_data = []
    ys = []
    counter = 0
    total = df.shape[0]
    
    for index, mol in df.iterrows():
        name, rd, y = multi_channel_reciprocal_data(mol, sigma, L, N, elements, reduce_data)
        
        names.append(name)
        reciprocal_data.append(rd)
        ys.append(y)
        
        counter += 1
        print(f"Calculated {counter}/{total} molecules.      ", end="\r")
        
    return names, reciprocal_data, ys


class MolLoader(object):
    def __init__(self, df, sigma, L, N, batch_size, nchannel=1, elements=None,
                 shuffle=False, rotate_randomly=False, device=torch.device('cpu'), reduce_data=True):
        
        if elements is None:
            names, reciprocal_data, ys = prepare_single_channel_data(df, sigma=sigma, L=L, N=N)
        else:
            names, reciprocal_data, ys = prepare_multi_channel_data(df, sigma=sigma, L=L, N=N,
                                                                    elements=elements, reduce_data=reduce_data)
            nchannel = len(elements)

        
        self.names = names
        self.reciprocal_data = reciprocal_data
        self.ys = ys
        self.batch_size = batch_size
        self.rotate_randomly = rotate_randomly
        self.device = device
        
        self.L = L
        self.N = N
        self.nchannel = nchannel
        
        self.shuffle = shuffle
        self.current = 0
        self.indices = np.arange(len(ys))
        self.N_data = len(self.indices)
        
        print(f"Initialised MolLoader with {self.N_data} molecules. sigma = {sigma}, L={self.L}, N={self.N}, nchannel={self.nchannel}, shuffle={self.shuffle}, rotate={self.rotate_randomly}, device={self.device}")
        
    def __iter__(self):
        return self
    
    def __next__(self):
         return self.next()

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))

    def next(self):
        N_data = self.N_data
        N = self.N # number of voxels
        L = self.L
        nchan = self.nchannel
        
        if self.current == 0 and self.shuffle:
            np.random.shuffle(self.indices)
        
        if self.current < N_data:
            n1 = self.current
            n2 = min(self.current + self.batch_size, N_data)
            
            x_names = []
            x = np.zeros((n2-n1, nchan, N, N, N), dtype="float32")
            y = np.zeros((n2-n1,1), dtype="float32")
            
            for i, j in enumerate(range(n1, n2)):
                data_index = self.indices[j] # supports shuffling
                
                y[i] = self.ys[data_index]
                x_names.append(self.names[data_index])
                
                R = np.eye(3)
                if self.rotate_randomly:
                    R = get_random_3D_rotation_matrix()
                
                if nchan == 1:
                    G, SG = self.reciprocal_data[data_index]
                    descriptor = make_voxel_grid(G, SG, L, N, rot=R)
                    x[i, 0, :, :, :] = descriptor
                else:
                    for j, element, (G, SG) in self.reciprocal_data[data_index]:
                        descriptor = make_voxel_grid(G, SG, L, N, rot=R)
                        x[i, j, :, :, :] = descriptor
                                           
            
            self.current = n2
            
            x = torch.from_numpy(x).to(self.device, non_blocking=True)
            y = torch.from_numpy(y).to(self.device, non_blocking=True)
            
            return (x_names, x, y)
        
        self.current = 0
        # print("Stop Iteration")
        raise StopIteration()

class LazyMolLoader(object):
    def __init__(self, df, sigma, L, N, batch_size, elements = None, nchannel=1,
                 shuffle=False, rotate_randomly=False, device=torch.device('cpu')):
        
        self.df = df
        self.elements = elements
        self.batch_size = batch_size
        self.rotate_randomly = rotate_randomly
        self.device = device
        
        self.L = L
        self.N = N
        self.sigma = sigma
        
        if elements is None:
            self.nchannel = nchannel
        else:
            self.nchannel = len(elements)
        
        self.shuffle = shuffle
        self.current = 0
        self.indices = np.arange(df.shape[0])
        self.N_data = len(self.indices)
        
        print(f"Initialised LazyMolLoader with {self.N_data} molecules. sigma={self.sigma}, L={self.L}, N={self.N}, nchannel={self.nchannel}, shuffle={self.shuffle}, rotate={self.rotate_randomly}, device={self.device}")
        
    def __iter__(self):
        return self
    
    def __next__(self):
         return self.next()

    def __len__(self):
        return int(np.ceil(self.N_data / self.batch_size))

    def next(self):
        N_data = self.N_data
        N = self.N # number of voxels
        L = self.L
        sigma = self.sigma
        nchan = self.nchannel
        
        if self.current == 0 and self.shuffle:
            np.random.shuffle(self.indices)
        
        if self.current < N_data:
            n1 = self.current
            n2 = min(self.current + self.batch_size, N_data)
            
            x_names = []
            x = np.zeros((n2-n1, nchan, N, N, N), dtype="float32")
            y = np.zeros((n2-n1,1), dtype="float32")
            
            for i, j in enumerate(range(n1, n2)):
                data_index = self.indices[j] # supports shuffling
                
                mol = self.df.iloc[data_index]
                
                R = np.eye(3)
                if self.rotate_randomly:
                    R = get_random_3D_rotation_matrix()
                
                if nchan == 1:
                    name, rd, y_i = single_channel_reciprocal_data(mol, sigma, L, N)
                    G, SG = rd
                    descriptor = make_voxel_grid(G, SG, L, N, rot=R)
                    x[i, 0, :, :, :] = descriptor
                    y[i] = y_i
                    x_names.append(name)
                
                else:
                    name, rd, y_i = multi_channel_reciprocal_data(mol, sigma, L, N, self.elements)
                    for j, element, (G, SG) in rd:
                        descriptor = make_voxel_grid(G, SG, L, N, rot=R)
                        x[i, j, :, :, :] = descriptor
                    y[i] = y_i
                    x_names.append(name)
                
                         
            self.current = n2
            
            x = torch.from_numpy(x).to(self.device, non_blocking=True)
            y = torch.from_numpy(y).to(self.device, non_blocking=True)
            
            return (x_names, x, y)
        
        self.current = 0
        # print("Stop Iteration")
        raise StopIteration()