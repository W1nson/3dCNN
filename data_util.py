import h5py  
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch 
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam, RMSprop, lr_scheduler
import torch.nn as nn
import scipy as sp 


def voxelize_3d(xyz, feat, vol_dim=[19,48,48,48], relative_size=True, size_angstrom=48, atom_radii=None, atom_radius=1, sigma=0):

	# get 3d bounding box
	xmin = min(xyz[:, 0])
	ymin = min(xyz[:, 1])
	zmin = min(xyz[:, 2])
	xmax = max(xyz[:, 0])
	ymax = max(xyz[:, 1])
	zmax = max(xyz[:, 2])

	# initialize volume
	vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

	if relative_size:
		# voxel size (assum voxel size is the same in all axis
		vox_size = float(zmax - zmin) / float(vol_dim[1])
	else:
		vox_size = float(size_angstrom) / float(vol_dim[1])
		xmid = (xmin + xmax) / 2.0
		ymid = (ymin + ymax) / 2.0
		zmid = (zmin + zmax) / 2.0
		xmin = xmid - (size_angstrom / 2)
		ymin = ymid - (size_angstrom / 2)
		zmin = zmid - (size_angstrom / 2)
		xmax = xmid + (size_angstrom / 2)
		ymax = ymid + (size_angstrom / 2)
		zmax = zmid + (size_angstrom / 2)
		vox_size2 = float(size_angstrom) / float(vol_dim[1])
		#print(vox_size, vox_size2)

	# assign each atom to voxels
	for ind in range(xyz.shape[0]):
		x = xyz[ind, 0]
		y = xyz[ind, 1]
		z = xyz[ind, 2]
		if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
			continue

		# compute van der Waals radius and atomic density, use 1 if not available
		if not atom_radii is None:
			vdw_radius = atom_radii[ind]
			atom_radius = 1 + vdw_radius * vox_size

		cx = (x - xmin) / (xmax - xmin) * (vol_dim[3] - 1)
		cy = (y - ymin) / (ymax - ymin) * (vol_dim[2] - 1)
		cz = (z - zmin) / (zmax - zmin) * (vol_dim[1] - 1)

		vx_from = max(0, int(cx - atom_radius))
		vx_to = min(vol_dim[3] - 1, int(cx + atom_radius))
		vy_from = max(0, int(cy - atom_radius))
		vy_to = min(vol_dim[2] - 1, int(cy + atom_radius))
		vz_from = max(0, int(cz - atom_radius))
		vz_to = min(vol_dim[1] - 1, int(cz + atom_radius))

		for vz in range(vz_from, vz_to + 1):
			for vy in range(vy_from, vy_to + 1):
				for vx in range(vx_from, vx_to + 1):
						vol_data[:, vz, vy, vx] += feat[ind, :]

	# gaussian filter
	if sigma > 0:
		for i in range(vol_data.shape[0]):
			vol_data[i,:,:,:] = sp.ndimage.filters.gaussian_filter(vol_data[i,:,:,:], sigma=sigma, truncate=2)

	return vol_data







class Dataset_ligand(Dataset): 
    '''
    Class Dataset(df, types=['train', 'valid', 'test'], pose=[1, 10])

    stores the comp_ids from for all the keys from hdf5 file 

    outputs: 
        (data, label, ligand_poses) 
        each data matches with each ligand poses 
    '''

    def __init__(self, df, types, pose=None, sigma=0):
        super(Dataset_ligand, self).__init__()
        
        file_path = '.'
        postera = 'postera_protease2_pos_neg_{}.hdf5'.format(types)

        self.mlhdf = h5py.File(os.path.join(file_path, postera), 'r')
        self.pose = pose 
        self.sigma = sigma

        if types == 'val': 
            types = 'valid'
        self.df = df[df['subset'] == types].filter(['cmpd_id', 'smiles', 'label']) 

        # print(self.df)
        self.comp_ids = []

        for m in tqdm(self.mlhdf.keys()): 
            comp_id = m.split('_')[0]
            if not self.df[self.df['cmpd_id']==comp_id]['label'].empty:
                self.comp_ids.append(m)
            
    def close(self): 
        self.mlhdf.close()
        
    
    def __len__(self): 
        return len(self.comp_ids)
    
    
    def __getitem__(self, idx): 
        name = self.comp_ids[idx]

        comp_id = name.split('_')[0] 
        
        # get the label each compound
        labels = self.df[self.df['cmpd_id']==comp_id]['label']
        label = labels.reset_index(drop=True)[0]
        

        p = self.mlhdf[name]['ligand'][()]
        xyz = p[:, 0:3] 
        feats = p[:, 3:]
        voxel_arr = voxelize_3d(xyz=xyz, feat=feats, sigma=self.sigma) 

        return torch.tensor(voxel_arr), torch.tensor(label)
        # [19, 48, 48, 48], 1 or 0 












