'''
Double the pixel value (max = 200)
'''
import os
from data.base_dataset import BaseDataset, get_transform
from data.npz_folder import make_dataset_bench
from PIL import Image
import random
from scipy.sparse import load_npz
import numpy as np 
import os 
import matplotlib.pyplot as plt
from matplotlib import cm
import io

def remove_unpaired_ls(A_files, B_files):
    '''
    remove unpaired from lists of paths,
    not deleting original files.
    '''
    A_set = set(A_files)
    B_set = set(B_files)
    for a in A_set:
        if a.replace('A.npz','B.npz').replace('FULL','MISS') not in B_set:
            A_files.remove(a)
            print(os.path.basename(a), 'is removed')
    for b in B_set:
        if b.replace('B.npz','A.npz').replace('MISS','FULL') not in A_set:
            B_files.remove(b)
            print(os.path.basename(b), 'is removed')
    
class Train1d2x4ConfigDataset(BaseDataset):
    """
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
 
    def __init__(self, opt, config='', benchmark=''):

        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.config_list = ['1-64-12', '1-128-6','2-128-12','3-128-3']

        self.A_paths = []
        self.B_paths = []
        for conf in self.config_list:
            self.dir_train = os.path.join(opt.dataroot,conf,'TRAIN') 
            # /tank/.../size1/64set-12way_sparse/final_data/1-64-12/TRAIN 
       
            self.dir_A = os.path.join(self.dir_train,'FULL')
            self.dir_B = os.path.join(self.dir_train,'MISS')
        
            self.A_paths += make_dataset_bench(self.dir_A, opt.max_dataset_size)   # load images from '/path/to/data/FULL'
            self.B_paths += make_dataset_bench(self.dir_B, opt.max_dataset_size)    # load images from '/path/to/data/MISS' 

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        remove_unpaired_ls(self.A_paths, self.B_paths) # remove unpaired images

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert self.A_size == self.B_size 
        print("Train size: ", self.A_size)

        btoA = self.opt.direction == 'BtoA'
        
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within range
        B_path = self.B_paths[index_A]

        # get cache config from path
        # ex. /tank/projects/neuralcachesim/sparse-matrices/SPEC189_4configs/1-128-6/TRAIN/FULL/471.omnetpp-188/471.omnetpp-188B.champsimtrace.xz.txt_952_A.npz
        config = A_path.split('/TRAIN')[0].split('/')[-1] # 1-128-6
        cache_set = float(config.split('-')[1])
        cache_way = float(config.split('-')[-1])

        # convert A and B from sparse matrix to ndarray to img
        sparse_A = load_npz(A_path) # load sparse matrix from path
        array_A = np.array(sparse_A.toarray(), dtype = np.uint8) * 2 #convert into array
        sparse_B = load_npz(B_path) # load sparse matrix from path
        array_B = np.array(sparse_B.toarray(), dtype = np.uint8) * 2 #convert into array

        A_img = Image.fromarray(array_A.astype('uint8'), mode = 'L') 
        B_img = Image.fromarray(array_B.astype('uint8'), mode = 'L') 

        A = self.transform_A(A_img) 
        B = self.transform_B(B_img)
        

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'cache_set':cache_set, 'cache_way':cache_way}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

