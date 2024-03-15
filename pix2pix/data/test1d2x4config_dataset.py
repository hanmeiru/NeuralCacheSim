'''
Used when dataroot contains TRAIN and TEST,
each containing FULL and MISS,
each containing benchmarks 
'''

import os
from data.base_dataset import BaseDataset, get_transform
from data.npz_folder import make_dataset
from PIL import Image
from scipy.sparse import load_npz
import numpy as np 
import os 

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
    

class Test1d2x4ConfigDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

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
         
        self.dir_A = os.path.join(opt.dataroot,config,'TEST/FULL',benchmark)  
        self.dir_B = os.path.join(opt.dataroot,config,'TEST/MISS',benchmark)  
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from benchmark folders of full heatmaps 
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from benchmark folders of miss heatmaps
        remove_unpaired_ls(self.A_paths, self.B_paths) # remove unpaired images
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert self.A_size == self.B_size 

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

        config = A_path.split('/TEST')[0].split('/')[-1] # 1-128-6
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

