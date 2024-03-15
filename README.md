# NeuralCacheSim

Within pix2pix (based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix):
 * **models** contains pix2pix models with experiments on different loss functions and cache parameter incorporation
 * **data** contains different image loading methods such as scaling vs. unscaling, representing images with 1 channel vs. 3 channels, spliting & shifting bits or not, etc. When we ran tests, we need to evaluate the performance by calculating the hitrates, which is based on the original image (no scaling/bit shifits) so we need to reverse the preprocessing based on the data loading methods (this is why we have multiple testxxx.py), ex. when we preprocessed the images (in both train & test) by scaling it by 2, we scaled it by 1/2 in test_by_benchmark_1d_2x.py to get original heatmaps.   
 * **scripts** contains scripts for experiments across various datasets (specified by --dataroot), models (specified by --model), image loading methods (specified by --dataset_mode),  and other hyperparameters and model architecture choice.
   

 
