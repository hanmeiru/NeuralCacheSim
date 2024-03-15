'''
No post-processing (check data/test1dnew_dataset.py).
Test for config 1-64-12.
Files saved as npz in config/benchmark folders.
No use of wandb.
'''

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import torch
import scipy.sparse
from scipy.sparse import load_npz
import numpy as np 
from PIL import Image
import time


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # opt.model = 'pix2pix'
    opt.netG = 'unet_256'
    model = create_model(opt)   # create a model given opt.model and other options
    model.setup(opt) # !!!!!! load model 
    if opt.eval:
        model.eval()

    testA_path = os.path.join(opt.dataroot, 'TESTALL/FULL')
    testB_path = os.path.join(opt.dataroot, 'TESTALL/MISS')

    try:
        os.mkdir(os.path.join(opt.dataroot,'fakeB'))
    except:
        pass

    # fakeB_dir = os.path.join(opt.dataroot,'fakeB',opt.name) 
    tank_root = '/tank/projects/neuralcachesim'
    fakeB_dir = os.path.join(tank_root,'fakeB',opt.name)

    try:
        os.mkdir(fakeB_dir)
    except:
        pass
        
    # keep a .txt file to save numerical results
    # saved under the dir of experiment name
    file1 = open(fakeB_dir+"/metric.txt","a")

    count_all = 0 # total number of images across all configs and benchmarks 
    mse_sum_all = 0
    baseline_sum_all = 0
    duration_sum_all = 0
    config_num = 0 # total number of configs
    bench_num_all = 0 # total number of benchmarks in all configs
    config_list = ['1-64-12']
    
    for config in config_list:
        mse_sum_config = 0
        baseline_sum_config = 0
        duration_sum_config = 0
        count_config = 0 # number of images in a config
        bench_num_per_config = 0 # number of benchmarks in one config
        config_num += 1
        
        # Create a folder in tank; 
        # Ex. /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/experiment_MISS/experiment1/464.h264ref-30B
        size,set,way = config.split('-')[0],config.split('-')[1],config.split('-')[2]
        try:
            os.mkdir(os.path.join(fakeB_dir,config))
        except:
            pass
        
        bench_list = os.listdir(testA_path)
        for bench in bench_list:
            mse_sum_bench = 0
            baseline_sum_bench = 0
            duration_sum_bench = 0
            bench_num_per_config += 1 
            bench_num_all += 1
            # Create sub-folder in each config (in tank); 
            #tank_bench_dir = os.path.join(tank_expetiment_name_dir,bench) # tank
            bench_dir = os.path.join(fakeB_dir,config,bench) 
            try:
                os.mkdir(bench_dir)
            except:
                pass

            # Create dataset for current config and benchmark
            dataset = create_dataset(opt,test=True,config=config,benchmark=bench)
            
            print("Length of dataset: ",len(dataset))
            for i, data in enumerate(dataset):
                start_time = time.time()
                # if i >= 15:  # only apply our model to opt.num_test images.
                #     break
                
                model.set_input(data)  # unpack data from data loader
                with torch.no_grad():
                    model.forward()  # run inference, forward() will update model.fake_B
        
                # no scale 
                unscaled_fake_B = util.tensor2im(model.fake_B)[:,:,0] # tensor2im gives (512,512,3)
                unscaled_real_B = util.tensor2im(model.real_B)[:,:,0]
                unscaled_real_A = util.tensor2im(model.real_A)[:,:,0]
    
                # end time
                end_time = time.time()
                duration = end_time - start_time

                # Calculate the per-pixel MSE for the images directly output from the model
                # this would be a naive measurement for how well the model generates images
                mse = ((unscaled_real_B - unscaled_fake_B) ** 2).sum().item() / (512*512)
                baseline = ((unscaled_real_B - unscaled_real_A) ** 2).sum().item() / (512*512)
            
                # accumulate 
                duration_sum_bench += duration
                duration_sum_config += duration
                duration_sum_all += duration
                mse_sum_all += mse 
                mse_sum_bench += mse
                mse_sum_config += mse
                baseline_sum_all += baseline
                baseline_sum_bench += baseline
                baseline_sum_config += baseline
                count_all += 1
                count_config += 1

                # convert to sparse matrix
                sparse_coo = scipy.sparse.coo_matrix(unscaled_fake_B)
                # save sparse as npz 
                file_name = model.image_paths[0].split('/')[-1].split(config)[-1][1:].replace('_A.npz','_fakeB.npz')
                obj_path_coo = os.path.join(bench_dir,file_name)
                scipy.sparse.save_npz(obj_path_coo, sparse_coo)
                print('saving ',obj_path_coo)
                # break
            mse_avg_bench = mse_sum_bench / (i+1)
            baseline_avg_bench = baseline_sum_bench / (i+1)
            file1.write('Average MSE across '+config+'/'+bench+': '+str(mse_avg_bench))
            file1.write('\n')
            file1.write('Average Baseline across '+config+'/'+bench+': '+str(baseline_avg_bench))
            file1.write('\n')
            file1.write('Total Inference Time for '+config+'/'+bench+': '+str(duration_sum_bench))
            file1.write('\n\n')
            # test only 5 benchmarks per config 
            # if bench_num_per_config >= 3:
            #     break   
            # break
        mse_avg_config = mse_sum_config / count_config
        baseline_avg_config = baseline_sum_config / count_config
        duration_avg_config = duration_sum_config / bench_num_per_config
        file1.write('Average MSE Across '+config+': '+str(mse_avg_config))
        file1.write('\n')
        file1.write('Average Baseline Across '+config+': '+str(baseline_avg_config))
        file1.write('\n')  
        file1.write('Average Benchmark Inference Time Across '+config+': '+str(duration_avg_config))
        file1.write('\n\n')
        # if config_num >= 1:
        #     break
        # break
    duration_avg_all = duration_sum_all / bench_num_all
    file1.write('Overall MSE: ' + str(mse_sum_all / count_all) + ' ')
    file1.write('\n')
    file1.write('Overall Baseline: ' + str(baseline_sum_all / count_all) + '\n')
    file1.write('Average Benchmark Inference Time Across All Benchmarks : ' + str(duration_avg_all) + '\n')
    file1.close()


