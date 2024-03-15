'''
Post-process by converting 3d image to 1d and unscaling (check data/test3dnew_dataset.py)
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
    opt.model = 'pix2pix'
    opt.netG = 'unet_256'
    model = create_model(opt)   # create a model given opt.model and other options
    model.setup(opt) # !!!!!! load model 
    if opt.eval:
        model.eval()

    testA_path = os.path.join(opt.dataroot, 'TEST/FULL')
    testB_path = os.path.join(opt.dataroot, 'TEST/MISS')

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
        tank_experiment_MISS_dir = os.path.join(tank_root, 'size'+size, set+'set-'+way+'way_sparse','experiment_MISS')
        # print(experiment_MISS_dir)
        try:
            os.mkdir(os.path.join(fakeB_dir,config))
            # os.mkdir(tank_experiment_MISS_dir)
        except:
            pass
        
        bench_list = os.listdir(testA_path)

        for bench in bench_list:
            # '456.hmmer-88B', '473.astar-42B', '481.wrf-1281B','fdtd_2d.'
            # if bench not in ['2mm.','403.gcc-16B','603.bwaves_s-1740B','649.fotonik3d_s-7084B','458.sjeng-767B','symm.']:
            #     continue
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
            
            for i, data in enumerate(dataset):
                start_time = time.time()
                
                model.set_input(data)  # unpack data from data loader
                with torch.no_grad():
                    model.forward()  # run inference, forward() will update model.fake_B
        
              
                fake_B_255 = util.tensor2im(model.fake_B) # tensor2im gives (512,512,3)
                real_B_255 = util.tensor2im(model.real_B)
                real_A_255 = util.tensor2im(model.real_A)
               
                # Post processing (for saving the image and hit-rate calculation):
                # Note: check test3dnew_dataset for how the arrays are scaled 
                # reversed img on scale of 0-255
                rev_fb = 255 - fake_B_255 
                rev_rb = 255 - real_B_255
                rev_ra = 255 - real_A_255
                # 2nd channel should be 0  
                unscaled_fake_B = (rev_fb[:,:,0]+rev_fb[:,:,1]*256)/200
                unscaled_real_B = (rev_rb[:,:,0]+rev_rb[:,:,1]*256)/200
                unscaled_real_A = (rev_ra[:,:,0]+rev_ra[:,:,1]*256)/200
    
                # end time
                end_time = time.time()
                duration = end_time - start_time

                # Calculate the per-pixel MSE for the images directly output from the model
                # this would be a naive measurement for how well the model generates images
                mse = ((real_B_255 - fake_B_255) ** 2).sum().item() / (512*512*3)
                baseline = ((real_B_255 - real_A_255) ** 2).sum().item() / (512*512*3)
                # print('mse: ',mse)
                # print('base: ', baseline)
                # print('hit rate estimate: ', 1-unscaled_fake_B.sum()/unscaled_real_A.sum())
                # print('hit rate actual: ', 1-unscaled_real_B.sum()/unscaled_real_A.sum())
            
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
                # obj_path = os.path.join(tank_bench_dir,file_name)
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
            # if bench_nu m_per_config >= 3:
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


