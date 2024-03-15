# # Poly32_newbase
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Polybench --name Poly32_newbase --model pix2pix --batch_size 4 --no_flip --n_epochs 2 --n_epochs_decay 0 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 --lambda_L1 150 --use_wandb --continue_train

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Polybench --name Poly32_newbase --model pix2pix --input_nc 1 --output_nc 1 --preprocess none --dataset_mode test1dnew2x  --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 

# Poly32_newbase_PixelD
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Polybench --name Poly32_newbase_PixelD --model pix2pix --batch_size 2 --no_flip --n_epochs 2 --n_epochs_decay 1 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD pixel --ngf 128 --lambda_L1 150 --use_wandb

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Polybench --name Poly32_newbase_PixelD --model pix2pix --input_nc 1 --output_nc 1 --preprocess none --dataset_mode test1dnew2x  --netG unet_256 --netD pixel --ngf 128 
