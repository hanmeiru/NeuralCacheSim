# SPEC 4 config (not splitted) - new base 
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/SPEC189_4configs --name SPEC189_4configs --model pix2pix --batch_size 4 --no_flip --n_epochs 2 --n_epochs_decay 1 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train1d2x4config --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 1 --ngf 128 --lambda_L1 150 --save_epoch_freq 1 --use_wandb
# python3 test_by_benchmark_1d_2x_4config.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/final_data --name SPEC189_4configs --model pix2pix --input_nc 1 --output_nc 1 --preprocess none --dataset_mode test1d2x4config  --netG unet_256 --netD n_layers --n_layers_D 1 --ngf 128 

# Poly 4 config (optimal base)
python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/Poly32_4configs --name Poly32_4configs --model pix2pix --batch_size 4 --no_flip --n_epochs 2 --n_epochs_decay 1 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train1d2x4config --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 1 --ngf 128 --lambda_L1 150 --save_epoch_freq 1 --use_wandb

python3 test_by_benchmark_1d_2x_4config.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/Poly32_4configs --name Poly32_4configs --model pix2pix --input_nc 1 --output_nc 1 --preprocess none --dataset_mode test1d2x4config  --netG unet_256 --netD n_layers --n_layers_D 1 --ngf 128 

