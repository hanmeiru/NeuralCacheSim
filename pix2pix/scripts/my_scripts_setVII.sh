# baseline: Poly10GenLossLambda150Ngf128Scalingby2
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10GenLossLambda150Ngf128Scalingby2 --model pix2pix --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 --lambda_L1 150 --use_wandb

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10GenLossLambda150Ngf128Scalingby2 --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew2x  --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 

# Poly10PenalizeLinesPerPixelL1lossOnlyReal
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10PenalizeLinesPerPixelL1lossOnlyReal --model pix2pixmaskedreall1 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 --lambda_L1 150 --use_wandb

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10PenalizeLinesPerPixelL1lossOnlyReal --model pix2pixmaskedreall1 --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew2x  --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 

# # Poly10PenalizeLinesPerPixelL2lossOnlyReal
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10PenalizeLinesPerPixelL2lossOnlyReal --model pix2pixmaskedreall2 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 --lambda_L1 150 --use_wandb

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10PenalizeLinesPerPixelL2lossOnlyReal --model pix2pixmaskedreall2 --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew2x  --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 

# # test Poly10 with lambda 10
# test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10  --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew --name smallPoly10_lambda10

# SPEC189GenLossLambda150Ngf128Scalingby2
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/SPEC189 --name SPEC189GenLossLambda150Ngf128Scalingby2 --model pix2pix --batch_size 4 --no_flip --n_epochs 2 --n_epochs_decay 1 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 --lambda_L1 150 --use_wandb

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/SPEC189 --name SPEC189GenLossLambda150Ngf128Scalingby2 --model pix2pix --input_nc 1 --output_nc 1 --preprocess none --dataset_mode test1dnew2x --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128 

# # Cropping experiments
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10_5epochCrop256_patchGAN --model pix2pix --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess crop --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --crop_size 256 --ngf 128


# SPEC189GenLossLambda150Ngf128Scalingby2PixelD
python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/SPEC189 --name SPEC189GenLossLambda150Ngf128Scalingby2PixelD --model pix2pix --batch_size 2 --no_flip --n_epochs 2 --n_epochs_decay 1 --lr 0.0002 --preprocess none --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD pixel --ngf 128 --lambda_L1 150 --use_wandb

python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/SPEC189 --name SPEC189GenLossLambda150Ngf128Scalingby2PixelD --model pix2pix --input_nc 1 --output_nc 1 --preprocess none --dataset_mode test1dnew2x --netG unet_256 --netD pixel --ngf 128 

