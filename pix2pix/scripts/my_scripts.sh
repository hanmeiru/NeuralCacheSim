# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10MaskedL1 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --model pix2pix0masked --input_nc 1 --output_nc 1 --dataset_mode trainunscaled --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --use_wandb

# python3 test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10  --model pix2pix0masked --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew --name Poly10MaskedL1

# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10PenalizeCount0 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --model pix2pixpen0 --input_nc 1 --output_nc 1 --dataset_mode trainunscaled --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --use_wandb

# python3 test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10  --model pix2pixpen0 --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew --name Poly10PenalizeCount0

# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10scalingby2 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --use_wandb --model pix2pix --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10  --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew2x --name Poly10scalingby2

# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10subtractby100 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --use_wandb --model pix2pix --input_nc 1 --output_nc 1 --dataset_mode train100minus --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4

# python3 test_by_benchmark_100minus.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10  --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1d100minus --name Poly10subtractby100

# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10PenalizeCount0_2 --batch_size 4 --no_flip --n_epochs 5 --n_epochs_decay 0 --lr 0.0002 --preprocess none --model pix2pixpen0 --input_nc 1 --output_nc 1 --dataset_mode trainunscaled --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --use_wandb

# python3 test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10  --model pix2pixpen0 --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew --name Poly10PenalizeCount0_2

# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10L1MaskedL2 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --model pix2pix0masked2 --input_nc 1 --output_nc 1 --dataset_mode trainunscaled --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --use_wandb

# python3 test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10  --model pix2pix0masked2 --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew --name Poly10L1MaskedL2

# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Polybench --name Poly32scalingby2 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --use_wandb --model pix2pix --input_nc 1 --output_nc 1 --dataset_mode train2x --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4

# python3 test_by_benchmark_1d_2x.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Polybench  --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew2x --name Poly32scalingby2

# more complex generator
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10LargerNgf --model pix2pix --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --use_wandb --input_nc 1 --output_nc 1 --dataset_mode trainunscaled --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128

# python3 test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10LargerNgf  --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew --netG unet_256 --netD n_layers --n_layers_D 4 --ngf 128

# larger lambda
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10GeneratorLosslambda150 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --use_wandb --model pix2pix --input_nc 1 --output_nc 1 --dataset_mode trainunscaled --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --lambda_L1 150

# python3 test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10GeneratorLosslambda150  --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew 

# # smaller lambda 
# python3 train.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10DiscriminatorLosslambda50 --batch_size 4 --no_flip --n_epochs 3 --n_epochs_decay 2 --lr 0.0002 --preprocess none --use_wandb --model pix2pix --input_nc 1 --output_nc 1 --dataset_mode trainunscaled --direction AtoB --netG unet_256 --netD n_layers --n_layers_D 4 --lambda_L1 50

# python3 test_by_benchmark_1d.py --dataroot /tank/projects/neuralcachesim/sparse-matrices/size1/64set-12way_sparse/Poly10 --name Poly10DiscriminatorLosslambda50 --model pix2pix --input_nc 1 --output_nc 1 --netG unet_256 --preprocess none --dataset_mode test1dnew 




