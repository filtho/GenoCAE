#!/bin/bash
#SBATCH --job-name=gcae_dist     # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00             # job length
#SBATCH --cpus-per-task=10   
#SBATCH --mem=50G

set -x

srun --mpi=pmi2 --exclusive singularity exec --nv --bind /proj/gcae_berzelius/users/filip:/mnt /proj/gcae_berzelius/users/filip/image_dir/image.sif python3 -u run_gcae_distributed.py   train --datadir Data/     --data HumanOrigins2067_filtered --model_id M1 --epochs 50 --save_interval 2 --train_opts_id ex3  --data_opts_id b_0_4 --trainedmodeldir=/mnt/saved_runs/$1 
srun --mpi=pmi2 --exclusive  singularity exec --nv --bind /proj/gcae_berzelius/users/filip:/mnt /proj/gcae_berzelius/users/filip/image_dir/image.sif python3 -u run_gcae_distributed.py project --datadir Data/     --data HumanOrigins2067_filtered --model_id M1 --train_opts_id ex3  --data_opts_id b_0_4 --trainedmodeldir=/mnt/saved_runs/$1 --superpops example_tiny/HO_superpopulations 
srun --mpi=pmi2 --exclusive  singularity exec --nv --bind /proj/gcae_berzelius/users/filip:/mnt /proj/gcae_berzelius/users/filip/image_dir/image.sif python3 -u run_gcae_distributed.py evaluate --datadir Data/   --data HumanOrigins2067_filtered --model_id M1 --train_opts_id ex3  --data_opts_id b_0_4 --trainedmodeldir=/mnt/saved_runs/$1 --superpops example_tiny/HO_superpopulations --metrics "hull_error,f1_score_3" 
