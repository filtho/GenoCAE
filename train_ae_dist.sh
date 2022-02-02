#!/bin/bash
#SBATCH --job-name=gcae_dist    # job name
#SBATCH --nodes=1               # Change this
#SBATCH --ntasks-per-node=1     # Don't change this
#SBATCH --gres=gpu:2            # Change this
#SBATCH --time=71:00:00         # job length
#SBATCH --cpus-per-task=8       # Change this, use 8 cpu cores per GPU (the quoted default by NSC. C)
#SBATCH --mem=125G              # Change this, NSC default is \approx 125 per GPU

set -x

# The additional argument in $1 is fed through to the next batch file, determines where output is saved

srun --mpi=pmi2 --exclusive singularity exec  --nv --bind /proj/gcae_berzelius/users/filip:/mnt /proj/gcae_berzelius/users/filip/image_dir/image.sif ./train_ae_dist_inner.sh $1
