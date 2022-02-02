#!/bin/bash

TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_max_cluster_size=100 " python3 -u run_gcae_pheno_dist.py train --datadir Data/     --data HumanOrigins2067_filtered --model_id M3j10X --epochs 20 --save_interval 2 --train_opts_id CCE295_L04_R10_N02_B05_E13  --data_opts_id d_0_4_160854 --trainedmodeldir=/mnt/comparison_pheno/seed_test/$1

python3 -u run_gcae_pheno_dist.py project --datadir Data/     --data HumanOrigins2067_filtered --model_id M3j10X --train_opts_id CCE295_L04_R10_N02_B05_E13  --data_opts_id d_0_4_160854 --trainedmodeldir=/mnt/comparison_pheno/seed_test/$1 --superpops example_tiny/HO_superpopulations  
