#!/bin/bash -l
#SBATCH -J ablang                               # Job name
#SBATCH -A opig                                 # Project Account
#SBATCH --time=10-00:00:00                      # Walltime
#SBATCH --cpus-per-task=1
#SBATCH --wait-all-nodes=1                      # Wait until all nodes are ready (other breaks when using multiple gpus)
#SBATCH --mem=40000                             # total memory (in MB) ### commented out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1                     # number of  tasks which can run simultaneously
#SBATCH --nodes=1                               # number of nodes - you would usually only use nagagpu01 or 02
#SBATCH --chdir=/homes/olsen/                   # From where you want the job to be run
#SBATCH --partition=naga-gpu-large              # Select a specific partition rather than default
#SBATCH -w nagagpu02.cpu.stats.ox.ac.uk         # Provide a specific node/nodelist rather than the standard nodelist associated with the partition>
#SBATCH --output=/homes/olsen/slurm_out/ablang.o #%j.out  # Writes standard output to this file. %j is jobnumber
#SBATCH --error=/homes/olsen/slurm_out/ablang.e  #%j.out   # Writes error messages to this file. %j is jobnumber
# 90 seconds before training ends re-submit (training will continue from where it stopped)
#SBATCH --signal=SIGUSR1@90

export PATH="/homes/olsen/miniconda3/bin:$PATH"

export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#export NCCL_DEBUG_SUBSYS=COLL
#export NCCL_P2P_LEVEL=SYS

# debugging flags (optional)
#export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

#nvidia-smi topo -m

source /homes/olsen/miniconda3/etc/profile.d/conda.sh
conda activate ablang-train

#export NCCL_SOCKET_IFNAME=^docker0,lo
#export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /homes/olsen/projects/ablang-train/src

chmod 775 run_training.py
srun python3 run_training.py --name ablang \
                            --mode train \
                            --gpus 1 \
                            --num_training_steps 8500 \
                            --num_hidden_layers 4 \
                            --dataDir /homes/olsen/multi_gpu/data/11022022_data

