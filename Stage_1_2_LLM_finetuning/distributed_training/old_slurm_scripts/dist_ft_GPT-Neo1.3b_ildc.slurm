#!/bin/bash
#SBATCH --job-name=dist_ft_GPT-Neo1.3b_ildc        # name of job

#SBATCH --mail-type=ALL                        # All messages will be notified by email
#SBATCH --mail-user=Nishchal.Prasad@irit.fr     # E-mail address to receive notifications


# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU

##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)
##SBATCH --partition=gpu_p4          # uncomment for gpu_p4 partition (40GB A100 GPU)
#SBATCH -C a100                      # uncomment for gpu_p5 partition (80GB A100 GPU)
#SBATCH -A btm@a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)

# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:6                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
##SBATCH --cpus-per-task=4           # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --cpus-per-task=3           # number of cores per task for gpu_p2 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=6           # number of cores per task for gpu_p4 (1/8 of 8-GPUs node)
#SBATCH --cpus-per-task=8           # number of cores per task for gpu_p5 (1/8 of 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=dist_ft_GPT-Neo1.3b_ildc%j.out    # name of output file
#SBATCH --error=dist_ft_GPT-Neo1.3b_ildc%j.out     # name of error file (here, in common with the output file)
 
# Cleans out the modules loaded in interactive and inherited by default 
module purge
 
# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
module load cpuarch/amd
 
# Loading of modules
#module load pytorch-gpu/py3/2.0.0
#module load /gpfslocalsup/pub/anaconda-py3/2022.05/envs/tensorflow-gpu-2.11.0+py3.10.8

# Echo of launched commands
set -x
 
# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution
module load python
export PATH=$WORK/.local/bin:$PATH
#python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
#accelerate launch LEGAL-PE/SIGIR_experiments/distributedTraining/dist_ft_LLM_torch.py
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=6 LEGAL-PE/SIGIR_experiments/distributedTraining/dist_ft_LLM_torch.py
