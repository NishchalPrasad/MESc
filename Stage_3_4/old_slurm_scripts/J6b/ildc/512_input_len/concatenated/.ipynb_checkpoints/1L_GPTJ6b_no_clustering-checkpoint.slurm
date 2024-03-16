#!/bin/bash
#SBATCH --job-name=1L_GPTJ6b_no_cluster        # name of job

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
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
##SBATCH --cpus-per-task=4           # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --cpus-per-task=3           # number of cores per task for gpu_p2 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=6           # number of cores per task for gpu_p4 (1/8 of 8-GPUs node)
#SBATCH --cpus-per-task=24           # number of cores per task for gpu_p5 (1/8 of 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=LEGAL-PE/Level-3_of_Framework/slurm_scripts/ildc/outputs/1L_GPTJ6b_no_cluster%j.out    # name of output file
#SBATCH --error=LEGAL-PE/Level-3_of_Framework/slurm_scripts/ildc/outputs/1L_GPTJ6b_no_cluster%j.out     # name of error file (here, in common with the output file)
 
# Cleans out the modules loaded in interactive and inherited by default 
module purge
 
# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
module load cpuarch/amd
 
# Loading of modules
#module load pytorch-gpu/py3/2.0.0
#module load /gpfslocalsup/pub/anaconda-py3/2022.05/envs/tensorflow-gpu-2.11.0+py3.10.8
module load tensorflow-gpu/py3/2.11.0


# Echo of launched commands
set -x
mv $HOME/.local $WORK
ln -s $WORK/.local $HOME

# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution
python LEGAL-PE/Level-3_of_Framework/Level_three/training_models-DimRed+Clustering.py \
            --dataset_subset "ildc" \
            --layers_from_end 4 \
            --save_model_trained_here True\
            --self_ft_extracted True\
            --ft_model_used_for_extraction "LEGAL-P_E/SIGIR_experiments/finetuned_models/ildc/EleutherAI_gpt-j-6B/Strategy_1/sub_strategy_0/ZeRO3_epoch_0__"\
            --pretrained_model 'EleutherAI/gpt-j-6B'\
            --to_test False \
            --to_train True \
            --verbose 1 \
            --epochs 5 \
            --batch_size 1 \
            --num_layers 1 \
            --bilstm_before_encoder False\
            --dropout_before_first_fnn True\
            --dropout_before_inner_fnn True\
            --ffn_dropout_value 0.15 \
            --dropout_after_first_encoder True \
            --dropout_after_second_encoder True \
            --encoder_dropout_value 0.25 \
            --dff 4096 \
            --include_rnn False \
            --from_encoder False \
            --pad_len 150 \
            --strategy 1 \
            --data_path "LEGAL-P_E/SIGIR_experiments/finetuned_models/ildc/Extracted_data/J6b/"