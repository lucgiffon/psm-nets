#!/bin/bash
#SBATCH --job-name=compression_palm_act          # nom du job
#SBATCH --partition=gpu                                         # partition GPU choisie
#SBATCH --ntasks=1                                                 # nombre de tache MPI (= nombre de GPU ici)
#SBATCH --ntasks-per-node=1                                        # nombre de tache MPI par noeud
#SBATCH --gres=gpu:1                                               # nombre de GPU par nœud
#SBATCH --cpus-per-task=2                                          # nombre de coeurs CPU par tache
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --time=20:00:00                                           # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=%j.out                                           # nom du fichier de sortie
#SBATCH --error=%j.err                                            # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-84                                             # array batch indices availabale wiht $SLURM_ARRAY_TASK_ID
#SBATCH --mem=30GB

ROOT_DIR=/data1/home/luc.giffon/DeployedProjects/palmnet
PYTHON_EXEC=/data1/home/luc.giffon/anaconda3/envs/palmnet-gpu/bin/python
PARAMETER_FILE=$ROOT_DIR/parameters/2020/09/6_7_compression_palm_act.txt
SCRIPT_FILE=$ROOT_DIR/code/scripts/2020/09/6_7_compression_sparse_facto_new.py


SIZE_PARAMETER_FILE=$(wc -l $PARAMETER_FILE)

# if [ "$SLURM_ARRAY_TASK_ID" -gt "$SIZE_PARAMETER_FILE" ]; then
#    exit 0
# fi

# nettoyage des modules charges en interactif et herites par defaut
# module purge

# chargement des modules
# module load ...
# module load anaconda-py3/2019.03

# conda activate palmnet-gpu


LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p $PARAMETER_FILE)
echo $LINE

# echo des commandes lancees
set -x
# execution du code
srun $PYTHON_EXEC $SCRIPT_FILE $LINE