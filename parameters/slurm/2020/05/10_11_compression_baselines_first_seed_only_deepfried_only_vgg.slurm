#!/bin/bash
#SBATCH --job-name=deepfried_only_vgg_seed_1                                       # nom du job
#SBATCH --partition=gpu_p1                                         # partition GPU choisie
#SBATCH --ntasks=1                                                 # nombre de tache MPI (= nombre de GPU ici)
#SBATCH --ntasks-per-node=1                                        # nombre de tache MPI par noeud
#SBATCH --gres=gpu:1                                               # nombre de GPU par nœud
#SBATCH --cpus-per-task=20                                          # nombre de coeurs CPU par tache (un quart du noeud ici)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread                                      # hyperthreading desactive
#SBATCH --time=95:00:00                                           # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=%j.out                                           # nom du fichier de sortie
#SBATCH --error=%j.err                                            # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-3                                             # array batch indices availabale wiht $SLURM_ARRAY_TASK_ID
#SBATCH --qos=qos_gpu-t4									 	   # autre option qos_gpu-t3 par defaut pour job de 20h sinon qos_gpu-dev	 pour les jobs courts

PARAMETER_FILE=/gpfsdswork/projects/rech/hpp/ulk32cz/DeployedProjects/palmnet/parameters/2020/05/10_11_compression_baselines_first_seed_only_deepfried_only_vgg.txt
SCRIPT_FILE=/gpfsdswork/projects/rech/hpp/ulk32cz/DeployedProjects/palmnet/code/scripts/2020/05/10_11_compression_baselines.py


SIZE_PARAMETER_FILE=$(wc -l $PARAMETER_FILE)

# if [ "$SLURM_ARRAY_TASK_ID" -gt "$SIZE_PARAMETER_FILE" ]; then
#    exit 0
# fi

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
# module load ...
module load anaconda-py3/2019.03

conda activate tf14


LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p $PARAMETER_FILE)
echo $LINE

# echo des commandes lancees
set -x
# execution du code
srun python $SCRIPT_FILE $LINE