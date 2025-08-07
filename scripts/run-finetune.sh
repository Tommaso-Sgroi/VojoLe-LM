#!/bin/sh
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=try25_sgroi ### NOME DEL PROGETTO QUA
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module load python/3.11.7
module load profile/deeplrn
module load cuda/12.1
module load nvhpc
module load gcc/12.2.0

export CUDA=$CUDA_HOME

source $WORK/VojoLe-LM/.venv_unsloth/bin/activate
cd $WORK/VojoLe-LM

python3 test/try_unsloth.py --run


# sbatch -J prod_generation --output=$WORK/VojoLe-LM/logs/ds_to_db.out --error=$WORK/VojoLe-LM/logs/ds_to_db.err ./tmp.sh