#!/bin/sh
#BATCH --job-name=midi-proj
#SBATCH --output gen%j.out
#SBATCH --error gen%j.err
#SBATCH -N 1
#SBATCH -n 14
#SBATCH -p defq

module load gcc/6.1.0
module load python2.7
source activate /home/cjuneau/.conda/envs/midi-env

if [ "$#" -ne 3 ]; then
  echo "Usage: ./generating.sh <path-to-input-dir> <path-to-weights-file> <composition-name>"
  echo "Exiting..."
  exit 1
fi

python generate.py $1 $2 $3
python -c "print('done')"
