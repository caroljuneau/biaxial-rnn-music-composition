#!/bin/sh
#SBATCH --job-name=midi-proj
#SBATCH -N 1
#SBATCH -n 14
#SBATCH --gres=gpu:1
#SBATCH --output job%j.out
#SBATCH --error job%j.err
#SBATCH -p gpu-v100-32gb

module load python2.7
module load cuda/9.0
source activate /home/cjuneau/.conda/envs/midi-env

if [ "$#" -ne 1 ]; then
  echo "Usage: ./training.sh <path-to-midi-files-dir>"
  echo "Exiting..."
  exit 1
fi

if ! [ -d $1 ]; then 
  echo "$1 not a directory"
  echo "Exiting..."
  exit 1
fi 

INPUT_DIR=$1

NOW=$(date +"%F_%H-%M-%S")
OUTPUT_DIR="training_$NOW"
mkdir output/$OUTPUT_DIR

THEANO_FLAGS="device=cuda,floatX=float32,cxx=''" python training_script.py $INPUT_DIR output/$OUTPUT_DIR 

conda deactivate 
python -c "print('done')"
