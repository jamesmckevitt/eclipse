#!/bin/sh
#SBATCH -J instr_resp
#SBATCH -N 1
#SBATCH --partition=zen3_0512
#SBATCH --qos=p71867_0512
#SBATCH --ntasks-per-node=128
#SBATCH -A p70652
#SBATCH --output=job.out

source $DATA/venvs/solar/bin/activate

python instr_response.py --config example_config.yaml