#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=test4
#SBATCH --ntasks=20
#SBATCH --mem=20G
#SBATCH --time=2-12:0:0

##SBATCH --array=1-2
##SBATCH --mail-type=ALL
#SBATCH --mail-user=shelvin.chand@csiro.au

module load python/3.7.2
source $(which virtualenvwrapper_lazy.sh)
workon test
python params_gp.py

