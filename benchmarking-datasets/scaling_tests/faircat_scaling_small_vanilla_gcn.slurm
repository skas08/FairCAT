#!/bin/bash
#SBATCH --job-name=faircat_scaling_15_vanilla_gcn
#SBATCH --output=faircat_scaling_15_vanilla_gcn_output_%j.txt
#SBATCH --partition=mcs.gpu.q   
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1

module load Python/3.10.13-GCCcore-11.3.0
module load CUDA/12.1.1

cd #!/bin/bash
#SBATCH --job-name=faircat_balancing_balanced_vanilla_gcn
#SBATCH --output=faircat_balancing_balanced_vanilla_gcn_output_%j.txt
#SBATCH --partition=PARTITION_NAME 
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1

module load Python/3.10.13-GCCcore-11.3.0
module load CUDA/12.1.1

cd $HOME/MUSUBI-FairGraphBase
source venv/bin/activate

export PATH="$HOME/MUSUBI-FairGraphBase/bin:$PATH"


python main.py --optimize --optrials 20 --dataset faircat_balancing_balanced --inprocessing vanilla --encoder gcn --metrics alpha

source venv/bin/activate

export PATH="/home/MUSUBI-FairGraphBase/bin:$PATH"


python main.py --optimize --optrials 20 --dataset faircat_scaling_15 --inprocessing vanilla --encoder gcn --metrics alpha
