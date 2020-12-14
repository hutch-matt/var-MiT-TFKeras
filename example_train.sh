#!/bin/bash
#SBATCH --job-name=EXAMPLE-train
#SBATCH -o %x.sh.log
#SBATCH -p normal
#SBATCH --constraint=xeon-g6
#SBATCH -N 8
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:volta:2

scontrol show hostname

source /etc/profile
module load anaconda/2020a
module load mpi/openmpi-4.0
module load cuda/10.1
module load nccl/2.5.6-cuda10.1

export FLAGS="--bind-to socket -map-by core"
export MPI_FLAGS="-mca pml ob1 -mca btl ^openib --report-bindings"

# Go to directory with the main.py script
cd /home/gridsan/groups/Moments_in_Time/AR-MiT

# Necessary for dataset reading on Supercloud
ulimit -n 1600

# Set arguments as desired.  See main.py for argument details.
which mpirun
mpirun ${FLAGS} ${MPI_FLAGS} -tag-output python main.py --job-id=${SLURM_JOBID} --job-name=${SLURM_JOB_NAME} --batch-size-per-gpu=32 --reports-folder=/home/gridsan/groups/Moments_in_Time/AR-MiT/reports --input-model=/home/gridsan/groups/Moments_in_Time/AR-MiT/models/VGG19-224x224x3-339-i.h5 --data-transform=1 --epochs=30 --training-set=/home/gridsan/mshutch/Moments_in_Time/data-copy/data/parsed/TrainingBatch_90 --output-model=/home/gridsan/groups/Moments_in_Time/AR-MiT/models/VGG19-224x224x3-339-im.h5 --early-stopping --validation-set=/home/gridsan/mshutch/Moments_in_Time/data-copy/data/parsed/ValidationBatch_90

# Necessary for dataset reading on Supercloud
ulimit -n 1024

echo ""
echo "all done"