#!/bin/bash

#===============================================================================
# SLURM Batch Script with Email Notification
# DeepSeek Inference
#===============================================================================

#SBATCH -D /users/aczd097/git/deepseek-inference/scripts    # Working directory
#SBATCH --job-name deepseek                                                 # Job name (8 characters or less)
#SBATCH --mail-type=ALL                                                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.sikar@city.ac.uk                                 # Where to send mail

#===============================================================================
# Resource Configuration
#===============================================================================

#SBATCH --partition=gengpu            # Partition choice: gengpu or preemptgpu
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node
#SBATCH --cpus-per-task=2             # CPUs per task
#SBATCH --mem=4GB                     # Expected CPU RAM needed
#SBATCH --time=72:00:00               # Time limit hrs:min:sec

#===============================================================================
# GPU Configuration
#===============================================================================

#SBATCH --gres=gpu:a100:1            # Request 1x A100 40GB GPU on gengpu partition, max 4 GPUs
##SBATCH --gres=gpu:a100_80g:1       # Uncomment to request 1x A100 80GB GPU on preemptgpu partition, max 2 GPUs

#===============================================================================
# Output Configuration
#===============================================================================

#SBATCH -e outputs/%x_%j.e             # Standard error log
#SBATCH -o outputs/%x_%j.o             # Standard output log
                                       # %j = job ID, %x = job name

#===============================================================================
# Environment Setup
#===============================================================================

# Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

# Clean environment
module purge

# Load required modules
module add gnu
# Add other required modules here if necessary

#===============================================================================
# Main Script
#===============================================================================

# Record start time
start=$(date +%s)

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo "Job started at $(date)"

# Execute the Python training script
python run_inference.py
#===============================================================================
# Email Job Output and Calculate Duration
#===============================================================================

# Get the output file path
output_file="outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o"

# Wait for file to be written
sleep 5

# Send last 100 lines by email
tail -n 100 "$output_file" | mail -s "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) Output" daniel.sikar@city.ac.uk

# Get the error file path
output_file="outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.e"

# Wait for file to be written
sleep 5

# Send last 100 lines by email
tail -n 100 "$output_file" | mail -s "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) Output" daniel.sikar@city.ac.uk

# Calculate execution time
end=$(date +%s)
diff=$((end - start))

# Convert seconds to hours, minutes, and seconds
hours=$((diff / 3600))
minutes=$(((diff % 3600) / 60))
seconds=$((diff % 60))

echo "Job completed at $(date)"
echo "Total execution time: $hours hours, $minutes minutes, $seconds seconds"