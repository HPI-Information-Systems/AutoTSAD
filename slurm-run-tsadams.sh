#!/bin/zsh
#SBATCH --job-name="tsadams"
#SBATCH --cpus-per-task 10
#SBATCH --mem=25G
#SBATCH -t 12:00:00
#SBATCH -A naumann
#SBATCH -p magic
#SBATCH --constraint=ARCH:X86

#source ~/.zshrc
set -o pipefail  # trace exit code of failed piped commands

trim() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    printf '%s' "$var"
}

echo "Processing Job ${SLURM_JOB_NAME}"

logfile="results/${SLURM_JOBID}-screen.log"
mkdir -p "results"
/hpi/fs00/home/sebastian.schmidl/opt/miniconda3/envs/autotsad/bin/python -m autotsad baselines tsadams "$@" 2>&1 | tee -a "${logfile}"

# copy logfile to results-folder
result_path=$(head -n 20 "${logfile}" | grep -e "RESULT directory" | cut -d '=' -f 2)
result_path=$(trim "${result_path}")
cp "${logfile}" "${result_path}/screen.log"
echo "$(hostname)" > "${result_path}/hostname.txt"
