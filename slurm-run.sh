#!/bin/zsh
#SBATCH --job-name="AutoTSAD"
#SBATCH --cpus-per-task 2
#SBATCH --mem=6G
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

# prepare temporary directory
mkdir -p /scratch/sebastian.schmidl/autotsad
chmod u+rwx /scratch/sebastian.schmidl
chmod og-rwx /scratch/sebastian.schmidl
chmod u+rwx /scratch/sebastian.schmidl/autotsad

# copy sources to /tmp for faster startup of processes (tmpfs = ramdisk)
mkdir -p /tmp/sebastian.schmidl/condaenvs
rsync -a --exclude .git --exclude results-* --exclude slurm*.out --exclude notebooks --exclude paper-plots --exclude scripts --exclude results --filter=':- .gitignore' --delete-after ${HOME}/projects/holistic-tsad /tmp/sebastian.schmidl/
rsync -a --delete-after ${HOME}/opt/miniconda3/envs/autotsad /tmp/sebastian.schmidl/condaenvs/

# start job
cd /tmp/sebastian.schmidl/holistic-tsad
export AUTOTSAD__GENERAL__TMP_PATH="/tmp/sebastian.schmidl/tmp-${SLURM_JOBID}"
#export AUTOTSAD__GENERAL__TMP_PATH=/scratch/sebastian.schmidl/autotsad
export AUTOTSAD__GENERAL__RESULT_PATH="${HOME}/projects/holistic-tsad/results-${SLURM_JOBID}"
#export AUTOTSAD__GENERAL__RESULT_PATH=${HOME}/projects/holistic-tsad/results
#export AUTOTSAD__GENERAL__N_JOBS=20
export AUTOTSAD__GENERAL__PROGRESS=off
logfile="${AUTOTSAD__GENERAL__RESULT_PATH}/${SLURM_JOBID}-screen.log"

mkdir -p "${AUTOTSAD__GENERAL__RESULT_PATH}"
/tmp/sebastian.schmidl/condaenvs/autotsad/bin/python -m autotsad run --config-path autotsad-exp-config.yaml "$@" 2>&1 | tee -a "${logfile}"
exit_code=$?

# create tarball of results
mkdir -p "${HOME}/projects/holistic-tsad/results-autotsad"
if [[ ${exit_code} -ne 0 ]]; then
  echo "Job failed with exit code ${exit_code}"
  tarball="${HOME}/projects/holistic-tsad/results-autotsad/$(date --rfc-3339=date)-${SLURM_JOBID}-${SLURM_JOB_NAME} (failed).tar.gz"
else
  tarball="${HOME}/projects/holistic-tsad/results-autotsad/$(date --rfc-3339=date)-${SLURM_JOBID}-${SLURM_JOB_NAME}.tar.gz"
fi
result_path=$(head -n 20 "${logfile}" | grep -e "RESULT directory" | cut -d '=' -f 2)
result_path=$(trim "${result_path}")
mkdir "${result_path}/tmp"
cp -r "${AUTOTSAD__GENERAL__TMP_PATH}/"* "${result_path}/tmp/"
cp "${logfile}" "${result_path}/screen.log"
cp autotsad-exp-config.yaml "${result_path}/autotsad-exp-config.yaml"
echo "$(hostname)" > "${result_path}/hostname.txt"
tar -czf "${tarball}" -C "${result_path}" '.'
