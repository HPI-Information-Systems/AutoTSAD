#!/bin/zsh
# run with: export SLURM_JOB_NAME="tsadams ..."; ./run-tsadams.sh --n-jobs 1 path/to/dataset.csv 2>&1 | tee tsadams.log

set -o pipefail  # trace exit code of failed piped commands

pid=$$

trim() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    printf '%s' "$var"
}

if [ -z "${SLURM_JOB_NAME-}" ]; then
  echo "${SLURM_JOB_NAME} not set!"
  exit 1
fi
if [ -z "${SLURM_JOBID-}" ]; then
  echo "${SLURM_JOBID} not set, using PID!"
  SLURM_JOBID=${pid}
fi

echo "Processing Job ${SLURM_JOB_NAME}"

logfile="results-tsadams/${SLURM_JOBID}-screen.log"
mkdir -p "results-tsadams"
python -m autotsad baselines tsadams "$@" 2>&1 | tee -a "${logfile}"

# move logfile to results-folder
result_path=$(head -n 20 "${logfile}" | grep -e "RESULT directory" | cut -d '=' -f 2)
result_path=$(trim "${result_path}")
mv "${logfile}" "${result_path}/screen.log"
echo "$(hostname)" > "${result_path}/hostname.txt"
