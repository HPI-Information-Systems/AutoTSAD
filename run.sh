#!/bin/zsh
# run with: export SLURM_JOB_NAME="AutoTSAD ..."; ./run.sh path/to/data.csv 2>&1 | tee autotsad.log

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
  echo "SLURM_JOB_NAME not set!"
  exit 1
fi
if [ -z "${SLURM_JOBID-}" ]; then
  echo "SLURM_JOBID not set, using PID!"
  SLURM_JOBID=${pid}
fi

echo "Processing Job ${SLURM_JOB_NAME}"

# start job
export AUTOTSAD__GENERAL__TMP_PATH="/tmp/sebastian.schmidl/tmp-${SLURM_JOBID}"
export AUTOTSAD__GENERAL__RESULT_PATH="${HOME}/projects/holistic-tsad/results"
export AUTOTSAD__GENERAL__N_JOBS=1
export AUTOTSAD__GENERAL__PROGRESS=off
export AUTOTSAD__DATA_GEN__DISABLE_CLEANING="no"
export AUTOTSAD__OPTIMIZATION__DISABLED="yes"

logfile="${AUTOTSAD__GENERAL__RESULT_PATH}/${SLURM_JOBID}-screen.log"

mkdir -p "${AUTOTSAD__GENERAL__RESULT_PATH}"
python -m autotsad run --config-path autotsad-exp-config.yaml "$@" 2>&1 | tee -a "${logfile}"
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
