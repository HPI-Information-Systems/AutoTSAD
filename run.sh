#!/bin/zsh

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

# start job
export AUTOTSAD__GENERAL__TMP_PATH="tmp-${pid}"
export AUTOTSAD__GENERAL__RESULT_PATH=results
export AUTOTSAD__GENERAL__N_JOBS=20
export AUTOTSAD__GENERAL__PROGRESS=off
logfile="${AUTOTSAD__GENERAL__RESULT_PATH}/${pid}-screen.log"

mkdir -p "${AUTOTSAD__GENERAL__RESULT_PATH}"
python -m autotsad run --config-path autotsad-exp-config.yaml "$@" 2>&1 | tee -a "${logfile}"
exit_code=$?

# create tarball of results
mkdir -p "results"
if [[ ${exit_code} -ne 0 ]]; then
  echo "Job failed with exit code ${exit_code}"
  tarball="results/$(date --rfc-3339=date)-${pid} (failed).tar.gz"
else
  tarball="results/$(date --rfc-3339=date)-${pid}.tar.gz"
fi
result_path=$(head -n 20 "${logfile}" | grep -e "RESULT directory" | cut -d '=' -f 2)
result_path=$(trim "${result_path}")
mkdir "${result_path}/tmp"
cp -r "${AUTOTSAD__GENERAL__TMP_PATH}/"* "${result_path}/tmp/"
cp "${logfile}" "${result_path}/screen.log"
cp autotsad-exp-config.yaml "${result_path}/autotsad-exp-config.yaml"
echo "$(hostname)" > "${result_path}/hostname.txt"
tar -czf "${tarball}" -C "${result_path}" '.'
