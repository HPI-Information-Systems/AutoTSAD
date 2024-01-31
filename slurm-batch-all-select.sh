#!/bin/zsh

datasets=(
  "IOPS/KPI-a8c06b47-cc41-3738-9110-12df0ee4c721"
  "IOPS/KPI-c69a50cf-ee03-3bd7-831e-407d36c7ee91"
  "IOPS/KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa"
  "IOPS/KPI-f0932edd-6400-3e63-9559-0a9860a1baa9"
  "IOPS/KPI-57051487-3a40-3828-9084-a12f7f23ee38"
  "IOPS/KPI-847e8ecc-f8d2-3a93-9107-f367a0aab37d"
  "IOPS/KPI-431a8542-c468-3988-a508-3afd06a218da"
  "IOPS/KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0"
  "IOPS/KPI-ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa"
  "IOPS/KPI-54350a12-7a9d-3ca8-b81f-f886b9d156fd"
  "KDD-TSAD/152_UCR_Anomaly_PowerDemand1"
  "KDD-TSAD/114_UCR_Anomaly_CIMIS44AirTemperature2"
  "KDD-TSAD/163_UCR_Anomaly_apneaecg2"
  "KDD-TSAD/030_UCR_Anomaly_DISTORTEDInternalBleeding19"
  "KDD-TSAD/070_UCR_Anomaly_DISTORTEDltstdbs30791AI"
  "KDD-TSAD/102_UCR_Anomaly_NOISEMesoplodonDensirostris"
  "KDD-TSAD/022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4"
  "KDD-TSAD/174_UCR_Anomaly_insectEPG2"
  "KDD-TSAD/202_UCR_Anomaly_CHARISfive"
  "KDD-TSAD/208_UCR_Anomaly_CHARISten"
  "MGAB/3"
  "MGAB/6"
  "MGAB/2"
  "MGAB/4"
  "NAB/art_increase_spike_density"
  "NAB/ec2_network_in_257a54"
  "NAB/Twitter_volume_AMZN"
  "NAB/rds_cpu_utilization_e47b3b"
  "NAB/ec2_cpu_utilization_24ae8d"
  "NAB/art_load_balancer_spikes"
  "NAB/ec2_cpu_utilization_5f5533"
  "NAB/Twitter_volume_CVS"
  "NAB/rds_cpu_utilization_cc0c53"
  "NAB/art_daily_jumpsdown"
  "NASA-MSL/P-15"
  "NASA-MSL/T-12"
  "NASA-MSL/C-2"
  "NASA-MSL/T-5"
  "NASA-MSL/M-7"
  "NASA-MSL/P-14"
  "NASA-MSL/F-5"
  "NASA-MSL/D-14"
  "NASA-MSL/T-8"
  "NASA-MSL/M-6"
  "NASA-SMAP/P-4"
  "NASA-SMAP/E-5"
  "NASA-SMAP/G-3"
  "NASA-SMAP/E-7"
  "NASA-SMAP/D-13"
  "NASA-SMAP/D-5"
  "NASA-SMAP/S-1"
  "NASA-SMAP/G-4"
  "NASA-SMAP/E-1"
  "NASA-SMAP/A-1"
  "NormA/Discords_annsgun"
  "NormA/Discords_dutch_power_demand"
  "NormA/SinusRW_Length_106000_AnomalyL_100_AnomalyN_60_NoisePerc_0"
  "NormA/Discords_marotta_valve_tek_14"
  "NormA/SinusRW_Length_104000_AnomalyL_200_AnomalyN_20_NoisePerc_0"
  "NormA/Discords_marotta_valve_tek_17"
  "NormA/SinusRW_Length_108000_AnomalyL_200_AnomalyN_40_NoisePerc_0"
  "TSB-UAD-artificial/-37_2_0.02_25"
  "TSB-UAD-artificial/-69_2_0.02_15"
  "TSB-UAD-artificial/-69_2_0.02_25"
  "TSB-UAD-artificial/-104_2_0.02_35"
  "TSB-UAD-synthetic/YAHOO_flat_region_0.02-Yahoo_A2synthetic_41_data"
  "TSB-UAD-synthetic/YAHOO_add_random_walk_trend_0.2-YahooA4Benchmark-TS28_data"
  "TSB-UAD-synthetic/YAHOO_change_segment_resampling_0.08-Yahoo_A2synthetic_93_data"
  "TSB-UAD-synthetic/YAHOO_change_segment_resampling_0.04-Yahoo_A2synthetic_32_data"
  "TSB-UAD-synthetic/YAHOO_change_segment_resampling_0.02-YahooA3Benchmark-TS84_data"
  "TSB-UAD-synthetic/YAHOO_flat_region_0.02-Yahoo_A1real_14_data"
  "TSB-UAD-synthetic/KDD21_change_segment_resampling_0.02-006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2_4000_5703_5727"
  "TSB-UAD-synthetic/YAHOO_flip_segment_0.08-YahooA4Benchmark-TS55_data"
  "TSB-UAD-synthetic/YAHOO_change_segment_resampling_0.04-YahooA3Benchmark-TS85_data"
  "TSB-UAD-synthetic/YAHOO_flat_region_0.04-YahooA3Benchmark-TS51_data"
  "WebscopeS5/A4Benchmark-63"
  "WebscopeS5/A2Benchmark-22"
  "WebscopeS5/A1Benchmark-2"
  "WebscopeS5/A4Benchmark-13"
  "WebscopeS5/A1Benchmark-25"
  "WebscopeS5/A4Benchmark-34"
  "WebscopeS5/A1Benchmark-34"
  "WebscopeS5/A4Benchmark-2"
  "WebscopeS5/A3Benchmark-54"
  "WebscopeS5/A3Benchmark-18"
)

gutentag_datasets=(
  "poly-diff-count-5"
  "rw-combined-diff-2"
  "ecg-same-count-1"
  "ecg-type-pattern-shift"
  "sinus-position-middle"
  "rw-length-100"
  "sinus-type-mean"
  "poly-length-500"
  "ecg-type-mean"
  "ecg-diff-count-1"
)

sand_datasets=(
  "803"
  "806"
  "SED"
  "803_806"
  "803_820"
  "803_SED"
  "806_SED"
  "803_805_820"
  "805_806_820"
  "803_805_806_820"
  "803_806_820_SED"
)

synthetic_datasets=(
  "gt-0"
#  "gt-1"
  "gt-2"
  "gt-3"
  "gt-4"
#  "gt-5"
#  "gt-6"
#  "gt-7"
#  "gt-8"
#  "gt-9"
)

n_jobs=1

#for d in "${datasets[@]}"; do
#  echo "Submitting Job >SELECT ${d} p=1<"
#  sbatch -J "SELECT ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy vertical "${HOME}/data/benchmark-data/univariate/${d}.test.csv"
#  sleep 0.5
#  sbatch -J "SELECT ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy horizontal "${HOME}/data/benchmark-data/univariate/${d}.test.csv"
#  sleep 0.5
#done
#
#for d in "${gutentag_datasets[@]}"; do
#  echo "Submitting Job >SELECT ${d} p=1<"
#  sbatch -J "SELECT GutenTAG ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy vertical "${HOME}/data/univariate-anomaly-test-cases/${d}/test.csv"
#  sleep 0.5
#  sbatch -J "SELECT GutenTAG ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy horizontal "${HOME}/data/univariate-anomaly-test-cases/${d}/test.csv"
#  sleep 0.5
#done
#
#for d in "${sand_datasets[@]}"; do
#  echo "Submitting Job >SELECT ${d} p=1<"
#  sbatch -J "SELECT SAND ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy vertical "${HOME}/data/sand-data/${d}.csv"
#  sleep 0.5
#  sbatch -J "SELECT SAND ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy horizontal "${HOME}/data/sand-data/${d}.csv"
#  sleep 0.5
#done

for d in "${synthetic_datasets[@]}"; do
  echo "Submitting Job >SELECT ${d}<"
  sbatch -J "SELECT ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy vertical "${HOME}/data/autotsad-data/synthetic/${d}.csv"
  sleep 0.5
  sbatch -J "SELECT ${d//\// } p=1" slurm-run-select.sh --n-jobs "${n_jobs}" --strategy horizontal "${HOME}/data/autotsad-data/synthetic/${d}.csv"
  sleep 0.5
done
