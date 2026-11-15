#!/bin/sh
export LD_LIBRARY_PATH=/data/local/tmp/snpebm_222/artifacts/aarch64-oe-linux-gcc8.2/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/snpebm_222/artifacts/aarch64-oe-linux-gcc8.2/lib;/system/lib/rfsa/adsp;/usr/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp;/etc/images/dsp;"
cd /data/local/tmp/snpebm_222/jacob_222
rm -rf output
/data/local/tmp/snpebm_222/artifacts/aarch64-oe-linux-gcc8.2/bin/snpe-net-run --container jacob_best_model_2.22.dlc --input_list jacob_test20.txt --output_dir output --use_gpu --gpu_mode float16 --perf_profile high_performance --profiling_level basic
