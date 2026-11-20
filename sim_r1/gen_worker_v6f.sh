#!/bin/bash
# gen_worker_v6f.sh GPU_INDEX WORKER_ID NWORKERS SEEDFILE OUT (env RARGS = extra generator arguments, e.g. the randomisation ranges): renders lines WORKER_ID+1, +NWORKERS, ... of SEEDFILE (seeds that passed the
# physics-only run dry6f/a) with gen_dual_v6f.py --data 1 (8 spp, 3 cameras) into OUT/sXXXX, skipping seeds that already have a meta.json.
GI=$1; W=$2; NW=$3; SF=$4; OUT=$5
cd /tmp/r1sim; mkdir -p $OUT
U=$(nvidia-smi -i $GI --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES=$U VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
FF=/path/to/home/miniconda3/envs/fastwam_rt/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
for s in $(awk -v w=$W -v n=$NW '(NR - 1) % n == w' $SF); do
  D=$OUT/$(printf s%04d $s); [ -f $D/meta.json ] && continue
  timeout 2400 ~/lingbot/.venvs/robotwin/bin/python col/gen_dual_v6f.py r1_column/column_dual/urdf/r1_column_dual_yup_s110.urdf r1_sim/camera_calib.json assets/robodojo col/layout_dual_v6.json $D --seed $s --nominal 0 --data 1 --spp 8 --ffmpeg $FF $RARGS > $D.log 2>&1
  rc=$?; [ -f $D/meta.json ] || echo "{\"seed\": $s, \"ok\": false, \"rc\": $rc}" > $D.failed
  echo "$(date +%T) seed $s rc $rc $(grep RESULT $D.log | cut -c1-260)"
done
echo "$(date +%T) WORKER_DONE $W"
