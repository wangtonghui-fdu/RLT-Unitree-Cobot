#!/bin/bash
# dry6f.sh TAG SEED0 N NPROC [extra args]: physics-only runs (no rendering, CPU) of gen_dual_v6f.py for seeds SEED0..SEED0+N-1 -> dry6f/TAG/
cd /tmp/r1sim; TAG=$1; S0=$2; N=$3; NP=$4; shift 4; mkdir -p dry6f/$TAG
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
seq $S0 $((S0 + N - 1)) | xargs -P $NP -I{} bash -c "timeout 900 ~/lingbot/.venvs/robotwin/bin/python col/gen_dual_v6f.py r1_column/column_dual/urdf/r1_column_dual_yup_s110.urdf r1_sim/camera_calib.json assets/robodojo col/layout_dual_v6.json dry6f/$TAG/s{} --seed {} --nominal 0 --render 0 $* > dry6f/$TAG/s{}.log 2>&1"
echo "DRY_DONE $TAG $(grep -l '"ok": true' dry6f/$TAG/s*/meta.json 2>/dev/null | wc -l) ok of $N"
