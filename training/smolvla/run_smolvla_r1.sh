#!/bin/bash
# SmolVLA full fine-tune on the R1 plug demos (11-19), CIC GPUs selected by nvidia-smi index -> UUID (CUDA order differs).
# usage: cd ~/lingbot/smolvla_c2r && GPU_IDX="4 5" bash tools/run_smolvla_r1.sh configs/smolvla_r1_plug_v2_2gpu.json [--smoke-test | --resume <ckpt>]
set -euo pipefail
cd /path/to/home/lingbot/smolvla_c2r
P=/path/to/home/venvs/smolvla_c2r/bin/python
U=""; for i in ${GPU_IDX:-4 5}; do U="$U${U:+,}$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader)"; done
export CUDA_VISIBLE_DEVICES=$U
N=$(echo $CUDA_VISIBLE_DEVICES | tr , '\n' | wc -l)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cfg=$1; shift
case " $* " in *" --smoke-test "*|*" --resume "*) ;; *) $P scripts/train_smolvla_r1.py --config $cfg --prepare-only ;; esac
exec $P -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$N scripts/train_smolvla_r1.py --config $cfg "$@"
