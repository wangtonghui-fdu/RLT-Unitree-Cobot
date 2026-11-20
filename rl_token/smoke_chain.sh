#!/bin/bash
# lingbot-cobot-ppu smoke chain (11-10): a = RLT on 20 steps + save @20; b = resume from @20 to 22 (RLT state reload);
# off = RLT off 10 steps (non-interference: per-step VLA losses must equal run a). Stops at the first failure.
Q=/path/to/storage/lingbot-cobot-ppu; L=$Q/logs; R=/root/cobot_ppu_runs
note(){ echo "[chain] $(date '+%F %T') $*" >> $L/smoke_chain.log; }
note "chain start pid $$"
bash $Q/tools/run_smoke.sh smoke_rlt 1 20 20 > $L/smoke_rlt_a.log 2>&1
if grep -q "Step 20/" $L/smoke_rlt_a.log && [ -f $R/smoke_rlt/checkpoints/global_step_20/rlt/rlt_state.pt ]; then note "run a ok"; else note "run a FAILED"; exit 1; fi
bash $Q/tools/run_smoke.sh smoke_rlt 1 22 0 > $L/smoke_rlt_b.log 2>&1
if grep -q "Step 22/" $L/smoke_rlt_b.log; then note "run b ok"; else note "run b FAILED"; exit 1; fi
bash $Q/tools/run_smoke.sh smoke_off 0 10 0 > $L/smoke_off.log 2>&1
if grep -q "Step 10/" $L/smoke_off.log; then note "run off ok"; else note "run off FAILED"; exit 1; fi
note "chain done"
