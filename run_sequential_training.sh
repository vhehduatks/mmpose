#!/bin/bash
source /home/hyeonghwan/anaconda3/etc/profile.d/conda.sh
conda activate mmpose
cd /home/hyeonghwan/github/mmpose

LOG_FILE="/home/hyeonghwan/github/mmpose/work_dirs/HMD_xregopose_hybrid_lifting_v1_full/20260127_015344/20260127_015344.log"

echo "[$(date)] Waiting for Hybrid Lifting training to complete..."

# Wait for Hybrid Lifting training to complete
while true; do
    if grep -q "Epoch(val) \[10\]" "$LOG_FILE" 2>/dev/null; then
        echo "[$(date)] Hybrid Lifting training completed!"
        break
    fi
    sleep 60
done

# ===== Train Decoupled =====
echo "[$(date)] Starting Decoupled training with 2 GPUs..."
bash tools/dist_train.sh my_code/custom_config/HMD_xregopose_decoupled_full_config.py 2
echo "[$(date)] Decoupled training completed!"

# ===== Train Hierarchical =====
echo "[$(date)] Starting Hierarchical training with 2 GPUs..."
bash tools/dist_train.sh my_code/custom_config/HMD_xregopose_hierarchical_full_config.py 2
echo "[$(date)] Hierarchical training completed!"

echo "[$(date)] All sequential trainings completed!"
