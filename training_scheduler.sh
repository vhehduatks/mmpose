#!/bin/bash
# =============================================================================
# Training Scheduler - Sequential Multi-GPU Training
# =============================================================================
# Usage: ./training_scheduler.sh
#
# Features:
# - Sequential training with automatic port management
# - Wait between trainings to release ports
# - Logging with timestamps
# =============================================================================

source /home/hyeonghwan/anaconda3/etc/profile.d/conda.sh
conda activate mmpose
cd /home/hyeonghwan/github/mmpose

LOG_FILE="/home/hyeonghwan/github/mmpose/training_scheduler.log"
BASE_PORT=29500

# =============================================================================
# Training Queue - Add configs here
# =============================================================================
CONFIGS=(
    "my_code/custom_config/HMD_xregopose_decoupled_full_config.py"
    "my_code/custom_config/HMD_xregopose_hierarchical_full_config.py"
    "my_code/custom_config/HMD_xregopose_vit_lifting_v6_full_config.py"
)

NUM_GPUS=2
WAIT_BETWEEN_TRAININGS=30  # seconds to wait between trainings

# =============================================================================
# Functions
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

kill_port() {
    local port=$1
    fuser -k ${port}/tcp 2>/dev/null
    sleep 2
}

run_training() {
    local config=$1
    local port=$2
    local config_name=$(basename "$config" .py)

    log "=========================================="
    log "Starting: $config_name"
    log "Port: $port, GPUs: $NUM_GPUS"
    log "=========================================="

    # Kill any process using this port
    kill_port $port

    # Run training
    MASTER_PORT=$port bash tools/dist_train.sh "$config" $NUM_GPUS 2>&1 | tee -a "$LOG_FILE"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log "Completed: $config_name (SUCCESS)"
    else
        log "Completed: $config_name (FAILED with exit code $exit_code)"
    fi

    return $exit_code
}

# =============================================================================
# Main
# =============================================================================
log "=========================================="
log "Training Scheduler Started"
log "Total configs: ${#CONFIGS[@]}"
log "=========================================="

CURRENT_PORT=$BASE_PORT
SUCCESSFUL=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    # Check if config file exists
    if [ ! -f "$config" ]; then
        log "ERROR: Config not found: $config"
        ((FAILED++))
        continue
    fi

    # Run training
    run_training "$config" $CURRENT_PORT

    if [ $? -eq 0 ]; then
        ((SUCCESSFUL++))
    else
        ((FAILED++))
    fi

    # Increment port for next training
    ((CURRENT_PORT++))

    # Wait before next training
    log "Waiting ${WAIT_BETWEEN_TRAININGS}s before next training..."
    sleep $WAIT_BETWEEN_TRAININGS
done

log "=========================================="
log "Training Scheduler Finished"
log "Successful: $SUCCESSFUL, Failed: $FAILED"
log "=========================================="
