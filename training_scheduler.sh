#!/bin/bash
# =============================================================================
# Training Scheduler - File-based Queue
# =============================================================================
# 사용법:
#   시작: nohup ./training_scheduler.sh &
#   추가: echo "my_code/custom_config/some_config.py" >> training_queue.txt
#   중지: touch training_stop
#
# training_queue.txt 형식:
#   config 경로를 한 줄에 하나씩. 완료되면 자동으로 [DONE] 표시.
#   [DONE] 줄은 건너뜀. 새 줄 추가하면 자동으로 훈련 시작.
# =============================================================================

WORK_DIR="/home/hyeonghwan/github/mmpose"
QUEUE_FILE="$WORK_DIR/training_queue.txt"
STOP_FILE="$WORK_DIR/training_stop"
LOG_FILE="$WORK_DIR/training_scheduler.log"

NUM_GPUS=2
BASE_PORT=29500
POLL_INTERVAL=30  # seconds between queue checks

source /home/hyeonghwan/anaconda3/etc/profile.d/conda.sh
conda activate mmpose
cd "$WORK_DIR"

# Remove stale stop file
rm -f "$STOP_FILE"

# Create queue file if not exists
touch "$QUEUE_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

kill_port() {
    fuser -k ${1}/tcp 2>/dev/null
    sleep 2
}

get_next_config() {
    # Find first line that is NOT [DONE] and NOT empty/comment
    while IFS= read -r line; do
        # Skip empty, comments, done
        [[ -z "$line" ]] && continue
        [[ "$line" == \#* ]] && continue
        [[ "$line" == \[DONE\]* ]] && continue
        echo "$line"
        return 0
    done < "$QUEUE_FILE"
    return 1
}

mark_done() {
    local config="$1"
    # Replace the exact line with [DONE] prefix
    sed -i "s|^${config}$|[DONE] ${config}|" "$QUEUE_FILE"
}

mark_failed() {
    local config="$1"
    sed -i "s|^${config}$|[FAILED] ${config}|" "$QUEUE_FILE"
}

run_training() {
    local config="$1"
    local port="$2"
    local config_name=$(basename "$config" .py)

    log "=========================================="
    log "START: $config_name"
    log "Config: $config"
    log "Port: $port, GPUs: $NUM_GPUS"
    log "=========================================="

    kill_port "$port"

    MASTER_PORT=$port bash tools/dist_train.sh "$config" $NUM_GPUS 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log "DONE: $config_name (SUCCESS)"
        mark_done "$config"
    else
        log "DONE: $config_name (FAILED, exit=$exit_code)"
        mark_failed "$config"
    fi

    return $exit_code
}

# =============================================================================
# Main Loop
# =============================================================================
log "=========================================="
log "Training Scheduler Started (Queue-based)"
log "Queue file: $QUEUE_FILE"
log "Stop file:  $STOP_FILE (touch to stop)"
log "=========================================="

PORT=$BASE_PORT

while true; do
    # Check stop signal
    if [ -f "$STOP_FILE" ]; then
        log "Stop signal received. Exiting."
        rm -f "$STOP_FILE"
        break
    fi

    # Get next config
    config=$(get_next_config)

    if [ -n "$config" ]; then
        # Check file exists
        if [ ! -f "$config" ]; then
            log "ERROR: Config not found: $config"
            mark_failed "$config"
            continue
        fi

        run_training "$config" $PORT

        # Increment port, wait before next
        ((PORT++))
        if [ $PORT -gt 29510 ]; then
            PORT=$BASE_PORT
        fi
        sleep 10
    else
        # No pending config, wait and poll again
        sleep $POLL_INTERVAL
    fi
done

log "=========================================="
log "Training Scheduler Stopped"
log "=========================================="
