MPS_PIPE_DIR="/tmp/nvidia-mps"
MPS_LOG_DIR="/tmp/nvidia-mps-log"
 
case "${1:-}" in
    start)
        mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
        export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
        export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"
        nvidia-cuda-mps-control -d 2>/dev/null
        sleep 2
        echo "[MPS] Daemon started."
        ;;
    stop)
        echo quit | nvidia-cuda-mps-control 2>/dev/null
        sleep 1
        echo "[MPS] Daemon stopped."
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac