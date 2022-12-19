#!/bin/sh

set -eu

FIFO="/tmp/.launcher_fifo"
LOGDIR="/tmp/.launcher_fifo_logs"
PORT=5555
PORT_BASE=8001
CONTAINER_FMT="trt_server_asr_"
START_TIMEOUT=5
STOP_TIMEOUT=5

SUCCESS="0"
FAILURE="1"

LOG_INFO="INFO"
LOG_WARN="WARN"

REPO="$HOME/DeepLearningExamples/Kaldi/SpeechRecognition"
LAUNCH_SCRIPT="$REPO/scripts/docker/launch_server.sh"

log() {
    echo "$1: $2"
}

get_container() {
    echo "${CONTAINER_FMT}$1"
}

get_port() {
    echo "$((PORT_BASE+$1))"
}

kill_server() {
    docker container stop "$1" -t "$STOP_TIMEOUT"
}

launch_server() (
    cd "$REPO"
    GPU="$1" GRPC_PORT="$2" exec "$LAUNCH_SCRIPT" > "$LOGDIR/$1.log" 2>&1
)

handle_input() {
    num_servers="$1"
    : $((num_servers-=1))

    [ "$num_servers" -lt 0 ] && {
	echo "num_servers < 0!" >&2
        return 1
    }

    set --

    for server in $(seq 0 $num_servers); do
	container="$(get_container "$server")"

	log "$LOG_INFO" "Stopping container '$container'..."
        kill_server "$container" &

	set -- "$* $!"
    done

    i=0

    for pid; do
	    wait "$pid" || log "$LOG_WARN" "Failed to stop container '$(get_container $i)'!" >&2
            docker container rm -f "$(get_container $i)" 2>/dev/null ||:

            : $((i+=1))
    done

    unset i

    for server in $(seq 0 $num_servers); do
	container="$(get_container "$server")"
	port="$(get_port "$server")"

	log "$LOG_INFO" "Launching container '$container' on port '$port' with GPU '$server'..."
	launch_server "$server" "$port" &
    done

    sleep "$START_TIMEOUT"

    return 0
}

rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

rm -f "$FIFO"
mkfifo "$FIFO"

{
    while read -r num_servers; do
        if handle_input "$num_servers" 1>&2; then
	    echo "$SUCCESS"
	else
	    echo "$FAILURE"
	fi
    done
} < "$FIFO" | nc -lk -p "$PORT" > "$FIFO"
