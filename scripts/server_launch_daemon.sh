#!/bin/sh

set -eu

FIFO="/tmp/.launcher_fifo"
LOG_FILE="/tmp/.launcher_fifo_log"
LOGDIR="/tmp/.launcher_fifo_logs"
PORT=${LAUNCHER_PORT:-5555}
CONTAINER_FMT="trt_server_asr_"
START_TIMEOUT=5
STOP_TIMEOUT=5

GRPC_CLIENT_BIN="$(realpath "${0%/*}/../.build/prebuilts/grpc_connect_client")"

SUCCESS="0"
FAILURE="1"

LOG_INFO="INFO"
LOG_ERROR="ERROR"

REPO="$HOME/DeepLearningExamples/Kaldi/SpeechRecognition"
LAUNCH_SCRIPT="$REPO/scripts/docker/launch_server.sh"

log() {
	echo "$1: $2"
}

get_container() {
	echo "${CONTAINER_FMT}$1"
}

kill_server() {
	docker container stop "$1" -t "$STOP_TIMEOUT" || :
	docker container rm -f "$1" 2>/dev/null || :
}

launch_server() (
	cd "$REPO"
	GPU="$1" GRPC_PORT="$2" exec "$LAUNCH_SCRIPT" >"$LOGDIR/$1.log" 2>&1
)

handle_input() {
	gpu=0
	pids=

	for port; do
		container="$(get_container "$gpu")"

		log "$LOG_INFO" "Stopping container '$container'..."
		kill_server "$container" &

		pids="$pids,$!"
		: "$((gpu += 1))"
	done

	for pid in $pids; do [ "$pid" ] && wait "$pid"; done

	gpu=0

	for port; do
		container="$(get_container "$gpu")"

		log "$LOG_INFO" "Launching container '$container' on port '$port' with GPU '$gpu'..."
		launch_server "$gpu" "$port" &

		: "$((gpu += 1))"
	done

	sleep "$START_TIMEOUT"

	time=0

	while [ "$time" -lt "$START_TIMEOUT" ]; do
		fail=

		for port; do
			$GRPC_CLIENT_BIN "localhost:$port" || {
				fail=true
				log "$LOG_ERROR" "Failed to connect to server at port '$port'"
			}
		done

		[ "$fail" ] || {
			log "$LOG_INFO" "Launched servers"
			return 0
		}

		sleep 1
		: "$((time += 1))"
	done

	log "$LOG_ERROR" "Timed out while waiting for servers to start"

	return 1
}

rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

rm -f "$FIFO"
mkfifo "$FIFO"

echo "Starting server to accept connections at localhost:$PORT. Logs for this script can be found at '$LOG_FILE'. Docker container logs are present at '$LOGDIR/<GPU>.log'"

{
	{
		while read -r id_ports; do
			IFS=" "
			set -- $id_ports

			# Extract ID
			id="$1"
			shift

			# Extract ports
			IFS=","
			set -- $@

			if handle_input "$@" 1>&2; then
				echo "$id $SUCCESS"
			else
				echo "$id $FAILURE"
			fi
		done
	} <"$FIFO" | nc -lk -p "$PORT" >"$FIFO"
} 2>&1 | tee "$LOG_FILE"
