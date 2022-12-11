#!/bin/sh

set -eu

BASEDIR="${0%/*}/.."
BASEDIR="$(realpath "$BASEDIR")"

cd "/data/data/LibriSpeech/test-clean"

DIRS="$(
	for dir in */*; do
		echo "$dir"
	done
)"

DIRS="$(echo "$DIRS" | head -n 5)"

cd "$BASEDIR/src/python/kaldi_asr_client"

for file in e2e_tests/*.py; do
	file="${file#e2e_tests/}"
	file="${file%.py}"

	python -m "e2e_tests.${file}" $DIRS || {
		echo "Test '$file' failed!" >&2
		return 1
	}
done
