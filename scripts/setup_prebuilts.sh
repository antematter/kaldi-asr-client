#!/bin/sh

set -eu

BASEDIR="${0%/*}/.."
BASEDIR="$(realpath "$BASEDIR")"

cd "$BASEDIR/.build"

BUILD_DIR="$PWD/libclient"
PREBUILTS_DIR="$PWD/prebuilts"

LIBS="
$BUILD_DIR/grpc_connect_client
$BUILD_DIR/libkaldi-asr-parallel-client.so
client/lib/libcrypto.so.1.1
client/lib/libgrpcclient.so
client/lib/libssl.so.1.1
kaldi/src/lib/libkaldi-base.so
kaldi/src/lib/libkaldi-util.so
kaldi/src/lib/libkaldi-matrix.so
kaldi/src/lib/libkaldi-feat.so
kaldi/src/lib/libkaldi-lat.so
kaldi/src/lib/libkaldi-transform.so
kaldi/src/lib/libkaldi-hmm.so
kaldi/src/lib/libkaldi-gmm.so
kaldi/src/lib/libkaldi-tree.so
kaldi/tools/openfst-1.7.2/lib/libfst.so.16
"

rm -rf "$BUILD_DIR"
cmake -B "$BUILD_DIR" "$PWD/../src/c++"
cmake --build "$BUILD_DIR"

rm -rf "$PREBUILTS_DIR"
mkdir -p "$PREBUILTS_DIR"

cp -fL -- $LIBS "$PREBUILTS_DIR/"

patchelf --set-rpath '${ORIGIN}' "$PREBUILTS_DIR/"*
