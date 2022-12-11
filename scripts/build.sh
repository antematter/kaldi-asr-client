#!/bin/sh

set -eu

BASEDIR="${0%/*}/.."
BASEDIR="$(realpath "$BASEDIR")"

cd "$BASEDIR"

CLIENT_LIBS="https://github.com/triton-inference-server/server/releases/download/v2.28.0/v2.28.0_ubuntu2004.clients.tar.gz"
KALDI="https://github.com/kaldi-asr/kaldi"
OPENSSL="https://www.openssl.org/source/openssl-1.1.1s.tar.gz"

CLIENT_TAR="clients-2.28.0.tar.gz"
OPENSSL_TAR="openssl-1.1.1s.tar.gz"
KALDI_REPO="kaldi"

BUILD_DIR="$BASEDIR/.build"
CLIENT_BUILD="$BUILD_DIR/client"
OPENSSL_BUILD="$BUILD_DIR/openssl"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

[ -f "$OPENSSL_TAR" ] || curl -L "$OPENSSL" > "$OPENSSL_TAR"
[ -f "$CLIENT_TAR" ] || curl -L "$CLIENT_LIBS" > "$CLIENT_TAR"
[ -d "$KALDI_REPO" ] || git clone --depth=1 "$KALDI"

(
	rm -rf "$OPENSSL_BUILD"
	mkdir -p "$OPENSSL_BUILD"
	cd "$OPENSSL_BUILD"

	tar xf "$BUILD_DIR/$OPENSSL_TAR"
	cd *

	./config --prefix=/ --libdir=/lib
	make -j$(nproc)
)

(
	rm -rf "$CLIENT_BUILD"
	mkdir -p "$CLIENT_BUILD"

	cd "$CLIENT_BUILD"
	tar xf "$BUILD_DIR/$CLIENT_TAR"

	make -C "$OPENSSL_BUILD/"* DESTDIR="$PWD" install
	patchelf --set-rpath '${ORIGIN}' lib/libgrpcclient.so
)

(
        cd "$KALDI_REPO"

	make -j"$(nproc)" -C tools

	cd src
	[ -f kaldi.mk ] || ./configure --shared

	make depend -j$(nproc)
	make -j$(nproc)
)
