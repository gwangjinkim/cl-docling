#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .build/native
if [ "$(uname -s)" = Darwin ]; then
    # Override for another libpng installation. No installation/download occurs.
    png_prefix=${DOCLING_PNG_PREFIX:-/opt/homebrew/opt/libpng}
    "${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror -ffp-contract=off -dynamiclib \
        -I"$png_prefix/include" -L"$png_prefix/lib" native/images.c -lpng -lm \
        -o .build/native/libdocling_images.dylib
else
    "${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror -ffp-contract=off -fPIC -shared \
        native/images.c $(pkg-config --cflags --libs libpng) -lm \
        -o .build/native/libdocling_images.so
fi
