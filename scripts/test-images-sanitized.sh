#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .build/native
if [ "$(uname -s)" = Darwin ]; then
    png_prefix=${DOCLING_PNG_PREFIX:-/opt/homebrew/opt/libpng}
    set -- -I"$png_prefix/include" -L"$png_prefix/lib" -lpng
else
    set -- $(pkg-config --cflags --libs libpng)
fi
"${CC:-cc}" -std=c11 -O1 -g -Wall -Wextra -Werror -ffp-contract=off \
    -fsanitize=address,undefined -fno-omit-frame-pointer tests/native-images.c \
    "$@" -lm -o .build/native/test-images-sanitized
.build/native/test-images-sanitized tests/fixtures/processor
