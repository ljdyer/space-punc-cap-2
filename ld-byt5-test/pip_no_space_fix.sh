#!/bin/sh

NEW_TMP_DIR="$PWD/tmp"
echo "Setting tmp dir to: $NEW_TMP_DIR"
mkdir $NEW_TMP_DIR
export TMPDIR=$NEW_TMP_DIR
echo "Done."