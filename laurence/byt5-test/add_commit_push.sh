#!/bin/sh

if [[ -z "${1+present}" ]] then
    echo "need commit message"
    exit
fi
git add --all
git commit -m $1
git push