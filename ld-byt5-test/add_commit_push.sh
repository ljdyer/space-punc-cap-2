#!/bin/sh

if [[ -z "${1+present}" ]] then
    echo "need commit message"
else 
    git add --all
    git commit -m $1
    git push
fi