#!/bin/bash

# check for the existence of at least one arg
if [ $# -eq 0 ]; then
  echo "You need to supply an executable path"
  echo "Training paths can be found in src/trainable"
  return
fi

export PYTHONPATH=src:gym_ww:$PYTHONPATH

python $1 "${@:2}"
