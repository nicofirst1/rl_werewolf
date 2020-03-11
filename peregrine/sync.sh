#!/bin/bash

rsync -v -a --exclude-from='peregrine/ignore_list.txt' . s4171632@peregrine.hpc.rug.nl:/home/s4171632/rl-werewolf