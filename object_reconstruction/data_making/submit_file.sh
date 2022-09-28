#!/bin/sh

cd $SLURM_SUBMIT_DIR
module add lib/hdf5/1.10.6-gcc

python extract_touch_charts.py
