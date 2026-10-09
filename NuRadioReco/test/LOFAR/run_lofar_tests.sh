#!/bin/bash

set -e

python3 test_lofar_data_pipeline.py
python3 test_lofar_simulation_pipeline.py

rm -rf test-data/