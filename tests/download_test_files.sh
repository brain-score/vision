#!/bin/bash

# get directory of this script (i.e. tests), following https://stackoverflow.com/a/246128/2225200
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

aws --no-sign-request s3 cp s3://brain-score-tests/tests/test_benchmarks/alexnet-freemanziemba2013.aperture-private.pkl $(SCRIPT_DIR)/test_benchmarks/
aws --no-sign-request s3 cp s3://brain-score-tests/tests/test_benchmarks/alexnet-majaj2015.private-features.12.pkl $(SCRIPT_DIR)/test_benchmarks/
aws --no-sign-request s3 cp s3://brain-score-tests/tests/test_benchmarks/cornet_s-kar2019.pkl $(SCRIPT_DIR)/test_benchmarks/
