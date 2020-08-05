#!/bin/bash

# get directory of this script (i.e. tests), following https://stackoverflow.com/a/246128/2225200
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for f in alexnet-freemanziemba2013.aperture-private.pkl alexnet-majaj2015.private-features.12.pkl CORnetZ-rajalingham2018public.pkl cornet_s-kar2019.pkl
do
  aws --no-sign-request s3 cp s3://brain-score-tests/tests/test_benchmarks/${f} ${SCRIPT_DIR}/tests/test_benchmarks/
done

pip install git+https://github.com/brain-score/model-tools --no-deps
pip install torch torchvision
