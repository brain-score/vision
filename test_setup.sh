#!/bin/bash

# get directory of this script (i.e. tests), following https://stackoverflow.com/a/246128/2225200
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for f in \
    alexnet-freemanziemba2013.aperture-private.nc \
    alexnet-majaj2015.private-features.12.nc \
    CORnetZ-rajalingham2018public.nc \
    cornet_s-kar2019.nc \
    alexnet-sanghavi2020-features.12.nc \
    alexnet-sanghavijozwik2020-features.12.nc \
    alexnet-sanghavimurty2020-features.12.nc \
    alexnet-rajalingham2020-features.12.nc \
    resnet-50-pytorch-3deg-Geirhos2021_colour.nc \
    resnet-50-pytorch-3deg-Geirhos2021_contrast.nc \
    resnet-50-pytorch-3deg-Geirhos2021_cueconflict.nc \
    resnet-50-pytorch-3deg-Geirhos2021_edge.nc \
    resnet-50-pytorch-3deg-Geirhos2021_eidolonI.nc \
    resnet-50-pytorch-3deg-Geirhos2021_eidolonII.nc \
    resnet-50-pytorch-3deg-Geirhos2021_eidolonIII.nc \
    resnet-50-pytorch-3deg-Geirhos2021_falsecolour.nc \
    resnet-50-pytorch-3deg-Geirhos2021_highpass.nc \
    resnet-50-pytorch-3deg-Geirhos2021_lowpass.nc \
    resnet-50-pytorch-3deg-Geirhos2021_phasescrambling.nc \
    resnet-50-pytorch-3deg-Geirhos2021_powerequalisation.nc \
    resnet-50-pytorch-3deg-Geirhos2021_rotation.nc \
    resnet-50-pytorch-3deg-Geirhos2021_silhouette.nc \
    resnet-50-pytorch-3deg-Geirhos2021_sketch.nc \
    resnet-50-pytorch-3deg-Geirhos2021_stylized.nc \
    resnet-50-pytorch-3deg-Geirhos2021_uniformnoise.nc
do
  aws --no-sign-request s3 cp s3://brain-score-tests/tests/test_benchmarks/${f} ${SCRIPT_DIR}/tests/test_benchmarks/
done

pip install git+https://github.com/brain-score/model-tools --no-deps
pip install torch torchvision
