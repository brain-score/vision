import os
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

import brainio_collection
from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_collection.packaging import package_data_assembly


animals = ['Chabo_IT_A', 'Chabo_IT_M', 'Tito_IT_A','Tito_IT_M','TitoR_IT_A','TitoR_IT_M','Nano_IT_A',
           'Nano_IT_M', 'Nano_IT_P','Magneto_IT_A', 'Magneto_IT_M', 'Magneto_IT_P', 'Chabo_V4',
           'Tito_V4','TitoR_V4','B6_IT', 'B8_IT', 'I2_IT', 'B6_V4', 'B8_V4','I2_V4', 'Magneto_V4',
           'B6_IT_170_270', 'B8_IT_170_270', 'I2_IT_170_270', 'I2_IT_latency_adjusted',
           'NanoR_V4', 'NanoR_IT_P', 'NanoR_IT_M', 'HU_IT', 'HT_IT', 'HU_IT_latency_adjusted', 'HT_IT_latency_adjusted']


juve = [17, 20, 29, 30]


var = ["0", "3", "6"]


def get_image_ids(variation, data, df_hvm):
    ids = df_hvm[df_hvm["var"]==f"V{variation}"]["id"]
    assert data.shape[0] == len(ids)
    return ids


def coords_from_darren(animal_region, variation, data, df_hvm):
    coords = {}
    coords["image_id"] = ("stimulus", get_image_ids(variation, data, df_hvm))
    coords["repetition"] = ("repetition", range(data.shape[1]))
    coords["animal"] = ("neuroid", [animal_region.split("_")[0]]*data.shape[2])
    coords["region"] = ("neuroid", [animal_region.split("_")[1]]*data.shape[2])
    coords["electrode"] = ("neuroid", range(data.shape[2]))
    coords["neuroid_id"] = ("neuroid", [f"{animal_region}_{i:02}" for i in range(data.shape[2])])
    return coords


def xr_from_darren(animal_region, variation, data, df_hvm):
    xr_data = xr.DataArray(
        data=data,
        coords=coords_from_darren(animal_region, variation, data, df_hvm),
        dims=["stimulus", "repetition", "neuroid"]
    )
    return xr_data


def load_responses(metric_bins_path, csv_path):
    assert metric_bins_path.exists()
    metric_bins_np = np.load(metric_bins_path, allow_pickle=True, encoding='latin1')
    df_hvm = pd.read_csv(csv_path)
    relevant_arrays = [{
        "animals_index": i,
        "animal_region": animals[i],
        "variation": k,
        "data": metric_bins_np[()]["data"]["bins"][k][i]
    } for i in juve for k in var]
    for ra in relevant_arrays:
        ra["xr_data"] = xr_from_darren(
            animal_region=ra["animal_region"],
            variation=ra["variation"],
            data=ra["data"],
            df_hvm=df_hvm
        )
    concatted_per_var = {}
    for v in var:
        aligned = xr.align(*[x["xr_data"] for x in relevant_arrays if x["variation"] == v],
                           join="outer",
                           exclude=["neuroid"],
                           fill_value=np.nan
                           )
        concatted_per_var[v] = xr.concat(aligned, dim="neuroid")
    stacked_per_var = {}
    for v in concatted_per_var:
        stacked_per_var[v] = concatted_per_var[v].stack(presentation=["repetition", "stimulus"])
    final = xr.concat(stacked_per_var.values(), dim="presentation")#.transpose()
    final = final.expand_dims('time_bin', 2)
    final['time_bin_start'] = ('time_bin', [70])
    final['time_bin_end'] = ('time_bin', [170])
    print(final.dims)
    assert final.dims == ("neuroid", "presentation", "time_bin")
    return final


def main():
    metric_bins_path = Path("/braintree/home/darren/work/metric_bins.npy")
    csv_path = Path(__file__).parents[2] / "notebooks" / "2020-11-22_hvm_from_dldata.csv"

    stimuli = brainio_collection.get_stimulus_set('dicarlo.hvm')
    assembly = load_responses(metric_bins_path, csv_path)
    assembly.name = 'dicarlo.Seibert2019'

    print('Packaging assembly')
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=stimuli.identifier,
                          bucket_name='brainio-brainscore')


if __name__ == '__main__':
    main()
