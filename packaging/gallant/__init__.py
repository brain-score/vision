import glob
import os

import argparse
import numpy as np
import pandas as pd
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default=os.path.join('V1Data', 'NatRev'))
    args = parser.parse_args()
    print("Running with args {}".format(vars(args)))

    data_files = glob.glob(os.path.join(args.directory, 'data', '*.csv'))
    data = pd.concat((pd.read_csv(f) for f in data_files))
    num_duplicates = len(data) - len(data.drop_duplicates())
    # assert num_duplicates == 0
    data.rename(columns={'cellName': 'neuroid', 'stimuliPaths': 'image_file_name'}, inplace=True)
    data['image_file_name'] = [file.replace('\\', '/') for file in data['image_file_name']]
    data['image_id'] = [os.path.basename(os.path.splitext(file)[0])
                        for file in data['image_file_name']]
    neuroids, neuroid_indices = unique_ordered(data['neuroid'].values, return_index=True)
    print("Found responses for {} cells, average spike count {:.4f}, {} duplicates".format(
        len(neuroids), np.mean(data['response']), num_duplicates))

    responses = np.full((len(data), len(neuroids)), np.nan)
    row = 0
    for neuroid_iter, neuroid in enumerate(neuroids):
        neuroid_data = data[data['neuroid'] == neuroid]
        responses[row:row + len(neuroid_data), neuroid_iter] = neuroid_data['response'].values
        row += len(neuroid_data)

    assembly = xr.DataArray(responses,
                            coords={
                                'image_id': data['image_id'],
                                'image_file_name': ('image_id', data['image_file_name']),
                                'category_name': ('image_id', data['stimulusCategory']),
                                'stimulusRepeats': ('image_id', data['stimulusRepeats']),

                                'neuroid': neuroids,
                                'region': ('neuroid', data['area'].iloc[neuroid_indices]),
                                'animal': ('neuroid', data['animal'].iloc[neuroid_indices])},
                            dims=['image_id', 'neuroid'])
    print("Created {} assembly".format(" x ".join(map(str, assembly.shape))))
    savepath = os.path.abspath(os.path.join(args.directory, 'data.nc'))
    assembly.to_netcdf(savepath)
    print("Saved to {}".format(savepath))


def unique_ordered(x, return_index=False):
    _, indices = np.unique(x, return_index=True)
    ordered_x = x[np.sort(indices)]
    if not return_index:
        return ordered_x
    return ordered_x, indices


if __name__ == '__main__':
    main()
