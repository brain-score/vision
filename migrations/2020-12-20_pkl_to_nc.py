import pickle
from pathlib import Path

import boto3
from brainio_collection.packaging import write_netcdf

session = boto3.session.Session(profile_name="dicarlolab_jjpr")
s3 = session.client("s3")


pkl_names = [
    "alexnet-freemanziemba2013.aperture-private.pkl",
    "alexnet-majaj2015.private-features.12.pkl",
    "CORnetZ-rajalingham2018public.pkl",
    "cornet_s-kar2019.pkl",
    "alexnet-sanghavi2020-features.12.pkl",
    "alexnet-sanghavijozwik2020-features.12.pkl",
    "alexnet-sanghavimurty2020-features.12.pkl",
    "alexnet-rajalingham2020-features.12.pkl",
]


def main():
    # assert xarray is 0.12.3
    bucket_name = "brain-score-tests"
    prefix_path = Path("tests", "test_benchmarks")
    target_dir_path = Path(__file__).parent / "test_pkl"
    # for each file
    for pkl_name in pkl_names:
        pkl_path = Path(pkl_name)
        nc_path = pkl_path.with_suffix(".nc")
        object_key_pkl = prefix_path / pkl_path
        target_file_pkl = target_dir_path / pkl_path
        target_file_nc = target_dir_path / nc_path
        object_key_nc = prefix_path / nc_path
        #   fetch file
        s3.download_file(bucket_name, str(object_key_pkl), str(target_file_pkl))
        #   unpickle
        with open(target_file_pkl, 'rb') as f:
            unpickled = pickle.load(f)
        #   write netcdf
        write_netcdf(unpickled["data"], str(target_file_nc))
        #   upload
        s3.upload_file(str(target_file_nc), bucket_name, str(object_key_nc))


if __name__ == '__main__':
    main()

