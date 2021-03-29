import pickle
from pathlib import Path

import boto3
from brainio_base.assemblies import BehavioralAssembly
from brainio_collection.packaging import write_netcdf


local_pkl_names = [
    'alexnet-probabilities.pkl',
    'resnet34-probabilities.pkl',
    'resnet18-probabilities.pkl'
]


s3_pkl_names = [
    "alexnet-freemanziemba2013.aperture-private.pkl",
    "alexnet-majaj2015.private-features.12.pkl",
    "CORnetZ-rajalingham2018public.pkl",
    "cornet_s-kar2019.pkl",
    "alexnet-sanghavi2020-features.12.pkl",
    "alexnet-sanghavijozwik2020-features.12.pkl",
    "alexnet-sanghavimurty2020-features.12.pkl",
    "alexnet-rajalingham2020-features.12.pkl",
]


def local_pkls():
    target_dir_path = Path(__file__).parents[1] / "tests" / "test_metrics"
    for pkl_name in local_pkl_names:
        pkl_path = target_dir_path / pkl_name
        nc_path = pkl_path.with_suffix(".nc")
        if not nc_path.exists():
            print(f"{nc_path} does not exist.  ")
            with open(pkl_path, 'rb') as f:
                unpickled = pickle.load(f)
                #   write netcdf
                sha1 = write_netcdf(BehavioralAssembly(unpickled["data"]), str(nc_path))
        else:
            print(f"{nc_path} already exists.  ")


def s3_pkls():
    session = boto3.session.Session(profile_name="dicarlolab_jjpr")
    s3 = session.client("s3")
    bucket_name = "brain-score-tests"
    def exists(key):
        try:
            s3.head_object(Bucket=bucket_name, Key=key)
            return True
        except s3.exceptions.NoSuchKey:
            return False

    prefix_path = Path("tests", "test_benchmarks")
    target_dir_path = Path(__file__).parent / "test_pkl"

    for pkl_name in s3_pkl_names:
        pkl_path = Path(pkl_name)
        nc_path = pkl_path.with_suffix(".nc")
        object_key_pkl = prefix_path / pkl_path
        target_file_pkl = target_dir_path / pkl_path
        target_file_nc = target_dir_path / nc_path
        object_key_nc = prefix_path / nc_path

        if not exists(str(object_key_nc)):
            print(f"{object_key_nc} does not exist.  ")
            if not target_file_nc.exists():
                if not target_file_pkl.exists():
                    #   fetch file
                    s3.download_file(bucket_name, str(object_key_pkl), str(target_file_pkl))
                #   unpickle
                with open(target_file_pkl, 'rb') as f:
                    unpickled = pickle.load(f)
                    #   write netcdf
                    sha1 = write_netcdf(unpickled["data"], str(target_file_nc))
            #   upload
            s3.upload_file(str(target_file_nc), bucket_name, str(object_key_nc))
        else:
            print(f"{object_key_nc} already exists.  ")


def main():
    # assert xarray is 0.12.3
    local_pkls()
    s3_pkls()


if __name__ == '__main__':
    main()

