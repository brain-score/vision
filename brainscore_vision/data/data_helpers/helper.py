import pandas as pd


version_id_path = '../data_helpers/version_ids.csv'
version_id_df = pd.read_csv(version_id_path, index_col='Unnamed: 0')


def build_filename(identifier: str, file_type: str):
    if file_type == '.nc':
        filename = f'assy_{identifier.replace(".", "_")}.nc'
    elif file_type == '.csv':
        filename = f'image_{identifier.replace(".", "_")}.csv'
    else:
        filename = f'image_{identifier.replace(".", "_")}.zip'
    return filename
