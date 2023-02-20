import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_weights_if_needed():
    container_dir = Path(__file__).parent
    state_dict_dir = container_dir / 'state_dicts'
    if state_dict_dir.is_dir():
        return  # no download needed, already exists
    state_dicts_zip = container_dir / "state_dicts.zip"

    url = (
        "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
    )

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2 ** 20  # Mebibyte
    t = tqdm(total=total_size, unit="MiB", unit_scale=True)

    with open(state_dicts_zip, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception("Error, something went wrong")

    print("Download successful. Unzipping file...")
    with zipfile.ZipFile(state_dicts_zip, "r") as zip_ref:
        zip_ref.extractall(container_dir)
        print("Unzip file successful!")
