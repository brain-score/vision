
import hashlib
import requests
from pathlib import Path
import zipfile
from collections import OrderedDict
from warnings import warn
import json
import torch
import zipfile
_MODELS = {}
_ALIASES = {}


def get_file(fname, origin, file_hash=None, cache_dir=".cache", cache_subdir="datasets", extract=True):
    """
    Download a file from a URL, cache it locally, and optionally verify its hash and extract it.

    Args:
        fname (str): The name of the file to save locally.
        origin (str): The URL to download the file from.
        file_hash (str, optional): The expected hash of the file to verify integrity. Defaults to None.
        cache_dir (str): The root cache directory. Defaults to ".cache".
        cache_subdir (str): The subdirectory within the cache directory. Defaults to "datasets".
        extract (bool): Whether to extract the file if it's a ZIP archive. Defaults to False.

    Returns:
        str: The path to the cached (and optionally extracted) file.
    """
    cache_path = Path(cache_dir) / cache_subdir
    cache_path.mkdir(parents=True, exist_ok=True)

    file_path = cache_path / fname

    if not file_path.exists():
        print(f"Downloading {origin} to {file_path}...")
        response = requests.get(origin, stream=True)
        response.raise_for_status()  
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete: {file_path}")

    if file_hash:
        print("Verifying file hash...")
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        downloaded_file_hash = sha256.hexdigest()
        if downloaded_file_hash != file_hash:
            raise ValueError(f"File hash does not match! Expected {file_hash}, got {downloaded_file_hash}")
        print("File hash verified.")

    if extract and zipfile.is_zipfile(file_path):
        extract_path = cache_path 
        json_file = extract_path / f"{fname.replace('.zip', '')}.json"
        weight_file = extract_path / f"{fname.replace('.zip', '')}.pth"
        if not json_file.exists() and not weight_file.exists():
            print(f"Extracting {file_path} to {extract_path}")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)  
            print(f"Extraction complete: {extract_path}")
        
        return str(extract_path)

    return str(file_path)


def clear_models_and_aliases(*cls):
    if len(cls) == 0:
        _MODELS.clear()
        _ALIASES.clear()
    else:
        for c in cls:
            if c in _MODELS:
                del _MODELS[c]
            if c in _ALIASES:
                del _ALIASES[c]
                
def register_model(cls, key, url, hash):
    # key must be a valid file/folder name in the file system
    models = _MODELS.setdefault(cls, OrderedDict())
    key not in models or warn(
        "re-registering model '{}' (was already registered for '{}')".format(
            key, cls.__name__
        )
    )
    models[key] = dict(url=url, hash=hash)


def register_aliases(cls, key, *names):
    # aliases can be arbitrary strings
    if len(names) == 0:
        return
    models = _MODELS.get(cls, {})
    key in models or ValueError(f"model '{key}' is not registered for '{cls.__name__}'")
    
    aliases = _ALIASES.setdefault(cls, OrderedDict())
    for name in names:
        aliases.get(name, key) == key or warn(
            "alias '{}' was previously registered with model '{}' for '{}'".format(
                name, aliases[name], cls.__name__
            )
        )
        aliases[name] = key


def get_registered_models(cls, return_aliases=True, verbose=False):
    models = _MODELS.get(cls, {})
    aliases = _ALIASES.get(cls, {})
    model_keys = tuple(models.keys())
    model_aliases = {
        key: tuple(name for name in aliases if aliases[name] == key) for key in models
    }
    if verbose:
        # this code is very messy and should be refactored...
        _n = len(models)
        _str_model = "model" if _n == 1 else "models"
        _str_is_are = "is" if _n == 1 else "are"
        _str_colon = ":" if _n > 0 else ""
        print(
            "There {is_are} {n} registered {model_s} for '{clazz}'{c}".format(
                n=_n,
                clazz=cls.__name__,
                is_are=_str_is_are,
                model_s=_str_model,
                c=_str_colon,
            )
        )
        if _n > 0:
            print()
            _maxkeylen = 2 + max(len(key) for key in models)
            print("Name{s}Alias(es)".format(s=" " * (_maxkeylen - 4 + 3)))
            print("────{s}─────────".format(s=" " * (_maxkeylen - 4 + 3)))
            for key in models:
                _aliases = "   "
                _m = len(model_aliases[key])
                if _m > 0:
                    _aliases += "'%s'" % "', '".join(model_aliases[key])
                else:
                    _aliases += "None"
                _key = ("{s:%d}" % _maxkeylen).format(s="'%s'" % key)
                print(f"{_key}{_aliases}")
    return (model_keys, model_aliases) if return_aliases else model_keys


def get_model_details(cls, key_or_alias, verbose=True):
    models = _MODELS.get(cls, {})
   
    if key_or_alias in models:
        key = key_or_alias
        alias = None
    else:
        aliases = _ALIASES.get(cls, {})
        alias = key_or_alias
        alias in aliases or ValueError(f"'{alias}' is neither a key or alias for '{cls.__name__}'")
        key = aliases[alias]
    if verbose:
        print(
            "Found model '{model}'{alias_str} for '{clazz}'.".format(
                model=key,
                clazz=cls.__name__,
                alias_str=("" if alias is None else " with alias '%s'" % alias),
            )
        )
    return key, alias, models[key]


        
def get_model_folder(cls, key_or_alias):
    key, alias, m = get_model_details(cls, key_or_alias)
    target = Path("models") / cls.__name__ / key
    path = Path(
        get_file(
            fname=key + ".zip",
            origin=m["url"],
            file_hash=m["hash"],
            cache_subdir=target,
            extract=True,
        )
    )
  
    assert path.exists() and path.parent.exists()
    return path.parent
               


def get_model_instance(cls, key_or_alias):
    path = get_model_folder(cls, key_or_alias)
    json_file = path /key_or_alias /f"{key_or_alias}.json"  
    weight_file = path / key_or_alias/f"{key_or_alias}.pth"

    if not json_file or not weight_file:
        raise FileNotFoundError("Required .json or .pth file not found in the model folder.")

    with open(json_file, "r") as f:
        config = json.load(f)

    timesteps = config.get("timesteps", 1) 
    hook_type = config.get("hook_type", None)
    bio_unroll = config.get("bio_unroll", False)
    num_classes = config.get("num_classes", 1)

    model = cls(timesteps=timesteps, hook_type=hook_type, bio_unroll=bio_unroll, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(weight_file, map_location=device)

    filtered_state_dict = {
        k: v if not (isinstance(v, torch.Tensor) and v.dtype != torch.float64) else v.float()
        for k, v in state_dict.items()
        if not any(x in k for x in ["total_ops", "total_params"])
    }
    
    model.load_state_dict(filtered_state_dict)

   
    return model
