import numpy as np

def parallelize(func, iterable, n_jobs=1, verbose=0):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(item) for item in iterable)