import numpy as np
from collections import defaultdict

def latent_eval_agg_func(val_results):
    pred_state_cat = np.concatenate([res['pred_state'] for res in val_results], axis=0)
    next_state_cat = np.concatenate([res['next_state'] for res in val_results], axis=0)
    return latent_eval(pred_state_cat, next_state_cat)

def latent_eval(pred_states, next_states, topk=[1]):
    assert pred_states.shape == next_states.shape
    assert isinstance(pred_states, np.ndarray)
    assert isinstance(next_states, np.ndarray)
    hits_at = defaultdict(int)
    num_samples = pred_states.shape[0]
    print('Size of current evaluation batch: {}'.format(num_samples))

    # Flatten object/feature dimensions
    next_state_flat = next_states.reshape(num_samples, -1)
    pred_state_flat = pred_states.reshape(num_samples, -1)

    dist_matrix = pairwise_distance_matrix(next_state_flat, pred_state_flat)
    dist_matrix_diag = np.expand_dims(np.diag(dist_matrix), axis=-1)
    dist_matrix_augmented = np.concatenate([dist_matrix_diag, dist_matrix], axis=1)

    indices = []
    for row in dist_matrix_augmented:
        keys = (np.arange(len(row)), row)
        indices.append(np.lexsort(keys))
    indices = np.stack(indices, axis=0)

    labels = np.expand_dims(np.zeros(indices.shape[0], dtype=np.int64), axis=-1)

    for k in topk:
        match = indices[:, :k] == labels
        num_matches = match.sum()
        hits_at[k] += float(num_matches)

    match = indices == labels
    ranks = np.argmax(match, axis=1).astype(np.float64)

    reciprocal_ranks = np.reciprocal(ranks + 1.)
    rr_sum = float(reciprocal_ranks.sum())
    val_results = {
        'MRR': rr_sum / num_samples,
        'num_samples': num_samples,
        }
    print('MRR: {}'.format(rr_sum / num_samples))

    for k in topk:
        val_results['Hits_at_{}'.format(k)] = hits_at[k] / num_samples
        print('Hits @ {}: {}'.format(k, hits_at[k] / num_samples))
    return val_results

def pairwise_distance_matrix(x, y):
    assert x.shape == y.shape
    num_samples = x.shape[0]

    x = np.repeat(np.expand_dims(x, axis=1), num_samples, axis=1)
    y = np.repeat(np.expand_dims(y, axis=0), num_samples, axis=0)

    return np.power(x - y, 2).sum(2)
