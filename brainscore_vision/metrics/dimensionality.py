import numpy as np
from brainscore_core import Score

"""
@inproceedings{esser2020disentangling,
  title={A disentangling invertible interpretation network for explaining latent representations},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9223--9232},
  year={2020},
  url={https://arxiv.org/abs/2004.13166}
}
"""

class Dimensionality:
    def __init__(self, factor):
        self.factor = factor #factor to return
        
    def __call__(self,assembly1, assembly2):
        #each assembly should have 3 dimensions: (factor, sample_num, neuron_i) 
        dims_percent = self._dim_est(assembly1,assembly2)
        factor_dim = dims_percent[self.factor]
        return Score(factor_dim)
        
    def _dim_est(self, za, zb):
        score_by_factor = dict() 
        zall = np.concatenate([za,zb], 0)
        mean = np.mean(zall, 0, keepdims=True) #mean and variance per neuron
        var = np.sum(np.mean((zall-mean)*(zall-mean), 0))
        for f in set(za["factor"].values):
            za_factor = za[za.factor == f].values
            zb_factor = zb[zb.factor == f].values
            mean_f_factor = 0.5*(np.mean(za_factor, 0, keepdims=True)+np.mean(zb_factor, 0, keepdims=True))
            cov_f_factor = np.mean((za_factor-mean_f_factor)*(zb_factor-mean_f_factor), 0)
            raw_score_f_factor = np.sum(cov_f_factor)
            score_by_factor[f] = raw_score_f_factor/var   
        
        assert "residual" not in score_by_factor 
        score_by_factor["residual"] = 1.0 #by default
        score_names = score_by_factor.keys()
        scores = np.fromiter(score_by_factor.values(),dtype=float)
        dims_percent = self._softmax_dim(score_names,scores,za.shape[1])  
        return dims_percent
    
    def _softmax_dim(self, score_names, scores, N):
        m = np.max(scores)
        e = np.exp(scores-m)
        softmaxed = e / np.sum(e)
        dim = N
        dims = [int(s*dim) for s in softmaxed]
        dims[-1] = dim - sum(dims[:-1])
        dims_percent = {}
        for name, i in zip(score_names,range(len(scores))):
            dims_percent[name] = dims[i] / sum(dims)
        return dims_percent