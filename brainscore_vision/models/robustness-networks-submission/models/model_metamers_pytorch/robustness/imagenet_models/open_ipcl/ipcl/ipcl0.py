'''
    re-implementation of original ipcl model using NCE.
    
    NCE implementation from: 
        https://github.com/zhirongw/lemniscate.pytorch
    
    model implementation following Moco style:
        https://github.com/facebookresearch/moco/blob/master/moco/builder.py
'''
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from IPython.core.debugger import set_trace

__all__ = ['IPCL0']

class IPCL0(nn.Module):
    """
        Build an Instance-Prototype Contrastive Learning (IPCL) model.

        https://www.biorxiv.org/content/biorxiv/early/2020/06/16/2020.06.15.153247.full.pdf

    """

    def __init__(
        self,
        base_encoder,
        numTrainImages,
        K=4096,
        T=0.07,
        out_dim=128,
        n_samples=5,
        norm_prototype=False,
        store_prototype=False,
        device=None,
    ):
        """
        base_encoder: backbone that encodes images
        numTrainImages: total number of images in training dataset
        K: queue size; number of negative items (default: 4096)
        T: softmax temperature (default: 0.07)
        norm_prototype: whether to l2 normalize the prototypes (default: False)
        """
        super(IPCL0, self).__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_encoder = base_encoder
        self.numTrainImages = numTrainImages
        self.K = K
        self.T = T
        self.register_buffer('Z', torch.tensor(-1))
        self.out_dim = out_dim
        self.n_samples = n_samples
        self.norm_prototype = norm_prototype
        self.store_prototype = store_prototype
        self.criterion = NCECriterion(numTrainImages)
        self.distributed = False
        
        # create the queue
        stdv = 1. / math.sqrt(out_dim/3)
        self.register_buffer('queue', torch.rand(out_dim, K).mul_(2*stdv).add_(-stdv))
        # self.register_buffer("queue", torch.randn(out_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _update_queue(self, new_items):
        if torch.distributed.is_initialized():
            new_items = concat_all_gather(new_items)
        K = self.K
        bs = new_items.shape[0]
        self.queue[:, 0:K] = torch.cat(
            [new_items.detach().T, self.queue[:, 0:-bs]], dim=1
        )

    @torch.no_grad()
    def _check_samples_are_continuous(self, x, y):
        bs, numFeatures = x.size(0), x.size(1)
        n_samples = self.n_samples

        # **hack alert**; compare the first and second set of labels to check whether
        # they are identical (samples_are_contiguous==True)
        # samples_are_contiguous = y.data[0] == y.data[1]
        # maybe just make this a parameter, then do an assert here to verify
        first_labels = y.data.view(-1, bs // n_samples)[0]
        second_labels = y.data.view(-1, bs // n_samples)[1]

        samples_are_interleaved = (first_labels == second_labels).all()
        samples_are_contiguous = not samples_are_interleaved

        return samples_are_contiguous

    @torch.no_grad()
    def _compute_prototypes(self, x, y, samples_are_contiguous):
        bs, numFeatures = x.size(0), x.size(1)
        n_samples = self.n_samples

        # compute the average representation across samples of the same instance
        if samples_are_contiguous:
            x_mean = x.data.reshape(-1, n_samples, numFeatures).mean(dim=1).detach()
        else:
            # 
            # reshaping as n_samples,bs//n_samples makes it so that each row
            # corresponds to a samples, each column an image, and 3rd dim = numFeatures
            # e.g., 128 output features, bs=64, n_samples=5, we're reshaping to 5x64x128, then
            # averaging across the dim=0 (across samples) to get the average
            x_mean = x.view(n_samples, bs // n_samples, -1).mean(dim=0).detach()

        if self.norm_prototype:
            x_mean = nn.functional.normalize(x_mean, dim=1)

        return x_mean

    def forward(self, x, y):

        T = self.T
        Z = self.Z
        n_samples = self.n_samples
        
        # get encoder output, l2 normalized
        x = self.base_encoder(x)
        x = nn.functional.normalize(x, dim=1)

        # check whether samples_are_contiguous
        samples_are_contiguous = self._check_samples_are_continuous(x, y)

        # compute prototypes
        x_mean = self._compute_prototypes(x, y, samples_are_contiguous)

        # positives: compute similarity between samples prototype
        if samples_are_contiguous:
            l_pos = torch.einsum(
                "nc,nc->n",
                [x, torch.repeat_interleave(x_mean, repeats=n_samples, dim=0)],
            ).unsqueeze(-1)
        else:
            l_pos = torch.einsum(
                "nc,nc->n", [x, x_mean.repeat(n_samples, 1)]
            ).unsqueeze(-1)

        # negatives: compute similarity between samples / queue
        l_neg = torch.einsum("nc,ck->nk", [x, self.queue.clone().detach()])

        # concatenate: NxK 1
        out = torch.cat([l_pos, l_neg], dim=1)

        # normalize by tau
        out = torch.exp(torch.div(out, T))
        
        if Z < 0:
            if self.distributed:
                # gather output from all sources and compute Z from all outputs (so each node uses same constant)
                gather_out = [torch.ones_like(out) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_out, out)
                out_ = torch.cat(gather_out)
                Z = (out_.mean() * self.numTrainImages).detach()
            else:
                Z = (out.mean() * self.numTrainImages).detach()
            self.Z = Z
            print("normalization constant Z is set to {:.1f}".format(Z))

        #out = torch.div(out, Z).squeeze().contiguous()
        out = torch.div(out, Z)
        
        # labels: positive key indicators
        # first item is the target (positive pair) in all cases
        targets = torch.zeros(out.shape[0], dtype=torch.long).to(x.device)

        # compute loss
        loss = self.criterion(out, targets)

        # update queue
        if self.training:
            if self.store_prototype:
                self._update_queue(x_mean)
            else:
                self._update_queue(x)

        return loss, [x, x_mean]

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output    

eps = 1e-7

class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1)-1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)
        
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt) 
        Pmt = x.select(1,0)
        Pmt_div = Pmt.add(K * Pnt + eps)
        lnPmt = torch.div(Pmt, Pmt_div)
        
        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1,1,K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)
     
        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()
        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)
        
        loss = - (lnPmtsum + lnPonsum) / batchSize
        
        return loss