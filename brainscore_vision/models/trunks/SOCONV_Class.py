#!/usr/bin/env python
# coding: utf-8

import torch 
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import math
import copy
import pdb
from statistics import mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SOCONV (nn.Module):
  def __init__ (self,in_channels, num_kernels,Kernel_size,stride,alpha=8,sigma=8,decay_rate=-3,padding=0):
    super(SOCONV, self).__init__()

    self.decay_rate=decay_rate


    self.in_channels=in_channels

    self.stride = stride
    
    self.kh, self.kw = _pair(Kernel_size)

    self.dh, self.dw = _pair(stride)
    
    self.num_kernels=num_kernels

    self.NP=(self.in_channels*Kernel_size**2)

    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels, kernel_size=Kernel_size, padding=padding, stride=stride).to(device)

    self.alpha = float(alpha)

    self.sigma=float(sigma)

    self.alpha_initial=self.alpha

    self.sigma_initial=self.sigma

    self.locations = nn.Parameter(torch.Tensor(list(self.neuron_locations())), requires_grad=False)

    self.bmu_locations = None

    self.bmu_weights=None

    self.neighbourhood_func=None

    self.bmu_indexes=None

    self.resize_weight_grad=None

    self.resize_weight=None

    self.SUB=None

    self.mean=None

    self.image_extract=None

    self.resize_out_conv=None

    self.bmu_indexes_2D=None

    self.torch_gradient=None

    self.bmu_value_weight=None

    self.gradient_norm =[]

    self.min_alpha = 0.003

    self.min_sigma =0.003

  def gather_nd(self,params, indices, batch_dims=0):
        
        """ The same as tf.gather_nd.
        indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

        output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

        Args:
            params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
            indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

        Returns: gathered Tensor.
            shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

        """
        if batch_dims == 0:
            orig_shape = list(indices.shape)
            num_samples = int(np.prod(orig_shape[:-1]))
            m = orig_shape[-1]
            n = len(params.shape)

            if m <= n:
                out_shape = orig_shape[:-1] + list(params.shape[m:])
            else:
                raise ValueError(
                    f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
                )
            indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
            output = params[indices]    # (num_samples, ...)
            return output.reshape(out_shape).contiguous()
        else:
            batch_shape = params.shape[:batch_dims]
            orig_indices_shape = list(indices.shape)
            orig_params_shape = list(params.shape)
            assert (
                batch_shape == indices.shape[:batch_dims]
            ), f'if batch_dims is not 0, then both "params" and "indices" have batch_dims leading batch dimensions that exactly match.'
            mbs = np.prod(batch_shape)
            if batch_dims != 1:
                params = params.reshape(mbs, *(params.shape[batch_dims:]))
                indices = indices.reshape(mbs, *(indices.shape[batch_dims:]))
            output = []
            for i in range(mbs):
                output.append(gather_nd(params[i], indices[i], batch_dims=0))
            output = torch.stack(output, dim=0)
            output_shape = orig_indices_shape[:-1] + list(orig_params_shape[orig_indices_shape[-1]+batch_dims:])
            return output.reshape(*output_shape).contiguous()
  


  def forward(self,x): 
    return self.conv1(x)

  def extract_patches(self,x):
      """
      extract patches out of images and reshaping it

      """
      b, c, h, w = x.size()

      batch_size=b
      
      kh, kw = self.kh,self.kw

      dh, dw = self.dh,self.dw

      x = x.unfold(2, kh, dh).unfold(3, kw, dw)

      out=x.permute(0,2,3,4,5,1).reshape([batch_size,self.num_patches,self.NP]).reshape(batch_size*self.num_patches,-1)
     
      return out
  
  def BMU(self):
      
      self.torch_gradient=self.reshape_gradient()

      self.bmu_indexes_2D=((self.torch_gradient==torch.max(self.torch_gradient)).nonzero())

      bmu_indexes = self.torch_gradient.argmax() #change to argmax
            
      bmu_locations=self.locations[bmu_indexes]

      self.resize_weight_grad=self.resize_weights_gradient()

      self.resize_weight=self.resize_weights()

      self.bmu_value_weight=self.resize_weight[bmu_indexes]


      bmu_value=self.resize_weight_grad[bmu_indexes]

        
      return bmu_indexes,bmu_locations,bmu_value

  def neuron_locations(self):
      
      F1=int(math.sqrt(self.num_kernels))
      sizes=[F1,F1]
      def flatten(test_tuple):
          if isinstance(test_tuple, int):
            res = [test_tuple]
            return res
          res = []
          for sub in test_tuple:
            res += flatten(sub)
          return res

      for i in range(len(sizes)-1): 
        current_result = []
        if i == 0: 
          previous_result = []
          items1 = range(sizes[i])  # [0, 2, ..., 9]
        else: 
          items1 = previous_result
        items2 = range(sizes[i+1])
        for it1 in items1: 
          for it2 in items2: 
            current_result.append((it1, it2))   # [(0, 0), (0, 1), ... , (9, 10)]
        previous_result = copy.deepcopy(current_result)
      for i in range(len(current_result)): 
        current_result[i] = flatten(current_result[i])
      return current_result

  def resize_weights_gradient(self):         
      return (self.conv1.weight.grad.reshape(self.num_kernels,self.NP)) #[b*P,num_kernels,NP]

  def resize_weights(self):
      return (self.conv1.weight.reshape(self.num_kernels,self.NP))


  def get_alpha_sigma(self):
        return self.alpha,self.sigma
  
  
  def reshape_gradient(self):
    torch_gradient = torch.linalg.norm(self.conv1.weight.grad.reshape(self.num_kernels,-1), dim=(1)).reshape(int(math.sqrt(self.num_kernels)),int(math.sqrt(self.num_kernels)))
    return torch_gradient



  def self_organizing(self, current_iter, max_iter):
        #W(t+1)=W(t)+A(t)L(T)(V(T)-W(t)
        
        itteration = 1.0 - current_iter / max_iter

        itteration = torch.exp(torch.tensor(self.decay_rate * current_iter / max_iter))

        self.alpha = (self.alpha_initial*itteration)     

        self.sigma = (self.sigma_initial*itteration)

        self.sigma = max(self.min_sigma, self.sigma)

        self.alpha = max(self.min_alpha, self.alpha)

        self.bmu_indexes,self.bmu_locations,self.bmu_value=self.BMU()

        distance_squares = self.locations.float()-self.bmu_locations.float()

        distance_squares.pow_(2)

        distance=distance_squares.sum(dim=-1) 

        self.neighbourhood_func = torch.exp(torch.neg(torch.div(distance, 2*self.sigma**2)))
       
        itteration=(self.neighbourhood_func * self.alpha).unsqueeze(1).to(device)

        self.SUB= self.bmu_value_weight.expand(self.num_kernels,-1)-self.conv1.weight.reshape(self.num_kernels,self.NP)
        
        delta = itteration*(self.SUB) 

        #self.conv1.weight.grad=self.conv1.weight.grad*0
        #self.conv1.weight.grad.reshape(self.num_kernels,self.NP)[self.bmu_indexes]=self.bmu_value
        #self.conv1.weight.grad.view(self.num_kernels,self.NP).data.add_(delta)
        self.conv1.weight.view(self.num_kernels,self.NP).data.add_(delta)








