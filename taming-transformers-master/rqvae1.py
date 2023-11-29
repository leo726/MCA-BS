from functools import partial
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
from vqvae1 import VectorQuantize
from coco import CodebookAttention

from einops import rearrange, repeat

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        shared_codebook = False,
        # heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        **kwargs
    ):
        super().__init__()
        # assert heads == 1, 'residual vq is not compatible with multi-headed codes'

        self.num_quantizers = num_quantizers

        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

        self.quantize_dropout = quantize_dropout

        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    def forward(
        self,
        x,
        return_all_codes = False
    ):
        b, n, *_, num_quant, device = *x.shape, self.num_quantizers, x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        if should_quantize_dropout:
            rand_quantize_dropout_index = randrange(self.quantize_dropout_cutoff_index, num_quant)

        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                null_indices = torch.full((b, n), -1., device = device, dtype = torch.long)
                null_loss = torch.full((1,), 0., device = device, dtype = x.dtype)

                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            quantized = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        return quantized_out

if __name__ == "__main__":
    residual_vq = ResidualVQ(
                                    dim = 256,
                                    codebook_size = 1024,
                                    num_quantizers = 4,
                                    kmeans_init = True,   # set to True
                                    kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
                                    heads=8,
                                    codebook_dim=32,
                                    decay = 0.99,
                                    separate_codebook_per_head = True
        )

    x = torch.randn(1, 1024, 256)
    quantized, indices, commit_loss = residual_vq(x)
    print(type(x))
    print(quantized)
    print(indices)
    print(commit_loss)