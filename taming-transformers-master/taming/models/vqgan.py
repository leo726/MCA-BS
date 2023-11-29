import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, ResEncoder, ResDecoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from vqvae1 import VectorQuantize as MultiheadVQ
from rqvae import ResidualVQ
from coco import CodebookAttention
from vitcoder import ViTEncoder, ViTDecoder
from typing import List, Tuple, Dict, Any, Optional
from einops import rearrange

from taming.models.restormer import Restormer, TransformerBlock
from taming.models.network_swinir import SwinIR
import math
from taming.modules.vqvae.rq_quantize import RQBottleneck
from torchvision import transforms
import numpy as np
import cv2

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                #  ckpt_path='/home/yangling/VQGAN/taming-transformers-master/pretrained-model/vqgan_imagenet_f16_1024.ckpt',
                 ckpt_path='/mnt/sda/yangling/VQGAN/taming-transformers-master/pretrained-model/imagenet_vitvq_base.ckpt',
                #  ckpt_path = '/mnt/sdc/yangling/logs/2023-06-19T20-49-16_BoneShadowX-rays_vqgan/checkpoints/epoch=000962.ckpt',
                #  ckpt_path='/mnt/sdc/yangling/logs/2023-07-12T23-20-49_BoneShadowX-rays_vqgan/checkpoints/epoch=000990.ckpt',
                #  ckpt_path='/mnt/sdc/yangling/logs/2023-07-17T16-07-54_BoneShadowX-rays_vqgan/checkpoints/epoch=000986.ckpt',
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        # self.encoder = Encoder(**ddconfig)
        # self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.vitencoder = ViTEncoder(image_size=256, patch_size=8, dim=768, depth=12, heads=12, mlp_dim=3072)
        self.vitdecoder = ViTDecoder(image_size=256, patch_size=8, dim=768,depth=12,heads=12,mlp_dim=3072)
        # VQGan codebook
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)
        
        # multi-head codebook
        # self.quantize = MultiheadVQ(dim = 256,
        #                             codebook_dim = 32,                  # a number of papers have shown smaller codebook dimension to be acceptable
        #                             heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
        #                             separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
        #                             codebook_size = 1024, # 8196
        #                             accept_image_fmap = True
        #                            )

        # multi-head codebook attention
        self.quantize = MultiheadVQ(
                                    dim = 256, # 32
                                    codebook_dim = 32,    # 4              # a number of papers have shown smaller codebook dimension to be acceptable
                                    heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
                                    separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
                                    codebook_size = 1024,
                                    accept_image_fmap = False
                                )
        
        # rq codebook
        # self.quantize = ResidualVQ(
        #                             dim = 256,
        #                             codebook_size = 1024,
        #                             num_quantizers = 4,
        #                             kmeans_init = True,   # set to True
        #                             kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
        #                             heads=8,
        #                             codebook_dim=32,
        #                             decay = 0.99,
        #                             separate_codebook_per_head = True,
        # )
        
        # self.codebookAttention = CodebookAttention(
        #                                             codebook_dim = 256,
        #                                             depth = 1,
        #                                             num_latents = 256,
        #                                             latent_dim = 256,
        #                                             latent_heads = 8,
        #                                             latent_dim_head = 64,
        #                                             cross_heads = 1,
        #                                             cross_dim_head = 64)

        # self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # self.pre_quant = nn.Linear(768, 32)
        # self.post_quant = nn.Linear(32, 768)
        # self.restormer=Restormer()
        # self.swin = SwinIR(upscale=1, img_size=(256, 256),in_chans=1,
        #            window_size=8, img_range=1., depths=[6, 6, 6, 6],
        #            embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
            
        self.pre_quant = nn.Linear(768, 256)
        self.post_quant = nn.Linear(256, 768)
        # self.val_loss_value = 100.0
        # self.val_loss = 'val/rec_loss'
        
        
        
        # pretrain_model_path = '/home/yangling/VQGAN/taming-transformers-master/taming/models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
        # pretrained_model = torch.load(pretrain_model_path)
        # param_key_g = 'params'
        # self.swin.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
 
    # def encode(self, x):
    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # h1 = self.encoder(x)
        # h1 = self.quant_conv(h1)

        # vit-encoder
        h = self.vitencoder(x)
        h = self.pre_quant(h)
         
        # code attention
        # h = h.reshape(h.shape[0],h.shape[1],h.shape[2]*h.shape[3])   # RQVAE
        # codebook = self.quantize.codebook.reshape(1,1024,256)
        # h = h.reshape(256,256)
        # x1 = self.codebookAttention(h,codebook)

        # quant, emb_loss, info = self.quantize(h)  # VQGAN:quant, emb_loss, info; Multi: quant, info, emb_loss
        # x1 = quant 
        x1 = self.quantize(h)
        return x1

    def decode(self, quant):
        # quant = quant.reshape(quant.shape[0],quant.shape[1],int(math.sqrt(quant.shape[2])),int(math.sqrt(quant.shape[2])))  # RQVAE
        # quant = quant.reshape(1,256,16,16)
        # quant = self.post_quant_conv(quant)
        # dec = self.decoder(quant)

        # vitdecoder
        quant = self.post_quant(quant)
        quant = quant.reshape(1,1024,768)
        dec = self.vitdecoder(quant)
        return dec

    def forward(self, input):
        # quant, diff, _ = self.encode(input)
        quant = self.encode(input)
        dec = self.decode(quant)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)
        loss = F.mse_loss(x, xrec)

        # ssim loss
        # loss = (1 - self.ssim(x,xrec))*100
        # loss = torch.tensor(loss,requires_grad=True)

        self.log("train_loss", loss)
        loss.requires_grad_(True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)
        loss = F.mse_loss(x, xrec)

        # ssim loss
        # loss = (1 - self.ssim(x,xrec))*100
        # loss = torch.tensor(loss,requires_grad=True)

        self.log("val_loss", loss)
        loss.requires_grad_(True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.vitencoder.parameters())+
                                #   list(self.restormer.parameters())+
                                #   list(self.swin.parameters())+   # swin参数
                                  list(self.vitdecoder.parameters())+
                                  list(self.quantize.parameters())+
                                # #   list(self.codebookAttention.parameters())+
                                #   list(self.quant_conv.parameters())+
                                #   list(self.post_quant_conv.parameters()),
                                  list(self.pre_quant.parameters())+
                                  list(self.post_quant.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    def ssim(self, img1, img2):
        img1 = torch.narrow(img1,1,0,1).reshape(256,256)
        img2 = torch.narrow(img2,1,0,1).reshape(256,256)
        img1 = np.array(img1.cpu().detach())
        img2 = np.array(img2.cpu().detach())
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

class BoneVQModel(pl.LightningModule):     # load bone image and output bse bone image
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                #  ckpt_path = '/home/yangling/VQGAN/taming-transformers-master/logs/2023-02-28T18-02-54_BoneShadowX-rays_vqgan/checkpoints/epoch=000549.ckpt',
                #  ckpt_path = '/home/yangling/VQGAN/taming-transformers-master/logs/2023-02-24T14-56-08_BoneShadowX-rays_vqgan/checkpoints/epoch=000949.ckpt',
                #  ckpt_path = '/mnt/sdc/yangling/logs/2023-06-19T20-49-16_BoneShadowX-rays_vqgan/checkpoints/epoch=000962.ckpt',
                #  ckpt_path='/mnt/sdc/yangling/logs/2023-07-15T21-32-02_BoneShadowX-rays_vqgan/checkpoints/epoch=000418.ckpt',
                #  ckpt_path= '/mnt/sdc/yangling/logs/2023-07-15T21-36-10_BseBoneShadowX-rays_vqgan/checkpoints/epoch=000061.ckpt',
                 ckpt_path = '/mnt/sdc/yangling/logs/2023-08-22T16-13-17_BoneShadowX-rays_vqgan/checkpoints/epoch=000993.ckpt',
                 ignore_keys=[],
                 #  image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        # self.image_key = image_key
        # self.encoder = Encoder(**ddconfig)
        # self.encoder2 = Encoder(**ddconfig) # load the bse bone images
        # self.decoder = Decoder(**ddconfig)
        self.vitencoder = ViTEncoder(image_size=256, patch_size=8, dim=768, depth=12, heads=12, mlp_dim=3072)
        self.vitencoder2 = ViTEncoder(image_size=256, patch_size=8, dim=768, depth=12, heads=12, mlp_dim=3072) # load the bse bone images
        self.vitdecoder = ViTDecoder(image_size=256, patch_size=8, dim=768,depth=12,heads=12,mlp_dim=3072)
        # self.loss = instantiate_from_config(lossconfig)
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)
        # multi-head codebook
        # self.quantize = MultiheadVQ(dim = 256,codebook_dim = 32,                  # a number of papers have shown smaller codebook dimension to be acceptable
        #                             heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
        #                             separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
        #                             codebook_size = 1024, # 8196
        #                             accept_image_fmap = True
        #                            )
        # self.quantize = ResidualVQ(
        #                             dim = 256,
        #                             codebook_size = 1024,
        #                             num_quantizers = 4,
        #                             kmeans_init = True,   # set to True
        #                             kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
        #                             heads=8,
        #                             codebook_dim=32,
        #                             decay = 0.99,
        #                             separate_codebook_per_head = True,
        # )
        # Before Encoder Swin
        # self.swin = SwinIR(upscale=1, img_size=(256, 256),in_chans=3,
        #            window_size=8, img_range=1., depths=[6, 6, 6, 6],
        #            embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

        # After Encoder Swin
        # self.swin = SwinIR(upscale=1, img_size=(16, 16),in_chans=256,
        #            window_size=8, img_range=1., depths=[6, 6, 6, 6],
        #            embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

        # self.swin2 = SwinIR(upscale=1, img_size=(256, 256),in_chans=1,
        #            window_size=8, img_range=1., depths=[6, 6, 6, 6],
        #            embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
        # self.restormerblocks = nn.Sequential(*[TransformerBlock(dim=256, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(4)])
        
        # codeattention codebook
        self.quantize = MultiheadVQ(
                                    dim = 256,
                                    codebook_dim = 32,                  # a number of papers have shown smaller codebook dimension to be acceptable
                                    heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
                                    separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
                                    codebook_size = 1024,
                                    accept_image_fmap = True
                                )

        # vit-encoder decoder
        # self.quantize = MultiheadVQ(
        #                             dim = 256,
        #                             codebook_dim = 32,                  # a number of papers have shown smaller codebook dimension to be acceptable
        #                             heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
        #                             separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
        #                             codebook_size = 1024,
        #                             accept_image_fmap = False,
        #                             channel_last = False
        #                         )
        
        # rq codebook
        # self.quantize = ResidualVQ(
        #                             dim = 256,
        #                             codebook_size = 1024,
        #                             num_quantizers = 4,
        #                             kmeans_init = True,   # set to True
        #                             kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
        #                             heads=8,
        #                             codebook_dim=32,
        #                             decay = 0.99,
        #                             separate_codebook_per_head = True,
        # )
        
        # self.codebookAttention = CodebookAttention(
        #                                             codebook_dim = 256,
        #                                             depth = 1,
        #                                             num_latents = 256,
        #                                             latent_dim = 256,
        #                                             latent_heads = 8,
        #                                             latent_dim_head = 64,
        #                                             cross_heads = 1,
        #                                             cross_dim_head = 64)
        self.restormer = Restormer()
        # self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # self.pre_quant = nn.Linear(768, 32)
        # # self.pre_quant2 = nn.Linear(768, 32)
        # self.post_quant = nn.Linear(32, 768)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # self.image_key = image_key

        # decoder add restomer layers
        # self.decoder.up[1].add_module('3', nn.Sequential(*[TransformerBlock(dim=128, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(8)]))
        # self.decoder.up[2].add_module('3', nn.Sequential(*[TransformerBlock(dim=256, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(8)]))
        # self.decoder.up[3].add_module('3', nn.Sequential(*[TransformerBlock(dim=256, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(8)]))
        # self.decoder.up[4].add_module('3', nn.Sequential(*[TransformerBlock(dim=512, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(8)]))
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.pre_quant = nn.Linear(768, 256)
        self.post_quant = nn.Linear(256, 768)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # h = self.encoder(x)
        # h = self.quant_conv(h)

        # vit-encoder
        h = self.vitencoder(x)
        h = self.pre_quant(h)
        h = rearrange(h, f'b (h w) c -> b c h w', h=256) # h =32
        
        # h = h.reshape(h.shape[0],h.shape[1],h.shape[2]*h.shape[3])   # RQVAE
        # codebook = self.quantize.codebook.reshape(1,1024,256)
        # h = h.reshape(256,256)
        # x1 = self.codebookAttention(h,codebook)

        x1 = self.quantize(h)
        return x1
    
    def codebook(self, x):
        
        # h = self.encoder(x)
        h = self.vitencoder(x)
        # h = self.pre_quant(h)
        # h = rearrange(h, f'b (h w) c -> b c h w', h=32)
        return h
    
    def encode2(self, x2):
        # restormer
        x2 = self.restormer(x2)

        # swinIR
        # x2 = self.swin.conv_first(x2)
        # x2 = self.swin.conv_after_body(self.swin.forward_features(x2)) + x2
        # x2 = self.swin.upsample(x2)

        # h2 = self.encoder2(x2)
        # h2 = self.restormerblocks(h2)

        # # swinIR
        # h2 = self.swin.conv_first(h2)
        # h2 = self.swin.conv_after_body(self.swin.forward_features(h2)) + h2
        # h2 = self.swin.upsample(h2)

        # h = self.quant_conv(h2)
        
        # h = h.reshape(h.shape[0],h.shape[1],h.shape[2]*h.shape[3])   # RQVAE
        # codebook = self.quantize.codebook.reshape(1,1024,256)
        # h = h.reshape(256,256)
        # x2 = self.codebookAttention(h,codebook)

        # vit-encoder
        h = self.vitencoder2(x2)
        h = self.pre_quant(h)
        h = rearrange(h, f'b (h w) c -> b c h w', h=32)

        x2 = self.quantize(h)
        return x2
    
    def codebook2(self, x2):
        # restormer
        x2 = self.restormer(x2)

        # swinIR
        # x2 = self.swin.conv_first(x2)
        # x2 = self.swin.conv_after_body(self.swin.forward_features(x2)) + x2
        # x2 = self.swin.upsample(x2)

        # h2 = self.encoder2(x2)
        # h2 = self.restormerblocks(h2)
        
        # swinIR
        # h2 = self.swin.conv_first(h2)
        # h2 = self.swin.conv_after_body(self.swin.forward_features(h2)) + h2
        # h2 = self.swin.upsample(h2)
        h2 = self.vitencoder2(x2)
        # h2 = self.pre_quant2(h2)
        # h2 = rearrange(h2, f'b (h w) c -> b c h w', h=32)
        return h2
    
    def decode(self, quant):
        # quant = quant.reshape(quant.shape[0],quant.shape[1],int(math.sqrt(quant.shape[2])),int(math.sqrt(quant.shape[2])))  # RQVAE
        # quant = quant.reshape(1,256,16,16)
        # quant = self.post_quant_conv(quant)
        # dec = self.decoder(quant)

        # vitdecoder
        quant = self.post_quant(quant)
        dec = self.vitdecoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x1, x2):
        quant = self.encode2(x1)
        dec = self.decode(quant)
        quant2 = self.encode(x2)
        dec2 = self.decode(quant2)
        return dec, dec2

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x1 = self.get_input(batch, 'bone_base_image') # self.image_key)
        x2 = self.get_input(batch, 'bse_base_image') # self.image_key)
        h2 = self.codebook2(x1)
        h = self.codebook(x2)

        # ssim loss
        # xrec,xrec2 = self(x1,x2)
        # loss = (1 - self.ssim(x2,xrec))*50
        # loss = torch.tensor(loss,requires_grad=True)
        loss = F.mse_loss(h2, h)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = self.get_input(batch, 'bone_base_image') # self.image_key)
        x2 = self.get_input(batch, 'bse_base_image') # self.image_key)
        h2 = self.codebook2(x1)
        h = self.codebook(x2)
        # ssim loss
        # x, xrec = self(x1,x2)
        # loss = (1 - self.ssim(x2,xrec))*50
        # loss = torch.tensor(loss,requires_grad=True)
        loss = F.mse_loss(h2, h)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.vitencoder2.parameters())
                                #   +list(self.pre_quant2.parameters()),
                                  +list(self.restormer.parameters()),
                                    # +list(self.swin.parameters()),
                                #   +list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, 'bone_base_image')
        x = x.to(self.device)
        x2 = self.get_input(batch, 'bse_base_image')
        x2 = x2.to(self.device)
        # ximg = self.restormer.getimg(x)
        # xadd = self.restormer.getadd(x)
        # xres = self.restormer(x)
        xrec, xrec2 = self(x,x2)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        if x2.shape[1] > 3:
            # colorize with random projection
            assert xrec2.shape[1] > 3
            x2 = self.to_rgb(x2)
            xrec2 = self.to_rgb(xrec2)
        log["bone_inputs"] = x
        log["bone_reconstructions"] = xrec   
        log["bse_inputs"] = x2
        log["bse_reconstructions"] = xrec2
        
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    def ssim(self, img1, img2):
        img1 = torch.narrow(img1,1,0,1).reshape(256,256)
        img2 = torch.narrow(img2,1,0,1).reshape(256,256)
        img1 = np.array(img1.cpu().detach())
        img2 = np.array(img2.cpu().detach())
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

class ChestVQModel(pl.LightningModule):     # load chest image and output bse chest image
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 #  ckpt_path='/home/yangling/VQGAN/taming-transformers-master/logs/2022-10-27T00-40-30_BoneShadowX-rays_vqgan/checkpoints/last.ckpt',
                 # 256 黑白颠倒去骨前生成去骨后图像
                #  ckpt_path='/home/yangling/VQGAN/taming-transformers-master/logs/2023-03-03T13-50-24_BseBoneShadowX-rays_vqgan/checkpoints/epoch=000598.ckpt',
                 ckpt_path='/mnt/sdc/yangling/logs/2023-07-19T20-18-33_BseBoneShadowX-rays_vqgan/checkpoints/epoch=000473.ckpt',
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        # self.encoder = Encoder(**ddconfig)
        # self.encoder2 = Encoder(**ddconfig)
        # self.decoder = Decoder(**ddconfig)
        # # self.loss = instantiate_from_config(lossconfig)
        # # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        # #                                 remap=remap, sane_index_shape=sane_index_shape)
        # self.quantize = ResidualVQ(
        #                             dim = 256,
        #                             codebook_size = 1024,
        #                             num_quantizers = 4,
        #                             kmeans_init = True,   # set to True
        #                             kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
        #                             heads=8,
        #                             codebook_dim=32,
        #                             decay = 0.99,
        #                             separate_codebook_per_head = True,
        # )

        # self.restormer = Restormer()
        # self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.vitencoder = ViTEncoder(image_size=256, patch_size=8, dim=768, depth=12, heads=12, mlp_dim=3072)
        self.vitencoder2 = ViTEncoder(image_size=256, patch_size=8, dim=768, depth=12, heads=12, mlp_dim=3072) # load the bse bone images
        self.vitdecoder = ViTDecoder(image_size=256, patch_size=8, dim=768,depth=12,heads=12,mlp_dim=3072)
        self.quantize = MultiheadVQ(
                                    dim = 32,
                                    codebook_dim = 4,                  # a number of papers have shown smaller codebook dimension to be acceptable
                                    heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
                                    separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
                                    codebook_size = 1024,
                                    accept_image_fmap = True
                                )
        self.restormer = Restormer()
        self.pre_quant = nn.Linear(768, 32)
        self.post_quant = nn.Linear(32, 768)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # def encode(self, x):
    #     x = self.restormer(x)
    #     h = self.encoder2(x)
    #     h = self.quant_conv(h)
    #     h = h.reshape(h.shape[0],h.shape[1],h.shape[2]*h.shape[3])  # RQVAE
    #     quant, emb_loss, info = self.quantize(h)
    #     return quant, emb_loss, info

    # def decode(self, quant):
    #     quant = quant.reshape(quant.shape[0],quant.shape[1],int(math.sqrt(quant.shape[2])),int(math.sqrt(quant.shape[2])))  # RQVAE
    #     quant = self.post_quant_conv(quant)
    #     dec = self.decoder(quant)
    #     return dec

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    # def forward(self, x):
    #     quant, diff, _ = self.encode(x)
    #     dec = self.decode(quant)
    #     return dec, diff

    def encode(self, x):
        # restormer
        x = self.restormer(x)

        # vit-encoder
        h = self.vitencoder2(x)
        h = self.pre_quant(h)
        h = rearrange(h, f'b (h w) c -> b c h w', h=32)

        x2 = self.quantize(h)
        return x2
    
    def codebook2(self, x2):
        # restormer
        x2 = self.restormer(x2)
        h2 = self.vitencoder2(x2)
        return h2
    
    def decode(self, quant):
        # vitdecoder
        quant = self.post_quant(quant)
        dec = self.vitdecoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x):
        quant = self.encode(x)
        dec = self.decode(quant)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.vitencoder.parameters()),
        #                         #   list(self.encoder2.parameters())+
        #                         #   list(self.decoder.parameters())+
        #                         #   list(self.quantize.parameters())+
        #                         #   list(self.quant_conv.parameters())+
        #                         #   list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           
    