import einops
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import math
import os
from torchvision.datasets.imagenet import load_meta_file
from .models.vqgan import vqgan
from .models.vq_model import VQ_models


class VQGANTok(nn.Module):
    def __init__(self, vocab_size, checkpoint=None):
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.vae_config = dict(
            embed_dim=256,
            n_embed=vocab_size,
            sane_index_shape=True,
            ddconfig=dict(
                double_z=False,
                z_channels=256,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 1, 2, 2, 4],  # num_down = len(ch_mult)-1
                num_res_blocks=2,
                attn_resolutions=[16],
                dropout=0.0,
            ),
            # lossconfig=dict(
            #     target=".modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
            #     params=dict(
            #         disc_conditional=False,
            #         disc_in_channels=3,
            #         disc_start=0,
            #         disc_weight=0.8,
            #         codebook_weight=1.0,
            #     ),
            # ),
            lossconfig=None,
        )
        self.vae = vqgan.VQModel(**self.vae_config).cpu()
        if checkpoint is not None:
            assert vocab_size in [1024, 16384]
            info = self.vae.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
            print(info, flush=True)

    def batch_encode(self, x: torch.Tensor):
        self.vae.eval()
        b, c, h, w = x.shape
        quant, emb_loss, info = self.vae.encode(x)
        xb, xc, xh, xw = quant.shape
        # token_id = info[-1].reshape(b, xh, xw)
        token_id = info[-1]
        return token_id.view(b, -1)
    
    def batch_decode(self, x: torch.Tensor):
        self.vae.eval()
        b, l = x.shape
        x = x.clamp(min=0, max=self.vocab_size - 1)
        h = int(math.sqrt(l))
        w = l // h
        assert w * h == l, "we only accept square image."
        quant = self.vae.quantize.embedding(x.view(-1))
        quant = quant.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        dec = self.vae.decode(quant)
        b, c, h, w = dec.shape
        return dec
        
        
class LlamaGenTok(VQGANTok):
    def __init__(self, vocab_size, checkpoint=None):
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.vae = VQ_models['VQ-16'](codebook_size=vocab_size).cpu()
        if checkpoint is not None:
            assert vocab_size in [16384]
            info = self.vae.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=False)
            print(info, flush=True)
   
    
def build_image_tokenizer(vocab_size, checkpoint):
    if "vq_ds16_c2i.pt" in checkpoint:
        return LlamaGenTok(vocab_size, checkpoint=checkpoint)
    else:
        return VQGANTok(vocab_size, checkpoint=checkpoint)


class ImageNetVisionTokenizer:
    def __init__(
        self, vocab_size, checkpoint=None, add_mask_token=True, add_special_tokens=True
    ):
        self.pad_token_id = None
        self.pad_token = None
        if add_mask_token:
            self.mask_token = vocab_size
            self.mask_token_id = vocab_size
            self.vocab_size = vocab_size + 1  # mask token
        else:
            self.vocab_size = vocab_size
        if add_special_tokens:
            self.bos_token_id = vocab_size
            self.bos_token = vocab_size
            self.eos_token_id = vocab_size + 1
            self.eos_token = vocab_size + 1
            self.vocab_size = self.vocab_size + 2  # mask token, bos_token, eos_token
        else:
            self.vocab_size = self.vocab_size
        
        self.dummy = True
        self.tokenizer = build_image_tokenizer(vocab_size, checkpoint=checkpoint)
        
    def __len__(self):
        return self.vocab_size

    def __call__(self, x):
        return x


def load_wnids(file_list):
    with open(file_list, "r") as f:
        vlist = f.readlines()
    samples = [x.strip() for x in vlist if x.strip() != ""]
    return samples


def load_imagenet_val(file_list, root, wnid_to_idx):
    with open(file_list, "r") as f:
        vlist = f.readlines()[1:]
    samples = [(os.path.join(root, v.split(",")[0] + ".JPEG"), wnid_to_idx[v.split(",")[1].split(" ")[0]]) for v in vlist]
    targets = [s[1] for s in samples]
    return samples, targets


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, train, hflip=True, **kwargs):
        if train:
            super().__init__(root=root, split="train")
        else:
            file_list = os.path.join(os.path.dirname(__file__), "imagenet/LOC_val_solution.csv")
            self.wnids = load_wnids(os.path.join(os.path.dirname(__file__), "imagenet/class_wnids.txt"))
            root = self.root = os.path.expanduser(root)
            self.split = "val"
            self.parse_archives()
            wnid_to_classes = load_meta_file(self.root)[0]
            self.root = root
            self.wnid_to_idx = {x: i for i, x in enumerate(self.wnids)}
            self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
            self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}
            self.samples, self.targets = load_imagenet_val(file_list, root=os.path.join(root, "val"), wnid_to_idx=self.wnid_to_idx)
            self.loader = torchvision.datasets.folder.default_loader
            self.target_transform = None
            self.transform = None
        assert len(self.wnid_to_idx) == 1000, f"{len(self.wnid_to_idx)}"
            
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.RandomHorizontalFlip(0.5 if hflip else -1.0),
            torchvision.transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample = sample * 2 - 1 # should be [-1, 1] based on taming-transformer
        return {"input_ids": sample, "labels": target, "tovae": True}


