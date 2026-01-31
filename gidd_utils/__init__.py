import torch
import torch.nn.functional as F
from typing import Optional
from .loss import GiddLoss
from .noise_scheduler import HybridDiffusion


# modified from https://github.com/dvruette/gidd/blob/f67acc961cc858cc69e8e6d91e7eb67a0a79151c/gidd/sampling.py
class GIDD_DenoisingStep(torch.nn.Module):
    def __init__(self, model, noise_schedule, tokenizer, min_p=0.0):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.min_p = min_p

    def forward(self, logits, z_t, t, s):
        # logits = self.model(z_t, t)
        logits[..., self.tokenizer.mask_token_id] = -1e6

        # if i > 0:
        q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s)
        q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t)
        q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

        alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
        alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

        alpha_ts = alpha_t / alpha_s
        beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

        vz_t = F.one_hot(z_t, num_classes=len(self.tokenizer))
        beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
        # q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)
        q_ts = (alpha_ts.unsqueeze(-1) * vz_t + beta_pi_ts_at_zt)

        q_st = q_ts * q_s / q_zt
        return q_st
    
        if self.min_p > 0.0:
            is_small = (q_st < self.min_p).float()
            q_st = (1 - is_small) * q_st
            q_st = q_st / q_st.sum(-1, keepdim=True)
        return sample_categorical(q_st)


# config based on https://github.com/dvruette/gidd/blob/f67acc961cc858cc69e8e6d91e7eb67a0a79151c/gidd/configs/gidd.yaml
# config based on https://github.com/dvruette/gidd/blob/f67acc961cc858cc69e8e6d91e7eb67a0a79151c/README.md
class GIDDHelper:
    loss_fn = None
    noise = None
    loss_weighting = "dynamic"  # [dynamic, clip, none]
    min_loss_weight = 0.0
    max_loss_weight = 2.0
    p_uniform = 0.0
    t_eps = 1e-4
    
    def __init__(self, vocab_size=None, mask_token_id=None):
        if vocab_size is not None:
            assert mask_token_id is not None
            self.__do_init__(vocab_size, mask_token_id)
    
    def __do_init__(self, vocab_size, mask_token_id):
        class ftok:
            def __init__(self, vocab_size, mask_token_id):
                self.vocab_size = vocab_size
                self.mask_token_id = mask_token_id
            
            def __len__(self):
                return self.vocab_size
           
        class fcfgLoss:
            def __init__(self, loss_weighting, min_loss_weight, max_loss_weight):
                self.loss_type = "gidd"
                self.loss_weighting = loss_weighting
                self.min_loss_weight = min_loss_weight
                self.max_loss_weight = max_loss_weight
                
        class fcfg:
            loss = fcfgLoss(self.loss_weighting, self.min_loss_weight, self.max_loss_weight)
        
        afcfg = fcfg()
        aftok = ftok(vocab_size, mask_token_id)
        noise = HybridDiffusion(aftok, p_uniform=self.p_uniform)
        self.loss_fn = GiddLoss(afcfg, aftok, noise)
        self.noise = noise
        self.sampler = GIDD_DenoisingStep(None, noise, aftok)
       
    def to(self, dtype=None, device=None):
        if next(self.noise.buffers()).device != device:
            self.noise.to(device=device)
            
        if next(self.noise.buffers()).dtype != dtype:
            self.noise.to(dtype=dtype)

        # if next(self.loss_fn.buffers()).device != device:
        #     self.loss_fn.to(device=device)
         
        # if next(self.loss_fn.buffers()).dtype != dtype:
        #     self.loss_fn.to(dtype=dtype)
        
    def get_loss(
        self, 
        logits: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask_id: Optional[int] = None,
        t: torch.Tensor = -1,
        *args, **kwargs,
    ):
        b, l, v = logits.shape
        if self.loss_fn is None:
            self.__do_init__(
                vocab_size=logits.shape[-1],
                mask_token_id=mask_id,
            )
        loss, elbo, _ = self.loss_fn(
            logits=logits, 
            input_ids=labels, 
            attention_mask=torch.ones_like(labels), 
            z_t=inputs, 
            t=t,
        )
        loss = loss.view(1, 1).repeat(*labels.shape)
        elbo = elbo.reshape(b, l)
        return loss, elbo
    
    @torch.no_grad()
    def sample_one_step(
        self,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        t: torch.Tensor = 0.1,
        s: torch.Tensor = 0.1,
        *args, **kwargs,
    ):
        b, l, v = logits.shape
        b, l = inputs.shape
        t = t.view(-1)
        s = s.view(-1)
        assert t.numel() == s.numel() and t.numel() == b
        return self.sampler(logits=logits, z_t=inputs, t=t, s=s)
        
