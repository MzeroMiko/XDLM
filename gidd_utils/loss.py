from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(torch.nn.Module, ABC):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.vocab_size = len(tokenizer)

    @abstractmethod
    def loss(self, logits, input_ids, attention_mask, z_t, t):
        raise NotImplementedError

    def forward(self, logits, input_ids, attention_mask, z_t, t, reduction="tokenmean"):
        loss, elbo, metrics = self.loss(logits, input_ids, attention_mask, z_t, t)

        if reduction == "tokenmean":
            num_tokens = attention_mask.numel()
            loss = loss.sum() / num_tokens
        else:  # reduction == "none"
            pass

        return loss, elbo, metrics


class GiddLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id
        self.loss_weighting = config.loss.loss_weighting
        self.min_loss_weight = config.loss.min_loss_weight
        self.max_loss_weight = config.loss.max_loss_weight
        assert self.max_loss_weight > 0, "max_loss_weight must be positive"

    def get_weights(self, t, z_t, input_ids):
        orig_dtype = t.dtype
        t = t.unsqueeze(-1).to(torch.float64)
        t1m = (1 - t)

        gamma = self.noise_schedule.log_gamma.exp()
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)
        B = self.noise_schedule.log_B.exp()

        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        c_t_prime = (gamma / 2) * (1 - 2 * t) / (t * t1m) * c_t

        is_mask = (z_t == self.mask_id).to(t.dtype)
        is_x = (z_t == input_ids).to(t.dtype)

        alpha_ratio = -1 / (1 - t) - c_t_prime / (1 + c_t)
        N = self.vocab_size - 1
        weight_on_x = (c_t + (1 - t) * c_t_prime) / N / ((1 - t)*(1 - t + c_t/N))
        weight_on_u = (c_t + (1 - t) * c_t_prime) / ((1 - t)*c_t)
        weight_on_m = 1 / ((1 - t)*t)

        elbo_weights = is_x * weight_on_x + is_mask * weight_on_m + (1 - is_x - is_mask) * weight_on_u

        loss_weights = elbo_weights.clone()
        if self.loss_weighting == "clip":
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "dynamic":
            log_snr = torch.sigmoid(-t).clip(-20, 20)  # not exactly the log-SNR, but close enough if C_t is close to 1
            x_scale = B / self.vocab_size * torch.exp(gamma / 2 * log_snr)
            loss_weights = (1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)

        return alpha_ratio.to(orig_dtype), elbo_weights.to(orig_dtype), loss_weights.to(orig_dtype)

    def loss(self, logits, input_ids, attention_mask, z_t, t):
        dtype = logits.dtype
        alpha_ratio, elbo_weights, ws = self.get_weights(t, z_t, input_ids)

        logits[..., self.mask_id] = torch.finfo(dtype).min

        x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
        x_hat = logits.softmax(-1).to(dtype)  # prevent automatic upcasting
        log_q_t = self.noise_schedule.probs_at_t(x, t).log_().clip_(min=-1e6)
        log_p_t = self.noise_schedule.probs_at_t(x_hat, t).log_().clip_(min=-1e6)

        kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

        log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_ratio = log_q_zt - log_p_zt

        is_loss = log_ratio.exp() - log_ratio - 1
        elbo = elbo_weights * (kl_loss + is_loss)

        loss = ws * (kl_loss + is_loss)

        metrics = {
            "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "is_loss": (ws * is_loss.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
        }

        return loss, elbo, metrics


class MDLMLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id
        self.neg_infty = -1e6

    def get_sigmas(self, t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    def loss(self, logits, input_ids, attention_mask, z_t, t):
        dsigma, sigma_t = self.get_sigmas(t)

        logits[..., self.mask_id] = self.neg_infty
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        mask_ids = (z_t == self.mask_id)
        logits[~mask_ids] = self.neg_infty
        logits = torch.where(~mask_ids.unsqueeze(-1).expand_as(logits), logits.scatter(-1, z_t.unsqueeze(-1), 0), logits)

        rec_loss = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")

        weights = dsigma.unsqueeze(-1) / torch.expm1(sigma_t).unsqueeze(-1)
        weights = weights * mask_ids.to(weights.dtype)

        elbo = weights * rec_loss

        metrics = {
            "rec_loss": (weights * rec_loss.detach() * attention_mask).sum() / attention_mask.sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
        }

        return elbo, elbo, metrics


def get_loss(config, tokenizer, noise_schedule):
    if config.loss.loss_type == "gidd":
        return GiddLoss(config, tokenizer, noise_schedule)
    elif config.loss.loss_type == "mdlm":
        return MDLMLoss(config, tokenizer, noise_schedule)
    elif config.loss.loss_type == "ar":
        return nn.CrossEntropyLoss(reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {config.loss.loss_type}")



