import torch
from typing import Optional, Union


class XDMHelper:
    def prior_sample(
        batch_size: int, 
        seq_length: int, 
        k1: float=0.1,
        mask_id: int = -1,
        vocab_size: int = -1,
        device: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        k1/v * pi_J + k2 pi_M
        """
        rand = torch.rand((batch_size, seq_length), device=device, generator=generator)
        transfer_id = torch.randint(
            0, vocab_size, (batch_size, seq_length), 
            dtype=torch.int64, device=device, generator=generator
        )
        if k1 < 1.0:
            transfer_id[rand < (1 - k1)] = mask_id
        return transfer_id

    @staticmethod
    def forward_process(
        batch: torch.Tensor, 
        alpha_t: torch.Tensor, 
        k1: float = 0.1,
        mask_id: int = -1,
        vocab_size: int = -1,
        generator: Optional[torch.Generator] = None,
    ):
        mix_ratio = k1
        batch_size, seq_length = batch.shape
        device = batch.device
        alpha_t = alpha_t.view(batch_size, 1)
        with_uniform_noise = mix_ratio > 0.0
        with_absorb_noise = mix_ratio < 1.0

        rand = torch.rand((batch_size, seq_length), device=device, generator=generator)
        transfer_id = torch.randint(
            0, vocab_size, (batch_size, seq_length), 
            dtype=torch.int64, device=device, generator=generator
        )
        to_keep = rand < alpha_t

        if with_absorb_noise and with_uniform_noise:
            assert mask_id > 0 and vocab_size > 0
            to_noise = (~to_keep) & (rand < alpha_t + mix_ratio * (1 - alpha_t))
            noisy_batch = torch.where(to_keep, batch, mask_id)
            noisy_batch = torch.where(to_noise, transfer_id, noisy_batch)
        elif with_absorb_noise:
            assert mask_id > 0
            noisy_batch = torch.where(to_keep, batch, mask_id)
        elif with_uniform_noise:
            assert vocab_size > 0
            noisy_batch = torch.where(to_keep, batch, transfer_id)
        return noisy_batch

    @staticmethod
    def get_probs(
        logits: torch.Tensor,
        mask_id: Optional[int] = None,
        probs: torch.Tensor = None,
    ):
        device = logits.device if logits is not None else probs.device
        if mask_id is not None:
            mask_id_index = torch.tensor([mask_id], dtype=torch.long, device=device)
            # set probs of mask to be 0
            if logits is None:
                probs = probs.index_fill(dim=-1, index=mask_id_index, value=0.0)
            else:
                logits = logits.index_fill(dim=-1, index=mask_id_index, value=torch.finfo(logits.dtype).min)

        if logits is not None:
            probs = logits.softmax(dim=-1)
        
        return probs

    @classmethod
    @torch.no_grad()
    def sample_one_step(
        cls,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        k1: float = 0.1,
        mask_id: Optional[int] = None,
        alpha_t: torch.Tensor = 0.1,
        alpha_s: torch.Tensor = 0.1,
        probs: torch.Tensor = None,
    ):
        """Performs one reverse diffusion step from time t to s (where t > s).

        Mathematical Logic:
            f_t(x, e) = a_t <e,x> + b_t (k1 / v + k2 * (e == m))
            out = f_s(x_th, e) / f_t(x_th, zt) * f_{t|s}(zt, e)

        Args:
            logits: Predicted token distributions of shape [batch, length, vocab].
            inputs: Noisy input tokens (z_t) of shape [batch, length].
            k1: mixing ratio of absorb and uniform noise.
            mask_id: Index of the [MASK] token in the vocabulary.
            alpha_t: Signal rate at current timestep t.
            alpha_s: Signal rate at previous timestep s (alpha_s >= alpha_t).
            probs: Optional pre-computed probability distribution [B, L, V].

        Returns:
            out_probs: The probability distribution for the next state z_s [B, L, V].
        """
        shape = logits.shape if logits is not None else probs.shape
        b, l, v = shape

        k2 = 1 - k1
        alpha_t = alpha_t.view(b, 1, 1)
        alpha_s = alpha_s.view(b, 1, 1)
        beta_t = 1 - alpha_t
        beta_s = 1 - alpha_s
        beta_ts = (alpha_s - alpha_t) / alpha_s

        assert (alpha_t <= alpha_s).all(), "we do not support alpha_s < alpha_t."
        assert (alpha_s > 1e-9).all(), "we do not support alpha_s too small."
        assert k1 == 1.0 or (mask_id is not None), "when k1 < 1.0, mask_id must be provided."

        zt_eq_m = (inputs == mask_id).unsqueeze_(-1)
        probs = cls.get_probs(logits, mask_id=mask_id, probs=probs)
        prob_zt = torch.gather(probs, dim=-1, index=inputs.unsqueeze(-1)) # (b, l, 1)

        # when current state zt is the mask token
        probs_from_mask = (alpha_s - alpha_t) * probs / beta_t + k1 * beta_ts * beta_s / beta_t / v
        if mask_id is not None:
            probs_mask_from_mask = (beta_s / beta_t - k1 * beta_ts * beta_s * (v - 1) / (beta_t * v))
            probs_from_mask[:, :, mask_id] = probs_mask_from_mask.squeeze(-1)

        # when current state zt is not the mask token
        vfprob_s = alpha_s * v * probs + beta_s * k1 # v * f_s(x_th,e)
        vfprob_t_zt = alpha_t * v * prob_zt + beta_t * k1 # v * f_t(x_th,zt)
        probs_from_token = k1 * beta_ts * vfprob_s / vfprob_t_zt / v # f_s(x_th,e) / f_t(x_th,zt) * b_(t|s) (k1 / v)
        probs_from_token_zt_add = 1 - k1 * beta_ts / vfprob_t_zt # f_s(x_th,e) / f_t(x_th,zt) * (e == zt) a_(t|s)
        probs_from_token = torch.scatter_add(probs_from_token, dim=-1, index=inputs.unsqueeze(-1), src=probs_from_token_zt_add)
        if mask_id is not None:
            probs_mask_from_token = (k1 * beta_ts * beta_s * (k1 / v + k2) / vfprob_t_zt)
            probs_from_token[:, :, mask_id] = probs_mask_from_token.squeeze(-1)

        out_probs = torch.where(zt_eq_m, probs_from_mask, probs_from_token)
        return out_probs

    @classmethod
    def get_kl(
        cls,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask_id: Optional[int] = None,
        k1: float = 0.1,
        alpha_t: torch.Tensor = 0.1,
        alpha_s: torch.Tensor = 0.1,
        delta_alpha_scale: Union[torch.Tensor, float] = 1.0,
        limit_case: bool = True,
        probs: torch.Tensor = None,
    ):
        """Computes the KL divergence between diffusion timesteps s and t.

        
        Mathematical formulation:
            D_KL = delta_alpha_scale * rdivt * h_t(x, zt, x_th)
            h_t(x, zt, x_th) = 
                party / rdivt
                - frac{1}{a_s} log frac{f_t(x,zt)} {f_t(x_th,zt)}
                + log frac{f_s(x,x)} {f_s(x_th,x)}
                + frac{k b_{s}}{a_{s}} partx

            r(e) = (k1 / v + k2 * (e == m))
            f_t(x, e) = a_t <e,x> + b_t (k1 / v + k2 * (e == m))

            delta_alpha_scale = beta_{t|s} a_s
            rdivt = frac{r(zt)} {f_t(x,zt)}
            partx = frac{1}{N} sum_{e in V} log frac{f_s(x,e)} {f_s(x_th,e)}
            party = frac{a_{t|s}}{b_{t|s} a_{s}} frac{f_s(x,zt)}{f_t(x,zt)} 
                    log frac{f_s(x,zt) f_t(x_th,zt)} {f_t(x,zt)f_s(x_th,zt)}
                  ~= (s -> t) frac{r(zt)}{f_t(x,zt)} frac{p_{x, zt} - p_{x_th,zt}}{f_t(x_th,zt)}
        
        Details of party:
            tmpx = frac{a_t f_s(x,zt)}{a_s f_t(x,zt)} 
            log frac{f_s(x,zt) f_t(x_th,zt)} {f_t(x,zt)f_s(x_th,zt)}
                = log(1 + frac{((zt == x) - p_zt) * r(e) * (a_s b_t - a_t b_s)}{f_t(x,zt)f_s(x_th,zt)})
                = log(1 + (a_s - a_t) * rdivt * ((zt == x) - p_zt) / f_s(x_th,zt))
                = log(1 + (a_s - a_t) * tmpy)
            party = tmpx * log(1 + (a_s - a_t) * tmpy) / (a_s - a_t)

        Details of partx:
            (e eq x) * frac{f_s(x,e)} {f_s(x_th,e)} = frac{v a_s + b_t k1} {v a_s * p_e + b_t k1}
            (e ne x or m) * frac{f_s(x,e)} {f_s(x_th,e)} = frac{b_t k1} {v a_s * p_e + b_t k1}
            (e eq m) * frac{f_s(x,e)} {f_s(x_th,e)} = 1 = frac{b_t k1} {v a_s * p_e + b_t k1}

            partx = frac{1}{N} sum_{e in V} log frac{f_s(x,e)} {f_s(x_th,e)}
                = frac{N-1}{N} log {b_t k1} + frac{1}{N} log {v a_s + b_t k1}
                - frac{1}{N} sum_{e in V} log {v a_s * p_e + b_t k1}

        Args:
            logits: Predicted token distributions of shape [batch, length, vocab].
            inputs: Noisy input tokens (z_t) of shape [batch, length].
            labels: Ground truth tokens (x_0) of shape [batch, length].
            mask_id: Index of the [MASK] token in the vocabulary.
            k1: mixing ratio of absorb and uniform noise.
            alpha_t: Signal rate at current timestep t.
            alpha_s: Signal rate at previous timestep s (alpha_s >= alpha_t).
            delta_alpha_scale: Scaling factor (alpha_s - alpha_t) to align KL with NELBO loss.
            probs: Optional pre-computed probability distribution [B, L, V].

        Returns:
            A tensor of shape [batch, length] containing the KL divergence values.

        """
        b, l, v = logits.shape
        b, l = inputs.shape
        b, l = labels.shape
        
        alpha_t = alpha_t.view(b, 1, 1)
        alpha_s = alpha_s.view(b, 1, 1)
        beta_t = 1 - alpha_t
        beta_s = 1 - alpha_s

        assert (alpha_t <= alpha_s).all(), "we do not support alpha_s < alpha_t."
        assert k1 == 1.0 or (mask_id is not None), "when k1 < 1.0, mask_id must be provided."

        if not isinstance(delta_alpha_scale, torch.Tensor):
            delta_alpha_scale = torch.full_like(alpha_t, fill_value=delta_alpha_scale)

        zt_eq_x = (inputs == labels).unsqueeze_(-1)
        zt_eq_m = (inputs == mask_id).unsqueeze_(-1)
        vratio = k1 + v * (1 - k1) * zt_eq_m
        probs = cls.get_probs(logits, mask_id=mask_id, probs=probs)

        prob_x0 = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)) # (b, l, 1)
        prob_zt = torch.gather(probs, dim=-1, index=inputs.unsqueeze(-1)) # (b, l, 1)
        vfprob_s_x0 = alpha_s * v * prob_x0 + beta_s * k1
        vfprob_s_zt = alpha_s * v * prob_zt + beta_s * vratio
        vfprob_t_zt = alpha_t * v * prob_zt + beta_t * vratio

        vfhard_s_x0 = alpha_s * v + beta_s * k1
        vfhard_s_zt = alpha_s * v * zt_eq_x + beta_s * vratio
        vfhard_t_zt = alpha_t * v * zt_eq_x + beta_t * vratio
        rdivt = torch.where(zt_eq_x, k1 / (v * alpha_t + beta_t * k1), 1 / beta_t) # r(z_t) / f_t(x,z_t)

        if limit_case or (alpha_s - alpha_t).min() < 1e-6:
            party = (v * (zt_eq_x.float() - prob_zt) / vfprob_s_zt) * (vratio / vfhard_t_zt)
        else:
            tmpx_zt_eq_x = (v * alpha_t * alpha_s + alpha_t * beta_s * k1) / (v * alpha_s * alpha_t + alpha_s * beta_t * k1)
            tmpx_zt_ne_x = (alpha_t * beta_s) / (alpha_s * beta_t)
            tmpx = torch.where(zt_eq_x, tmpx_zt_eq_x, tmpx_zt_ne_x)
            tmpy = (v * (zt_eq_x.float() - prob_zt) / vfprob_s_zt) * rdivt
            party = tmpx * torch.log1p(tmpy * (alpha_s - alpha_t)) / (alpha_s - alpha_t)

        if k1 > 0:
            partx_0 = - (v * alpha_s * probs + beta_s * k1).log().mean(dim=-1, keepdim=True)
            partx_1 = (beta_s * k1).log() * (v - 1) / v
            partx_2 = (v * alpha_s + beta_s * k1).log() / v 
            partx = partx_0 + partx_1 + partx_2

        kl = delta_alpha_scale.view(-1, 1) * (
            party
            - rdivt / alpha_s * (vfhard_t_zt.log() - vfprob_t_zt.log())
            + rdivt * (vfhard_s_x0.log() - vfprob_s_x0.log())
            + (k1 * beta_s * rdivt * partx / alpha_s if k1 > 0 else 0)
        ).view(b, l)

        return kl

    @classmethod
    def get_kl_simp(
        cls,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask_id: Optional[int] = None,
        k1: float = 0.1,
        alpha_t: torch.Tensor = 0.1,
        probs: torch.Tensor = None,
    ):
        """Computes the KL divergence between diffusion timesteps s and t.

        Comment:
            simple case of get_kl with alpha_s = alpha_t, delta_alpha_scale = 1

        Args:
            logits: Predicted token distributions of shape [batch, length, vocab].
            inputs: Noisy input tokens (z_t) of shape [batch, length].
            labels: Ground truth tokens (x_0) of shape [batch, length].
            mask_id: Index of the [MASK] token in the vocabulary.
            k1: mixing ratio of absorb and uniform noise.
            alpha_t: Signal rate at current timestep t.
            alpha_s: Signal rate at previous timestep s (alpha_s >= alpha_t).
            delta_alpha_scale: Scaling factor (alpha_s - alpha_t) to align KL with NELBO loss.
            probs: Optional pre-computed probability distribution [B, L, V].

        Returns:
            A tensor of shape [batch, length] containing the KL divergence values.
        """
        b, l, v = logits.shape
        b, l = inputs.shape
        b, l = labels.shape
        
        alpha_t = alpha_t.view(b, 1, 1)
        beta_t = 1 - alpha_t

        assert k1 == 1.0 or (mask_id is not None), "when k1 < 1.0, mask_id must be provided."

        zt_eq_x = (inputs == labels).unsqueeze_(-1)
        zt_eq_m = (inputs == mask_id).unsqueeze_(-1)
        vratio = (k1 + v * (1 - k1) * zt_eq_m)
        probs = cls.get_probs(logits, mask_id=mask_id, probs=probs)

        prob_x0 = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)) # (b, l, 1)
        prob_zt = torch.gather(probs, dim=-1, index=inputs.unsqueeze(-1)) # (b, l, 1)
        vfprob_t_x0 = alpha_t * v * prob_x0 + beta_t * k1
        vfprob_t_zt = alpha_t * v * prob_zt + beta_t * vratio
        vfhard_t_x0 = alpha_t * v + beta_t * k1
        vfhard_t_zt = alpha_t * v * zt_eq_x + beta_t * vratio
        rdivt = torch.where(zt_eq_x, k1 / (v * alpha_t + beta_t * k1), 1 / beta_t) # r(z_t) / f_t(x,z_t)

        party = (v * (zt_eq_x.float() - prob_zt) / vfprob_t_zt) * (vratio / vfhard_t_zt)

        if k1 > 0:
            partx_0 = - (v * alpha_t * probs + beta_t * k1).log().mean(dim=-1, keepdim=True)
            partx_1 = (beta_t * k1).log() * (v - 1) / v
            partx_2 = (v * alpha_t + beta_t * k1).log() / v 
            partx = partx_0 + partx_1 + partx_2

        kl = (
            party
            - rdivt / alpha_t * (vfhard_t_zt.log() - vfprob_t_zt.log())
            + rdivt * (vfhard_t_x0.log() - vfprob_t_x0.log())
            + (k1 * beta_t * rdivt * partx / alpha_t if k1 > 0 else 0)
        ).view(b, l)

        return kl

    @classmethod
    def test(cls):
        b, l, v = 2, 128, 1024
        k1 = 0.1
        mask_id = v - 1
        device = "cuda:0"
        device = "cpu"
        while True:
            eps=1e-3
            t = torch.rand((b,), device=device)
            p_mask = (1 - eps) * t + eps
            alpha_t = 1 - p_mask
            delta_alpha_scale = torch.ones_like(alpha_t)
            logits = torch.rand((b, l, v), device=device)
            labels = (torch.rand((b, l), device=device) * v).long()
            labels[labels == mask_id] = 0
            inputs = cls.forward_process(labels, alpha_t=alpha_t, k1=k1, mask_id=mask_id, vocab_size=v)
            loss = cls.get_kl(
                logits=logits, inputs=inputs, labels=labels, 
                mask_id=mask_id, k1=k1, 
                alpha_t=alpha_t, alpha_s=alpha_t, delta_alpha_scale=delta_alpha_scale,
            )
            loss2 = cls.get_kl_simp(
                logits=logits, inputs=inputs, labels=labels, 
                mask_id=mask_id, k1=k1, 
                alpha_t=alpha_t,
            )
            if (loss - loss2).abs().max() > 1e-5:
                breakpoint()
            if loss.min() < 0 or torch.isnan(loss).any() or torch.isinf(loss).any():
                print(loss)
            print(".", end="", flush=True)


if __name__ == "__main__":
    XDMHelper.test()


