import torch
import torch.nn.functional as F
import trainer_base
import utils
from xdm_utils import XDMHelper
from gidd_utils import GIDDHelper


def apply_nucleus_prob(p_x0, nucleus_p):
    sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs <= nucleus_p
    top_p_mask[..., 0] = True
    nucleus_probs = sorted_probs * top_p_mask
    nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
    p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
    return p_x0


def sample_categorical(categorical_probs: torch.Tensor, force_fp64=False):
    if force_fp64:
        categorical_probs = categorical_probs.to(torch.float64)

    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


class BaseDiffusionWithCond(trainer_base.Diffusion):
    def _validate_configuration(self):
        """
        just copy and remove some codes
        """
        trainer_base.TrainerBase._validate_configuration(self)

        # copied from trainer_base.Diffusion
        # trainer_base.Diffusion._validate_configuration(self)
        assert self.config.sampling.noise_removal in {"none", "ancestral", "greedy"}
        assert self.loss_type in {"elbo", "low_var"}
        if self.config.sampling.noise_removal == "greedy":
            assert self.sampler != "analytic"
            # assert self.parameterization in {'mean', 'subs'} # remove this here

        # copied from trainer_base.UniformState
        # but all removed
        # trainer_base.UniformState._validate_configuration(self)
        # assert self.time_conditioning
        # assert self.parameterization == 'mean'
        # if self.config.algo.name != 'distillation':
        #   assert self.T == 0

    def __init_cfg__(self):
        self.is_causal = False
        self.with_time_branch = not self.config.algo.causal_attention
        self.sample_force_float64 = self.config.algo.sample_force_float64
        self.with_cond = self.config.training.guidance
        self.num_classes = self.config.data.get("num_classes", -100)
        if self.with_cond:
            assert self.num_classes > 0

        self.cfg_gamma = self.config.sampling.guidance_gamma
        self.with_inner_cfg = False
        if self.with_cond and self.cfg_gamma > 0.0 and self.config.training.guidance_inner:
            self.with_inner_cfg = True

        self.p_nucleus_afterwards = False

    def forward(self, xt, sigma, cond=None, pure_logits=False):
        # trainer_base.Diffusion would set sigma based on self.time_conditioning
        sigma = self._process_sigma(sigma)

        def get_model_output(xt, sigma, cond):
            if self.with_time_branch:
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    model_output = self.backbone(xt, sigma, cond)
            else:
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    model_output: torch.Tensor = self.backbone(
                        xt, None, cond, is_causal=self.is_causal
                    )
                    if not self.is_causal:
                        info = getattr(model_output, "__info__", {})
                        assert info["is_causal"] == False
                        assert info["t_cond"] == False
                        delattr(model_output, "__info__")
            return model_output

        model_output = get_model_output(xt, sigma, cond)

        if self.with_inner_cfg:
            nocond = torch.full_like(cond, fill_value=self.num_classes)
            model_output_nocond = get_model_output(xt, sigma, nocond)
            log_prob_cond = model_output.log_softmax(dim=-1)
            log_prob_nocond = model_output_nocond.log_softmax(dim=-1)
            log_prob = (
                1 + self.cfg_gamma
            ) * log_prob_cond - self.cfg_gamma * log_prob_nocond
            model_output = log_prob

        if pure_logits:
            return model_output
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    def nll(
        self,
        x0,
        output_tokens,
        cond=None,
        current_accumulation_step=None,
        train_mode=False,
    ):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)
        log_x_theta = self.forward(
            xt,
            sigma=sigma,
            cond=cond,
        )
        utils.print_nans(log_x_theta, "model_output")
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == "low_var",
        )

    def _loss(
        self,
        x0,
        valid_tokens,
        cond=None,
        current_accumulation_step=None,
        train_mode=False,
    ):
        input_tokens, output_tokens, valid_tokens = self._process_model_input(
            x0, valid_tokens
        )
        loss = self.nll(
            input_tokens,
            output_tokens,
            cond=cond,
            current_accumulation_step=current_accumulation_step,
            train_mode=train_mode,
        )
        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens

        return trainer_base.Loss(
            loss=token_nll, nlls=nlls, prior_loss=0.0, num_tokens=num_tokens
        )

    def _ancestral_update_core(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        raise NotImplementedError

    def _ancestral_update(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        q_xs = self._ancestral_update_core(
            x=x, t=t, dt=dt, p_x0=p_x0, cond=cond, noise_removal_step=noise_removal_step
        )

        if self.p_nucleus_afterwards and self.p_nucleus < 1.0:
            q_xs = apply_nucleus_prob(q_xs, nucleus_p=self.p_nucleus)

        _x = sample_categorical(q_xs, self.sample_force_float64)
        return None, _x

    @torch.no_grad()
    def generate_samples(
        self, num_samples, num_steps=None, eps=1e-5, cond=None, show_stride=None
    ):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        outs = [x]
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ancestral":
                _, x = self._ancestral_update(x=x, t=t, dt=dt, p_x0=None, cond=cond)
            elif self.sampler == "ancestral_cache":
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=p_x0_cache, cond=cond
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x=x, t=t, dt=dt, cond=cond)

            if (show_stride is not None) and ((i + 1) % show_stride == 0):
                outs.append(x)

        t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        if self.config.sampling.noise_removal == "ancestral":
            if self.sampler == "analytic":
                x = self._denoiser_update(x=x, t=t0, cond=cond)
            else:
                _, x = self._ancestral_update(
                    x=x,
                    t=t0,
                    dt=None,
                    p_x0=p_x0_cache,
                    cond=cond,
                    noise_removal_step=True,
                )
        elif self.config.sampling.noise_removal == "greedy":
            sigma = self._sigma_from_alphat(self.noise(t0)[1])
            x = self.forward(
                xt=x,
                sigma=sigma,
                cond=cond,
            ).argmax(dim=-1)

        if show_stride is not None:
            outs.append(x)
            return outs
        return x

    def maybe_add_vae(self, device):
        if not hasattr(self, "image_tokenizer"):
            assert self.tokenizer.dummy
            self.image_tokenizer: torch.nn.Module = self.tokenizer.tokenizer.to(
                device=device
            )
            self.image_tokenizer.eval()
            for n, p in self.image_tokenizer.named_parameters():
                p.requires_grad = False

    def image_preprocess(self, batch):
        self.maybe_add_vae(batch["input_ids"].device)
        with torch.no_grad():
            input_ids = self.image_tokenizer.batch_encode(batch["input_ids"])
            attention_mask = torch.ones_like(input_ids)

        b, l = input_ids.shape
        assert (b, l) == attention_mask.shape

        return input_ids, attention_mask

    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        cond = None
        if self.with_cond:
            cond = batch.get("labels", None)

        if batch.get("tovae", None) is not None:
            input_ids, attention_mask = self.image_preprocess(batch)
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        losses = self._loss(
            input_ids,
            attention_mask,
            cond=cond,
            current_accumulation_step=current_accumulation_step,
            train_mode=True,
        )
        self.metrics.update_train(losses.nlls, losses.prior_loss, losses.num_tokens)
        self.log(
            name="trainer/loss",
            value=losses.loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return losses.loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        cond = None
        if self.with_cond:
            cond = batch.get("labels", None)

        if batch.get("tovae", None) is not None:
            input_ids, attention_mask = self.image_preprocess(batch)
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        losses = self._loss(input_ids, attention_mask, cond=cond)
        self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)
        return losses.loss

    def optimizer_step(self, *args, **kwargs):
        if not hasattr(self, "__info__check__unused_params__"):
            print("\n--- Checking for unused parameters ---")
            found_unused = False
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"--> Unused parameter found: {name}")
                    found_unused = True
            if not found_unused:
                setattr(self, "__info__check__unused_params__", "pass")
        return trainer_base.Diffusion.optimizer_step(self, *args, **kwargs)


class MDLM(BaseDiffusionWithCond):
    def __init__(self, config, tokenizer):
        # NOTE: Ideally, we should do
        # vocab_size = len(tokenizer), so that we account
        # for the special tokens added in dataloader.py.
        # But we use tokenizer.vocab_size so as to to be
        # consistent with the prior checkpoints.
        vocab_size = tokenizer.vocab_size
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        self.subs_masking = config.algo.subs_masking
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()

        self._validate_configuration()
        BaseDiffusionWithCond.__init_cfg__(self)

    def _validate_configuration(self):
        ...
        # ancestral sampling isn't desirable because it's slow
        # assert self.sampler == 'ancestral_cache'

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        model_output[:, :, self.mask_index] += self.neg_infinity

        # Normalize the model_output such that x.exp() is
        # a probability distribution over vocab_size.
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          alpha_t: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        xt = torch.where(move_indices, self.mask_index, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        return self.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.device
        )

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)
        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(
            unmasked_score, -1, x[..., None], torch.zeros_like(unmasked_score[..., :1])
        )
        unmasked_score[:, :, self.mask_index] = -(log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
        model_output = masked_score * masked_indices + unmasked_score * (
            1 - masked_indices
        )
        return model_output.exp()

    def _ancestral_update_core(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2
        if p_x0 is None:
            p_x0 = self.forward(
                x,
                self._sigma_from_alphat(alpha_t),
                cond=cond,
            ).exp()
            if (not self.p_nucleus_afterwards) and self.p_nucleus < 1.0:
                p_x0 = apply_nucleus_prob(p_x0, nucleus_p=self.p_nucleus)

        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        q_xs = q_xs / (1 - alpha_t).unsqueeze(-1)
        return q_xs

    def _ancestral_update_remdm(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
            dt = t
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2

        assert p_x0 is None
        p_x0_cache = None
        if p_x0 is None:
            p_x0 = self.forward(
                x,
                self._sigma_from_alphat(alpha_t),
                cond=cond,
            ).exp()

        ##############
        # copied from remdm
        # cap: p=0.9 eta=0.008
        # conf: p=0.9
        # rescale: p=0.9 eta=0.015
        # loop: p=0.9 eta=0.02 t_on=0.55 t_off=0.05 alpha_on=0.9
        ##############
        assert p_x0.shape[0] == t.numel(), f"{p_x0.shape} {t.shape}"
        assert isinstance(dt, float) or (t.numel() == dt.numel()), f"{t} {dt}"
        t = t.view(-1)
        dt = dt.view(-1) if isinstance(dt, torch.Tensor) else dt

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        mdlm_sampler = self.config.sampling.mdlm_sampler
        if p_x0_cache is None:
            if self.p_nucleus < 1.0:
                p_x0 = apply_nucleus_prob(p_x0, nucleus_p=self.p_nucleus)

        def _sample_categorical(q_xs):
            return sample_categorical(q_xs, self.sample_force_float64)

        assert mdlm_sampler.sampler in [
            "mdlm",
            "forward-backward",
            "remdm-cap",
            "remdm-capa",
            "remdm-rescale",
            "remdm-loop",
        ]
        if mdlm_sampler.auto:
            if mdlm_sampler.sampler in ["remdm-rescale"]:
                mdlm_sampler.eta = 0.015
            elif mdlm_sampler.sampler in ["remdm-cap"]:
                mdlm_sampler.eta = 0.008
            elif mdlm_sampler.sampler in ["remdm-capa"]:
                mdlm_sampler.eta = 0.04  # from paper table 1 and 6
                mdlm_sampler.sampler = "remdm-cap"
            elif mdlm_sampler.sampler in ["remdm-loop"]:
                mdlm_sampler.eta = 0.02
                mdlm_sampler.t_on = 0.55
                mdlm_sampler.t_off = 0.05
                mdlm_sampler.alpha_on = 0.9

        if mdlm_sampler.sampler == "mdlm":
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
            _x = _sample_categorical(q_xs)
            copy_flag = (x != self.mask_index).to(x.dtype)
            xs = copy_flag * x + (1 - copy_flag) * _x
        elif mdlm_sampler.sampler == "forward-backward":
            alpha_t = (1 - move_chance_t)[0].item()
            alpha_s = (1 - move_chance_s)[0].item()
            if alpha_t > 0:
                sigma = (alpha_s - alpha_t) / alpha_t
            else:
                sigma = 1
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., self.mask_index] = sigma
            q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
            q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (
                1 - alpha_t
            )
            copy_flag = (x != self.mask_index).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = _sample_categorical(q_xs)
        elif mdlm_sampler.sampler == "remdm-cap":
            alpha_t = (1 - move_chance_t)[0].item()
            alpha_s = (1 - move_chance_s)[0].item()
            if alpha_t > 0:
                sigma = min(mdlm_sampler.eta, (1 - alpha_s) / alpha_t)
            else:
                sigma = mdlm_sampler.eta
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., self.mask_index] = sigma
            q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
            q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (
                1 - alpha_t
            )
            copy_flag = (x != self.mask_index).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = _sample_categorical(q_xs)
        elif mdlm_sampler.sampler == "remdm-rescale":
            alpha_t = (1 - move_chance_t)[0].item()
            alpha_s = (1 - move_chance_s)[0].item()
            if alpha_t > 0:
                sigma_max = min(1, (1 - alpha_s) / alpha_t)
            else:
                sigma_max = 1
            sigma = mdlm_sampler.eta * sigma_max
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., self.mask_index] = sigma
            q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
            q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (
                1 - alpha_t
            )
            copy_flag = (x != self.mask_index).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = _sample_categorical(q_xs)
        elif mdlm_sampler.sampler == "remdm-conf":
            conf = self.conf
            if conf is None:
                conf = (
                    -torch.ones_like(x, device=self.device).to(torch.bfloat16)
                    * torch.inf
                )
            alpha_t = (1 - move_chance_t)[0].item()
            alpha_s = (1 - move_chance_s)[0].item()
            if alpha_t > 0:
                sigma_max = min(1, (1 - alpha_s) / alpha_t)
            else:
                sigma_max = 1
            eta = conf.softmax(dim=-1)
            masked_flag = (x == self.mask_index).to(torch.bool)
            eta[masked_flag] = 0
            sigma = eta * sigma_max
            q_xs = p_x0 * (1 - sigma[:, :, None])
            q_xs[..., self.mask_index] = sigma
            q_xs_2 = p_x0 * (
                (alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t)
            )
            q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (
                1 - alpha_t
            )
            copy_flag = (x != self.mask_index).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = _sample_categorical(q_xs)
            # update conf
            unmask_mask = (x == self.mask_index) & (xs != self.mask_index)
            batch_indices = torch.arange(xs.shape[0])[:, None]
            feature_indices = torch.arange(xs.shape[1])
            conf_values = -p_x0[batch_indices, feature_indices, xs]
            conf[unmask_mask] = conf_values[unmask_mask]
            remask_mask = (x != self.mask_index) & (xs == self.mask_index)
            conf[remask_mask] = -torch.inf
            self.conf = conf
        elif mdlm_sampler.sampler == "remdm-loop":
            time = t[0].item()
            # compute alpha_t and alpha_s
            if time > mdlm_sampler.t_on:
                move_chance_t = (
                    1 - (1 - t) * mdlm_sampler.alpha_on / (1 - mdlm_sampler.t_on)
                )[:, None, None]
                move_chance_s = (
                    1 - (1 - t + dt) * mdlm_sampler.alpha_on / (1 - mdlm_sampler.t_on)
                )[:, None, None]
            elif time <= mdlm_sampler.t_off:
                move_chance_t = (t * (1 - mdlm_sampler.alpha_on) / mdlm_sampler.t_off)[
                    :, None, None
                ]
                move_chance_s = (
                    (t - dt) * (1 - mdlm_sampler.alpha_on) / mdlm_sampler.t_off
                )[:, None, None]
            else:
                move_chance_t, move_chance_s = None, None
            # use MDLM
            if time > mdlm_sampler.t_on or time <= mdlm_sampler.t_off:
                q_xs = p_x0 * (move_chance_t - move_chance_s)
                q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
                _x = _sample_categorical(q_xs)
                copy_flag = (x != self.mask_index).to(x.dtype)
                xs = copy_flag * x + (1 - copy_flag) * _x
            else:  # use ReMDM
                sigma = mdlm_sampler.eta
                q_xs = p_x0 * (1 - sigma)
                q_xs[..., self.mask_index] = sigma
                q_xs_2 = p_x0 * (
                    (mdlm_sampler.alpha_on - (1 - sigma) * mdlm_sampler.alpha_on)
                    / (1 - mdlm_sampler.alpha_on)
                )
                q_xs_2[..., self.mask_index] = (
                    1 - mdlm_sampler.alpha_on - mdlm_sampler.alpha_on * sigma
                ) / (1 - mdlm_sampler.alpha_on)
                copy_flag = (x != self.mask_index).to(torch.bool)
                q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
                xs = _sample_categorical(q_xs)
        else:
            raise NotImplementedError

        return p_x0, xs

    def _ancestral_update(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        if self.config.sampling.mdlm_sampler.sampler != "none":
            return self._ancestral_update_remdm(
                x=x,
                t=t,
                dt=dt,
                p_x0=p_x0,
                cond=cond,
                noise_removal_step=noise_removal_step,
            )

        _, _x = BaseDiffusionWithCond._ancestral_update(
            self, x=x, t=t, dt=dt, p_x0=p_x0, cond=cond, noise_removal_step=noise_removal_step
        )
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, dt):
        sigma_t = self._sigma_from_alphat(self.noise(t)[1])
        sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
        dsigma = sigma_t - sigma_s
        score = self._get_score(x, sigma_t)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        score = self._get_score(x, sigma)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[
            ..., None
        ]
        return edge


class UDLM(BaseDiffusionWithCond):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        BaseDiffusionWithCond.__init_cfg__(self)
        self.nll_type = self.config.algo.parameterization
        self.config.sampling.use_float64 = self.config.sampling.use_float64 and (
            not self.sample_force_float64
        )

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output.log_softmax(dim=-1)

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape
            (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)
        xt = torch.where(move_indices, uniform_tensor, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        return torch.randint(
            0, self.vocab_size, batch_dims, dtype=torch.int64, device=self.device
        )

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
          x: Either clean input `x0` (one-hot),
            or model's predicted `x_theta` of shape (B, L, V).
          xt: The noisy latent (as indices) of shape (B, L).
          alpha_s: Noise level at s of shape (B, [L | 1], 1).
          alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
          Posterior / approximate posterior of shape (B, L, V).
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(self.dtype).to(self.device)
        return (
            alpha_t * self.vocab_size * x * xt_one_hot
            + (alpha_ts - alpha_t) * xt_one_hot
            + d_alpha * x
            + (1 - alpha_ts) * (1 - alpha_s) / self.vocab_size
        ) / (
            alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None])
            + (1 - alpha_t)
        )

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        x_reconst = log_x_theta.exp()
        x_bar_theta = (
            self.vocab_size * alpha_t[:, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        )
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt - 1 / xbar_theta_xt)

        const = (1 - alpha_t) / (self.vocab_size * alpha_t + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = (
            (self.vocab_size - 1) * const * x_eq_xt - (1 / const) * x_neq_xt
        ) * const.log()
        term2_theta = -term2_coefs * (
            x_bar_theta.log().sum(-1) - self.vocab_size * xbar_theta_xt.log()
        )
        term2_theta = (
            term2_theta
            - self.vocab_size
            * alpha_t
            / (1 - alpha_t)
            * (xbar_theta_x.log() - xbar_theta_xt.log())
            * x_neq_xt
        )
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)
        assert diffusion_loss.ndim == 2
        return diffusion_loss

    def _sample_t(self, n, accum_step, sampling_eps=None, antithetic_sampling=None):
        sampling_eps = self.sampling_eps if sampling_eps is None else sampling_eps
        antithetic_sampling = (
            self.antithetic_sampling
            if antithetic_sampling is None
            else antithetic_sampling
        )
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - sampling_eps) * _eps_t + sampling_eps
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t

    def _ancestral_update_core(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2

        p_x0 = self.forward(
            x,
            self._sigma_from_alphat(alpha_t),
            cond=cond,
        ).exp()
        if (not self.p_nucleus_afterwards) and self.p_nucleus < 1.0:
            p_x0 = apply_nucleus_prob(p_x0, nucleus_p=self.p_nucleus)

        q_xs = self._compute_posterior(
            x=p_x0,
            xt=x,
            alpha_s=alpha_s,
            alpha_t=alpha_t,
        )

        return q_xs


class XDLM(BaseDiffusionWithCond):
    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        self.subs_masking = config.algo.subs_masking
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()

        # =================
        self._validate_configuration()
        self.__init_cfg__()

        # self.nll_type = self.parameterization
        self.k1 = config.algo.k1
        self.sample_start_k1_zero = config.algo.sample_start_k1_zero
        self.forward_pure_logits = True
        self.gradient_align_version = self.config.algo.gradient_align_version

    def _process_model_output(self, model_output, xt, sigma):
        if self.forward_pure_logits:
            return model_output
        raise NotImplementedError

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape
            (batch_size, 1).
        """
        if self.ignore_bos:
            xt = x.clone()
            xt[:, 1:] = XDMHelper.forward_process(
                batch=x[:, 1:],
                alpha_t=alpha_t,
                k1=self.k1,
                mask_id=self.mask_index,
                vocab_size=self.vocab_size,
                generator=None,
            )
        else:
            xt = XDMHelper.forward_process(
                batch=x,
                alpha_t=alpha_t,
                k1=self.k1,
                mask_id=self.mask_index,
                vocab_size=self.vocab_size,
                generator=None,
            )
        return xt

    def prior_sample(self, *batch_dims):
        b, l = batch_dims
        k1 = 0 if self.sample_start_k1_zero else self.k1
        return XDMHelper.prior_sample(
            b,
            l,
            k1=k1,
            mask_id=self.mask_index,
            vocab_size=self.vocab_size,
            device=self.device,
            generator=None,
        )

    # this is for aligned training and val ppl
    def nll(
        self,
        x0,
        output_tokens,
        cond=None,
        current_accumulation_step=None,
        train_mode=False,
    ):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)
        logits = self.forward(
            xt,
            sigma=sigma,
            cond=cond,
        )
        utils.print_nans(logits, "model_output")
        return XDMHelper.get_kl(
            logits=logits,
            inputs=xt,
            labels=x0,
            mask_id=self.mask_index,
            k1=self.k1,
            alpha_t=alpha_t,
            alpha_s=alpha_t,
            delta_alpha_scale=-dalpha_t,
        )

    def _ancestral_update_core(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2

        logits = None
        probs = None
        assert p_x0 is None
        if p_x0 is None:
            p_x0 = self.forward(x, self._sigma_from_alphat(alpha_t), cond=cond)
            logits = p_x0
            if (not self.p_nucleus_afterwards) and self.p_nucleus < 1.0:
                if self.mask_index is not None:
                    p_x0[:, :, self.mask_index] = self.neg_infinity
                p_x0 = p_x0.softmax(-1)
                p_x0 = apply_nucleus_prob(p_x0, nucleus_p=self.p_nucleus)
                probs = p_x0
                logits = None

        # p_x0[:, :, self.mask_index] = torch.finfo(p_x0.dtype).min
        # p_x0 = p_x0.softmax(dim=-1)

        # q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        # q_xs[:, :, self.mask_index] = 1 - alpha_s

        # not using float64 by default to align with mdlm
        # if self.sample_force_float64:
        #   p_x0 = p_x0.to(torch.float64)

        q_xs = XDMHelper.sample_one_step(
            logits=logits,
            inputs=x,
            k1=self.k1,
            mask_id=self.mask_index,
            alpha_t=alpha_t,
            alpha_s=alpha_s,
            probs=probs,
        )
        return q_xs


class GIDD(XDLM):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        assert not self.time_conditioning
        self.gidd_helper = GIDDHelper(self.vocab_size, self.mask_index)
        self.__parameter_state_info__ = None

    def gidd_helper_to_device(self):
        if self.__parameter_state_info__ is None:
            _format = next(self.backbone.parameters())
            self.__parameter_state_info__ = dict(
                _format=dict(
                    device=_format.device,
                    dtype=_format.dtype,
                )
            )
        _format = self.__parameter_state_info__["_format"]
        self.gidd_helper.to(**_format)

    def nll(
        self,
        x0,
        output_tokens,
        cond=None,
        current_accumulation_step=None,
        train_mode=False,
    ):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        self.gidd_helper_to_device()

        xt = self.gidd_helper.noise.sample_zt(x0, t)
        logits = self.forward(
            xt,
            sigma=torch.zeros_like(t.unsqueeze(-1)),
            cond=cond,
        )
        utils.print_nans(logits, "model_output")
        loss, elbo = self.gidd_helper.get_loss(
            logits=logits,
            inputs=xt,
            labels=x0,
            mask_id=self.mask_index,
            t=t,
        )
        if not train_mode:
            return elbo
        return loss

    def prior_sample(self, *batch_dims):
        self.gidd_helper_to_device()
        return self.gidd_helper.noise.sample_prior(batch_dims)

    def _ancestral_update_core(
        self, x, t, dt, p_x0=None, cond=None, noise_removal_step=False
    ):
        if noise_removal_step:
            s = torch.full_like(t, fill_value=1e-8)
        else:
            s = t - dt
        if p_x0 is None:
            p_x0 = self.forward(
                x,
                sigma=torch.zeros_like(t.unsqueeze(-1) if t.ndim == 1 else t),
                cond=cond,
            )

            if (not self.p_nucleus_afterwards) and self.p_nucleus < 1.0:
                if self.mask_index is not None:
                    p_x0[:, :, self.mask_index] = self.neg_infinity
                p_x0 = p_x0.softmax(-1)
                p_x0 = apply_nucleus_prob(p_x0, nucleus_p=self.p_nucleus)
                p_x0 = p_x0.log()

        q_xs = self.gidd_helper.sample_one_step(
            logits=p_x0,
            inputs=x,
            t=t,
            s=s,
        )

        return q_xs


