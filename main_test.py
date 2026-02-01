import torch
import time
import gc
import contextlib
import os
import hydra
import lightning as L
import omegaconf
import algo
import dataloader

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


DiffusionModels = dict(
    xdlm=algo.XDLM,
    udlm=algo.UDLM,
    mdlm=algo.MDLM,
    gidd=algo.GIDD,
)


def t_time(func: callable, warmup=50, times=500, device=None, batch=1, with_grad=False):
    ctx = torch.no_grad() if not with_grad else contextlib.nullcontext()

    for _ in range(warmup):
        with ctx:
            func()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem0 = torch.cuda.memory_allocated(device) / 1024 / 1024
    start = time.time()
    with ctx:
        for _ in range(times):
            func()
    torch.cuda.synchronize()
    interval = time.time() - start
    gc.collect()
    torch.cuda.empty_cache()
    mem1 = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print(
        f"{interval / times * 1000} ms/it {times * batch / interval} it/sec, {mem1:.3f} - {mem0:.3f} = {mem1 - mem0:.3f} MB"
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    L.seed_everything(config.seed)
    tokenizer = dataloader.get_tokenizer(config)

    config.algo.causal_attention = False
    config.algo.time_conditioning = False

    diffusion_model = DiffusionModels.get(config.algo.name, None)
    if diffusion_model is None:
        raise ValueError(f"Invalid algorithm name: {config.algo.name}")
    model = diffusion_model(config, tokenizer=tokenizer).to("cuda")

    batch_size = 32
    seq_length = 1024
    sampling_steps = 32

    data = torch.randint(
        low=0,
        high=12345,
        size=(batch_size, seq_length),
        device="cuda",
        dtype=torch.long,
    )
    valid_tokens = torch.ones_like(data, dtype=torch.bool)

    def test_forward():
        with torch.no_grad():
            loss = model._loss(data, valid_tokens).loss

    def test_forward_backward():
        loss = model._loss(data, valid_tokens).loss
        loss.backward()

    def test_generate_sample():
        # model._eval_mode()
        samples = model.generate_samples(
            num_samples=batch_size,
            num_steps=sampling_steps,
            eps=1e-5,
        )
        # model._train_mode()

    t_time(test_forward, batch=batch_size, warmup=10, times=100)
    t_time(test_forward_backward, batch=batch_size, with_grad=True, warmup=10, times=100)
    t_time(test_generate_sample, batch=batch_size, warmup=1, times=10)


if __name__ == "__main__":
    main()

