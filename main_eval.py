import json
import os
import fsspec
import torch
import tqdm


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if "hf" in config.algo.backbone:
        return diffusion_model(config, tokenizer=tokenizer).to("cuda")

    try:
        return diffusion_model.load_from_checkpoint(
            config.eval.checkpoint_path, tokenizer=tokenizer, config=config
        )
    except Exception as e:
        model = diffusion_model(config, tokenizer=tokenizer).to("cuda")
        state_dict = torch.load(config.eval.checkpoint_path, map_location="cpu")
        if "ema" in  state_dict:
            state_dict = state_dict["ema"]
            assert config.training.ema > 0
            model.ema.load_state_dict(state_dict)
            model.ema.copy_to(model.parameters())
        else:
            import traceback
            traceback.print_exc()
        return model


def _generate_samples(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Sample Eval.")
    vision_show_stride = (
        None
        if config.sampling.vision_show_stride < 0
        else config.sampling.vision_show_stride
    )

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    all_id_samples = []
    all_samples = []
    for _ in tqdm.tqdm(range(config.sampling.num_sample_batches)):
        model._eval_mode()
        samples = model.generate_samples(
            num_samples=config.loader.eval_batch_size,
            num_steps=config.sampling.steps,
            eps=1e-5,
            show_stride=vision_show_stride,
        )
        model._train_mode()
        if isinstance(samples, list):
            samples = (
                torch.stack(samples, dim=0)
                .permute(1, 0, 2)
                .contiguous()
                .flatten(0, 1)
            )
        all_id_samples.append(samples.cpu())
        model.metrics.record_entropy(samples)
        text_samples = model.tokenizer.batch_decode(samples)
        model.metrics.record_generative_perplexity(
          text_samples, config.model.length, model.device)
        all_samples.extend(list(text_samples))

    entropy = model.metrics.sample_entropy.compute().item()
    samples_path = config.eval.generated_samples_path
    print("Sample entropy:", entropy)
    print("Num samples:", len(all_samples))
    torch.save(torch.stack(all_id_samples), samples_path + ".pt")
    with fsspec.open(samples_path, "w") as f:
        json.dump(
            {
                "entropy": entropy,
                "num_samples": len(all_samples),
                "generated_seqs": all_samples,
            },
            f,
            indent=4,
        )

    model.metrics.record_generative_perplexity(
        all_samples, config.model.length, model.device
    )
    generative_ppl = model.metrics.gen_ppl.compute().item()
    print("Generative perplexity:", generative_ppl)
    with fsspec.open(samples_path, "w") as f:
        json.dump(
            {
                "generative_ppl": generative_ppl,
                "entropy": entropy,
                "num_samples": len(all_samples),
                "generated_seqs": all_samples,
            },
            f,
            indent=4,
        )
    print("Samples saved at:", samples_path)


def _generate_samples_image(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Sample Eval.")
    with_cond = config.training.guidance
    num_classes = config.data.num_classes if with_cond else 1
    assert config.sampling.num_sample_batches % num_classes == 0
    num_sample_batches = config.sampling.num_sample_batches // num_classes
    vision_show_stride = (
        None
        if config.sampling.vision_show_stride < 0
        else config.sampling.vision_show_stride
    )

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    
    all_samples_dict = dict()
    pbar = iter(tqdm.tqdm(config.sampling.num_sample_batches, desc="generate"))
    for cond in range(num_classes):
        if config.sampling.vision_target_class >= 0:
            if cond != config.sampling.vision_target_class:
                continue
        _cond = None
        if with_cond:
            _cond = cond * torch.ones(
                config.loader.eval_batch_size, device=model.device, dtype=torch.long
            )

        all_samples = []
        for _ in range(num_sample_batches):
            model._eval_mode()
            samples = model.generate_samples(
                num_samples=config.loader.eval_batch_size,
                cond=_cond,
                num_steps=config.sampling.steps,
                show_stride=vision_show_stride,
            )
            model._train_mode()

            xsamples = samples if isinstance(samples, list) else [samples]
            for samples in xsamples:
                if getattr(model.tokenizer, "dummy", False):
                    model.maybe_add_vae(model.device)
                    text_samples = model.image_tokenizer.batch_decode(samples)
                    text_samples = (text_samples + 1) * 255 / 2.0
                    text_samples = text_samples.cpu()
                else:
                    text_samples = model.tokenizer.batch_decode(samples)
                    text_samples = text_samples.cpu()
                all_samples.extend(list(text_samples))

            try:
                next(pbar)
            except:
                pass

        all_samples = torch.stack(all_samples)
        all_samples_dict.update({cond: all_samples})

    samples_path = os.path.join(
        config.eval.generated_samples_path, f"raw_images_ncls{num_classes}.pt"
    )
    os.makedirs(config.eval.generated_samples_path, exist_ok=True)
    torch.save(all_samples_dict, samples_path)
    print("Samples saved at:", samples_path)

