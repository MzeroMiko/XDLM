# Training

## Train on OWT
```bash
HF_DATASETS_CACHE=".cache/datasets";
METHODS="algo=xdlm algo.k1=0.1 wandb.name=xdlm-k01-1M-owt";
OTHERS="";
# OTHERS="trainer.log_every_n_steps=1 training.finetune_path=tmp.ckpst";
python -u -m main loader.batch_size=16 loader.eval_batch_size=16 model=small data=openwebtext-split model.length=1024  +wandb.offline=True data.cache_dir=$HF_DATASETS_CACHE ${METHODS} ${OTHERS};
```

***Methods for models***  
* MDLM: `algo=mdlm wandb.name=mdlm-1M-owt`
* GIDD: `algo=gidd wandb.name=gidd-1M-owt`
* XDLM-k0001: `algo=xdlm algo.k1=0.001 wandb.name=xdlm-k0001-1M-owt`
* XDLM-k01: `algo=xdlm algo.k1=0.1 wandb.name=xdlm-k01-1M-owt`
* XDLM-k05: `algo=xdlm algo.k1=0.5 wandb.name=xdlm-k05-1M-owt`
* XDLM-k09: `algo=xdlm algo.k1=0.9 wandb.name=xdlm-k09-1M-owt`
* UDLM: `algo=udlm wandb.name=udlm-1M-owt`


## Train on LM1B
```bash
HF_DATASETS_CACHE=".cache/datasets";
METHODS="algo=xdlm algo.k1=0.1 wandb.name=xdlm-k01-1M-lm1b-wrap";
OTHERS="";
# OTHERS="trainer.log_every_n_steps=1 checkpointing.resume_ckpt_path=tmp.ckpt";
python -u -m main loader.batch_size=64 loader.eval_batch_size=32 model=small data=lm1b-wrap model.length=128 +wandb.offline=True data.cache_dir=$HF_DATASETS_CACHE ${METHODS} ${OTHERS};
```

## Train on ImageNet-1K
```bash
IMNET="/htmp/ImageNet/ILSVRC/Data/CLS-LOC";
TOKENIZER_CKPT="$HF_HOME/others/LLamaGen/vq_ds16_c2i.pt";
METHODS="algo=xdlm algo.k1=0.1 wandb.name=xdlm-k01-imnet-16k-cond";
OTHERS="";
python -u -m main is_vision=True loader.batch_size=64 loader.eval_batch_size=64 model=small model.length=256 data=imagenet data.train=$IMNET data.valid=$IMNET data.tokenizer_name_or_path=imagenet_16384 data.tokenizer_checkpoint=$TOKENIZER_CKPT trainer.max_steps=500000 +wandb.offline=True +trainer.check_val_every_n_epoch=null training.guidance=True ${METHODS} ${OTHERS};
```

## Train on CIFAR10
```bash
export CIFAR10="${HF_HOME}/datasets/cifar10";
METHODS="algo=xdlm algo.k1=0.1 wandb.name=xdlm-k01-cifar10-cond";
OTHERS="";
python -u -m main is_vision=True loader.batch_size=64 loader.eval_batch_size=64 model=unet model.length=3072 data=cifar10 data.train=$CIFAR10 data.valid=$CIFAR10 algo.backbone=unet optim.lr=2e-4 lr_scheduler.num_warmup_steps=5000 trainer.max_steps=300000 +wandb.offline=True +trainer.check_val_every_n_epoch=null training.guidance=True ${METHODS} ${OTHERS};
```

