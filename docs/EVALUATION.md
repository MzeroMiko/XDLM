
# Evaluation

## Perplexity Evaluation 
```bash
DATASET_CACHE="";
CKPT="";
DATASET="openwebtext-split"; # "ag_news", "scientific_papers_pubmed", "scientific_papers_arxiv", "lambada", "wikitext2", "wikitext103", "ptb", "lm1b-gpt2", "openwebtext-split",
METHODS="algo=xdlm algo.k1=0.1";
RUNDIR="/tmp/${CKPT}/ds${DATASET}";
python -u -m main mode=ppl_eval seed=1 loader.batch_size=16 loader.eval_batch_size=16 loader.eval_global_batch_size=128 data.insert_valid_eos=False model=small model.length=1024 +wandb.offline=True data.cache_dir=$DATASET_CACHE eval.checkpoint_path=$CKPT data=$DATASET hydra.run.dir=${RUNDIR} ${METHODS};
```


## OWT PPL
```bash
CKPT="";
STEP=32
METHODS="algo=xdlm algo.k1=0.1";
RUNDIR="/tmp/tmp/s${STEP}/tmp.$(date +%s)";
python -u -m main mode=sample_eval seed=1 model=small data=openwebtext-split +wandb.offline=True eval.generated_samples_path=sample.json loader.eval_batch_size=25 sampling.num_sample_batches=200 eval.checkpoint_path=$CKPT sampling.steps=$STEP hydra.run.dir=${RUNDIR} ${METHODS};
```

## LM1B PPL
```bash
CKPT="";
STEP=32;
METHODS="algo=xdlm algo.k1=0.1";
RUNDIR="/tmp/tmp/s${STEP}/tmp.$(date +%s)";
python -u -m main mode=sample_eval seed=1 model=small model.length=128 data=lm1b-wrap +wandb.offline=True eval.generated_samples_path=sample.json loader.eval_batch_size=25 sampling.num_sample_batches=200 eval.checkpoint_path=$CKPT sampling.steps=$STEP hydra.run.dir=${RUNDIR} ${METHODS};
```

## ImageNet FID
```bash
CKPT="";
STEP=32;
METHODS="algo=xdlm algo.k1=0.1";
RUNDIR="/tmp/tmp/s${STEP}/tmp.$(date +%s)";
TOKENIZER_CKPT="$HF_HOME/others/LLamaGen/vq_ds16_c2i.pt";
python -u -m main mode=sample_image is_vision=True seed=1 model=small model.length=256 data=imagenet data.tokenizer_name_or_path=imagenet_16384 data.tokenizer_checkpoint=$TOKENIZER_CKPT +wandb.offline=True loader.eval_batch_size=50 sampling.num_sample_batches=1000 sampling.noise_removal=none training.guidance=True eval.checkpoint_path=$CKPT$ sampling.steps=$STEP hydra.run.dir=${RUNDIR} ${METHODS};
```


## CIFAR10 FID
```bash
CKPT="";
STEP=32;
METHODS="algo=xdlm algo.k1=0.1";
RUNDIR="/tmp/tmp/s${STEP}/tmp.$(date +%s)";
python -u -m main mode=sample_image is_vision=True seed=1 model=unet model.length=3072 data=cifar10 algo.backbone=unet +wandb.offline=True loader.eval_batch_size=100 sampling.num_sample_batches=500 sampling.noise_removal=none training.guidance=True eval.checkpoint_path=$CKPT sampling.steps=$STEP hydra.run.dir=${RUNDIR} ${METHODS};
```





