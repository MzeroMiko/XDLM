# Preparations

## Prepare Environment
```bash
POSTFIX="-i https://mirrors.ustc.edu.cn/pypi/simple"
NAME="xdlm"
eval "$(conda shell.bash hook)";
conda tos accept;
conda create -n $NAME python==3.12 -y && conda activate $NAME;
which conda; which python; which pip;
pip install --resume-retries 999 datasets==2.15.0 einops==0.7.0 fsspec git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 notebook==7.1.1 nvitop==1.3.2 omegaconf==2.3.0 packaging==23.2 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 transformers==4.38.2 triton==2.2.0 torch==2.3.1 torchaudio==2.3.1 torchmetrics==1.6.1 torchvision==0.18.1 wandb timm ocifs hf_transfer huggingface-hub mauve-text==0.4.0 pytorch-image-generation-metrics==0.6.1 torch_fidelity==0.3.0 deepspeed==0.13.1 evaluate peft==0.10.0 accelerate==0.27.2 $POSTFIX;
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.7.4.post1+cu126torch2.3-cp312-cp312-linux_x86_64.whl;
```

## Prepare GPT2 and Bert Tokenizer
```bash
hf download gpt2
hf download bert-base-uncased
```

## Preapare Dataset for Language
```bash
python -c "import datasets; datasets.load_dataset('wikitext', name='wikitext-2-raw-v1')"
python -c "import datasets; datasets.load_dataset('wikitext', name='wikitext-103-raw-v1')"
python -c "import datasets; datasets.load_dataset('ptb_text_only')"
python -c "import datasets; datasets.load_dataset('scientific_papers', 'arxiv')"
python -c "import datasets; datasets.load_dataset('scientific_papers', 'pubmed')"
python -c "import datasets; datasets.load_dataset('ag_news')"
python -c "import datasets; datasets.load_dataset('lm1b')"
python -c "import datasets; datasets.load_dataset('openwebtext')"
mkdir -p $HF_DATASETS_CACHE/raw_data && cd $HF_DATASETS_CACHE/raw_data;
wget http://mattmahoney.net/dc/text8.zip;
wget https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl;
```

## Prepare Image Tokenizer

***VQVAE from LlamaGen***  
```bash
wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt
```

## Preapare Dataset for Image

***CIFAR10***  
use `python -c "import torchvision; torchvision.datasets.CIFAR10(root, train, download=True)"`

***ImageNet-1K***  
visit [kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=ILSVRC) to download ImageNet

