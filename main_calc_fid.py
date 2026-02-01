import torch_fidelity
import torch
import torchvision
import os
import copy


def load_imagenet_val(file_list, root):
    with open(file_list, "r") as f:
        vlist = f.readlines()[1:]
    classes = sorted([v.split(",")[1].split(" ")[0] for v in vlist])
    class_to_idx = {n: i for i, n in enumerate(classes)}
    samples = [(os.path.join(root, v.split(",")[0] + ".JPEG"), class_to_idx[v.split(",")[1].split(" ")[0]]) for v in vlist]
    targets = [s[1] for s in samples]
    return classes, class_to_idx, samples, targets


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, train, hflip=True, **kwargs):
        if train:
            super().__init__(root=root, split="train")
        else:
            file_list = os.path.join(os.path.dirname(__file__), "vision_datasets/imagenet/LOC_val_solution.csv")
            (
                self.classes, self.class_to_idx, self.samples, self.targets
            ) = load_imagenet_val(file_list, root=os.path.join(root, "val"))
            self.loader = torchvision.datasets.folder.default_loader
            self.target_transform = None
            self.transform = None
            
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.RandomHorizontalFlip(0.5 if hflip else -1.0),
            torchvision.transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        # sample = sample * 2 - 1 # should be [-1, 1] based on taming-transformer
        # return {"input_ids": sample, "labels": target, "tovae": True}
        
        sample = (sample * 255).to(torch.uint8)
        assert (sample.min() >= 0) and (sample.max() <= 255)
        return sample


class Cifar10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train = True, hflip=True):
        super().__init__(root, train)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.RandomHorizontalFlip(0.5 if hflip else -1.0),
            torchvision.transforms.ToTensor(),
        ])
        
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample = (sample * 255).to(torch.uint8)
        assert (sample.min() >= 0) and (sample.max() <= 255)
        return sample
    

class StoreDataset(torch.utils.data.Dataset):
    def __init__(self, path=None, mode="uint8", seed=42, verbose=False):
        super().__init__()
        num_classes = 0
        data = torch.zeros((1, 3, 1, 1)).to(torch.uint8)
        if path is not None:
            if verbose:
                print(f"loading data ...")
            data: torch.Tensor = torch.load(path, map_location="cpu")
            if isinstance(data, dict):
                if verbose:
                    print(f"merging data ...")
                num_classes = len(data.keys())
                data = torch.cat(list(data.values()), dim=0)
            if verbose:
                print(f"converting data ...")
            data = self.convert_samples(data, mode=mode)
        assert data.dtype == torch.uint8
        self.num_classes = num_classes
        self.num_samples = len(data)
        if verbose:
            print(f"num_classes: {num_classes} num_samples: {len(data)}")
        
        # https://github.com/FoundationVision/LlamaGen/blob/ce98ec41803a74a90ce68c40ababa9eaeffeb4ec/tokenizer/validation/val_ddp.py#create_npz_from_sample_folder
        b, c, h, w = data.shape
        generator = torch.Generator("cpu").manual_seed(seed)
        shuffled_indices = torch.randperm(b, generator=generator)
        data = data[shuffled_indices].contiguous() # This is very important for IS(Inception Score) !!!
        # data = data.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        self.data = data
        
    def convert_samples(self, data: torch.Tensor, mode="uint8"):
        if mode == "uint8":
            if data.max() > 255:
                data = torch.where(data > 255, 128, data)
                assert data.min() >= 0 and data.max() <= 255
                data = data.to(torch.uint8)
        elif mode == "symfloat":
            assert data.min() >= -1.0 and data.max() <= 1.0
            data = ((data + 1.0) * 255 / 2.0).int()
            data = data.to(torch.uint8)
        elif mode == "float":
            assert data.min() >= 0.0 and data.max() <= 1.0
            data = (data * 255).int()
            data = data.to(torch.uint8)
        elif mode == "uint8f":
            data = data.clamp(min=0, max=255).int()
            data = data.to(torch.uint8)
        return data
                
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def prepare_imagenet_cache(path):
    dataset = ImageNet(path, train=True, hflip=False)
    print(dataset[0].shape)
    datasetx = copy.deepcopy(dataset)
    datasetx.samples = datasetx.samples[:1000]
    torch_fidelity.calculate_metrics(
        input1 = datasetx,
        input2 = dataset,
        input2_cache_name = "imagenet_train_256_cache.pt",
        isc=True,
        fid=True,
        kid=True,
        ppl=False,
    )
    dataset = ImageNet(path, train=False, hflip=False)
    print(dataset[0].shape)
    datasetx = copy.deepcopy(dataset)
    datasetx.samples = datasetx.samples[:1000]
    torch_fidelity.calculate_metrics(
        input1 = datasetx,
        input2 = dataset,
        input2_cache_name = "imagenet_val_256_cache.pt",
        isc=True,
        fid=True,
        kid=True,
        ppl=False,
    )


def prepare_cifar10_cache(path):
    dataset = Cifar10(path, train=True, hflip=False)
    print(dataset[0].shape)
    datasetx = copy.deepcopy(dataset)
    datasetx.data = datasetx.data[:1000]
    torch_fidelity.calculate_metrics(
        input1 = datasetx,
        input2 = dataset,
        input2_cache_name = "cifar10_train_32_cache.pt",
        isc=True,
        fid=True,
        kid=True,
        ppl=False,
    )
    dataset = Cifar10(path, train=False, hflip=False)
    print(dataset[0].shape)
    datasetx = copy.deepcopy(dataset)
    datasetx.data = datasetx.data[:1000]
    torch_fidelity.calculate_metrics(
        input1 = datasetx,
        input2 = dataset,
        input2_cache_name = "cifar10_val_32_cache.pt",
        isc=True,
        fid=True,
        kid=True,
        ppl=False,
    )


def calc_cifar10_fid(path, verbose=False):
    others = None
    if isinstance(path, tuple):
        path, others = path
    ds = StoreDataset(path, mode="uint8")
    out = torch_fidelity.calculate_metrics(
        input1 = ds,
        input2 = StoreDataset(),
        input2_cache_name = "cifar10_train_32_cache.pt",
        isc=True,
        fid=True,
        kid=True,
        ppl=False,
        verbose=verbose,
        # save_cpu_ram=True,
    )
    out.update(dict(num_classes=ds.num_classes, num_samples=ds.num_samples))
    return out, others


def calc_imagenet_fid(path, others=None, verbose=False):
    others = None
    if isinstance(path, tuple):
        path, others = path
    ds = StoreDataset(path, mode="uint8f")
    out = torch_fidelity.calculate_metrics(
        input1 = ds,
        input2 = StoreDataset(),
        input2_cache_name = "imagenet_val_256_cache.pt",
        isc=True,
        fid=True,
        kid=True,
        ppl=False,
        verbose=verbose,
        # save_cpu_ram=True,
    )
    out.update(dict(num_classes=ds.num_classes, num_samples=ds.num_samples))
    return out, others


if __name__ == "__main__":
    # please run this only once to prepare the cache files
    prepare_imagenet_cache("./data/ImageNet/ILSVRC/Data/CLS-LOC")
    prepare_cifar10_cache("./data/cifar10")

    # start calculating FID
    path = ""
    out, _ = calc_cifar10_fid(path)
    out, _ = calc_imagenet_fid(path)
    [print(f"{k}={v}", end="\t") for k, v in out.items()]
    print("\n")

