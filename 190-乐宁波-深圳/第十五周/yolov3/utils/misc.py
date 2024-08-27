import torch
from torch.utils.data import DataLoader, DistributedSampler
from thop import profile


class CollateFunc:
    def __call__(self, batch):
        images = torch.stack([sample[0] for sample in batch], 0)
        targets = [sample[1] for sample in batch]
        return images, targets


def build_dataloader(num_workers, dataset, batch_size, collate_fn=None):
    sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return dataloader


def compute_flops(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x,), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
