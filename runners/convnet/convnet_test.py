#!/usr/bin/env python

import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from datasets.dataset_factory import build_dataset
from models.model_factory import build_model
from modules.saferegion_utils import collect_safe_regions_test_stats
from utils.utils import is_debug_session, load_config_yml


def test(config, use_gpu=True):
    # init wandb
    wandb.init(project=os.getenv("PROJECT"), dir=os.getenv("LOG"), config=config)

    # setup dataset
    test_dataset = build_dataset(config['dataset_name'], config['test_dataset_path'], train=False, args=config)

    kwargs = {}
    if torch.cuda.is_available() and not is_debug_session():
        kwargs = {'num_workers': 2, 'pin_memory': True}

    # construct the model
    num_classes = len(test_dataset.classes)
    model = build_model(config['model_name'], num_classes)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(config['train_restore_file'])['models'][0])
    if use_gpu:
        model = model.cuda()
    model.eval()

    with torch.no_grad():
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config.get("test_batch_size"),
                                     **kwargs)
        for i, samples in enumerate(test_dataloader):
            images = samples[0]
            labels = samples[1]

            # Create non_blocking tensors for distributed training
            if use_gpu:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # forward
            if config['gaussian_noise']:
                r1 = -1
                r2 = 1
                images = (r2 - r1) * torch.rand_like(images) + r1

            outputs = model(images)
            _ = criterion(outputs, labels)
            _, predicted = torch.max(outputs, dim=1)

            # detach vars
            images = images.detach().cpu()
            labels = labels.detach().cpu().numpy()
            predicted = predicted.detach().cpu().numpy()

            # plot sample stats
            sample_idx = 0
            sample_img = images[sample_idx]
            gt_caption = test_dataset.classes[labels[sample_idx]]
            predicted_caption = test_dataset.classes[predicted[sample_idx]]
            caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)

            log_dict = {
                "test/sample": wandb.Image(sample_img, caption=caption),
            }
            safe_regions_stats = collect_safe_regions_test_stats(model, config['boundary_definition'])
            log_dict.update(safe_regions_stats)
            wandb.log(log_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    # parser.add_argument("--address", type=str, help="Ray address to use to connect to a cluster.")
    # parser.add_argument("--num-workers", type=int, default=1, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--use-fp16", action="store_true", default=False, help="Enables mixed precision training")
    # parser.add_argument("--num-cpus-per-worker", type=int, default=1, help="Sets number of cpus per worker")
    # parser.add_argument("--use-tqdm", action="store_true", default=False, help="Enables tqdm")
    args = parser.parse_args()
    # ray.init(address=args.address, local_mode=is_debug_session())
    config = load_config_yml(args.config)
    test(config, args.use_gpu)
