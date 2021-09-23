import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms

from modules.saferegion import SafeRegion1d
from utils.utils import is_debug_session, load_config_yml, accuracy
from utils.viz_net import visualize_in_out

os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'


class MLP(nn.Module):
    def __init__(self, arch):
        super(MLP, self).__init__()
        assert arch is not None
        assert len(arch) > 2
        self.arch = arch
        self.net = nn.Sequential()
        n_in = arch.pop(0)
        n_out = arch.pop()
        prev = n_in
        for i, m in enumerate(arch):
            self.net.add_module(f'dense{i}', nn.Linear(prev, int(m)))
            self.net.add_module(f'dense{i}_saferegion', SafeRegion1d(int(m)))
            self.net.add_module(f'dense{i}_act', nn.ReLU())
            prev = int(m)
        self.net.add_module(f'dense_{len(arch)}', nn.Linear(prev, n_out))
        self.net.add_module(f'dense{n_out}_saferegion', SafeRegion1d(n_out))

    def forward(self, x):
        return self.net(x)


def _validate(model, val_loader, device):
    p = []
    gt = []
    for samples in val_loader:
        inputs = samples[0]
        batch_size = inputs.shape[0]
        in_dim = inputs.shape[2] * inputs.shape[3]
        inputs_vec = inputs.reshape((batch_size, in_dim))
        labels = samples[1]
        inputs_vec = inputs_vec.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs_vec)

        probability_outputs = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probability_outputs, dim=1)
        predicted_classes = predicted_classes.detach().cpu()
        labels = labels.detach().cpu()
        p.extend(predicted_classes.tolist())
        gt.extend(labels.tolist())

    p = np.array(p, dtype=np.int32)
    gt = np.array(gt, dtype=np.int32)
    acc = accuracy(p, gt)
    return acc


def train(config):
    # init wandb
    wandb.init(project='saferegions', config=config, dir=os.getenv('LOG'))
    model = MLP(config['arch'])
    model.train()

    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() and not is_debug_session() else {}
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(os.getenv('DATASETS'),
                             train=True,
                             download=True,
                             transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[len(dataset) - 100, 100])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['train_batch_size'],
                                               shuffle=True,
                                               **kwargs)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['val_batch_size'],
                                             shuffle=False,
                                             **kwargs)

    device = config['device']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    step = 0
    for epoch in range(config['num_epochs']):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            in_dim = data.shape[2] * data.shape[3]
            inputs_vec = data.reshape((data.shape[0], in_dim))
            optimizer.zero_grad()
            output = model(inputs_vec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            p = torch.softmax(output, dim=1)
            c = torch.argmax(p, dim=1)
            if step % config['train_log_freq'] == 0:
                # log scalars
                wandb.log({"train/loss": loss.item()}, step=step)
                # log random samples
                sample_idx = np.random.choice(config['train_batch_size'])
                predicted_caption = str(c[sample_idx].item())
                gt_caption = str(target[sample_idx].item())
                caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)
                sample_img = data[sample_idx].detach().cpu()
                wandb.log({"train/samples": wandb.Image(sample_img, caption=caption)}, step=step)
            if step % config['val_freq'] == 0:
                model.eval()
                with torch.no_grad():
                    model_name = '{}-{}.pth'.format(config['model'], step)
                    checkpoint_file = os.path.join(wandb.run.dir, model_name)
                    torch.save(model.state_dict(), checkpoint_file)
                    acc = _validate(model, val_loader, device)
                    wandb.log({"val/acc": acc}, step=step)
                model.train()

            step += 1

    with torch.no_grad():
        model_name = '{}-{}.pth'.format(config['model'], step)
        checkpoint_file = os.path.join(wandb.run.dir, model_name)
        torch.save(model.state_dict(), checkpoint_file)


def test(config):
    # init wandb
    wandb.init(project='saferegions', config=config, dir=os.getenv('LOG'))
    model = MLP(config['arch'])

    # load checkpoints
    if config['train_restore_file']:
        model.load_state_dict(torch.load(config['train_restore_file']))

    model.eval()

    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() and not is_debug_session() else {}
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.MNIST(os.getenv('DATASETS'),
                                  train=False,
                                  download=True,
                                  transform=transform)

    # hardcoding batch_size to 1
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              **kwargs)

    device = config['device']
    model = model.to(device)
    step = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        in_dim = data.shape[2] * data.shape[3]
        inputs_vec = data.reshape((data.shape[0], in_dim))
        # r1 = -10
        # r2 = 10
        # inputs_vec = (r1 - r2) * torch.rand([1, 784], device=device) + r2
        # noise = torch.rand(inputs_vec.shape, device=device)
        # inputs_vec = inputs_vec + noise
        output = model(inputs_vec)
        p = torch.softmax(output, dim=1)
        c = torch.argmax(p, dim=1)

        # log sample
        predicted_caption = str(c[0].item())
        gt_caption = str(target[0].item())
        caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)
        sample_img = data[0].detach().cpu()
        in_out_chart = visualize_in_out(model)
        log_dict = {
            "test/sample": wandb.Image(sample_img, caption=caption),
            "test/in_out": wandb.Image(in_out_chart)
        }
        wandb.log(log_dict, step=step)
        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool for experimenting with SafeRegion module')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/dev_config.yml', help='Path to yml config')
    args = parser.parse_args()
    config = load_config_yml(args.config)

    if args.train:
        train(config)
    else:
        test(config)
