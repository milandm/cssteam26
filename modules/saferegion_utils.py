import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

from modules.saferegion import _SafeRegion


def collect_safe_regions_stats(model, run_dir=None, step=None, plot_graphs_to_wandb_log_freq=0):
    temp_model = model
    if type(model) is DistributedDataParallel:
        temp_model = model.module

    stats = dict()
    idx = 0

    fig = plt.figure()

    for module in temp_model.modules():
        if isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion):
            # stats[f'safe_regions_stats/{idx}/avg_running_mean'] = module.running_mean.mean().item()
            # stats[f'safe_regions_stats/{idx}/avg_running_var'] = module.running_var.mean().item()
            # stats[f'safe_regions_stats/{idx}/avg_running_min'] = module.running_min.mean().item()
            # stats[f'safe_regions_stats/{idx}/avg_running_max'] = module.running_max.mean().item()
            # stats[f'safe_regions_stats/{idx}/num_samples_tracked'] = module.num_samples_tracked.item()
            # stats[f'safe_regions_stats/{idx}/num_batches_tracked'] = module.num_batches_tracked.item()

            # bar plot
            # labels = [f"unit{i}" for i in range(module.num_features)]
            # values = module.running_mean.detach().cpu().numpy()
            # data = [[label, val] for (label, val) in zip(labels, values)]
            # table = wandb.Table(data=data, columns=["unit", "mean"])
            # stats[f"safe_regions_stats/{idx}"] = wandb.plot.bar(table, "unit", "mean", title="Safe regions mean per unit")

            layer_name = 'layer' + str(idx).zfill(3)

            # wandb plot
            xs = np.arange(module.num_features)
            ys_mean = module.running_mean.detach().cpu().numpy()
            ys_min = module.running_min.detach().cpu().numpy()
            ys_max = module.running_max.detach().cpu().numpy()

            if plot_graphs_to_wandb_log_freq != 0 and step % plot_graphs_to_wandb_log_freq == 0:
                plot = wandb.plot.line_series(xs=xs,
                                              ys=[ys_min, ys_mean, ys_max],
                                              keys=["running_min", "running_mean", "running_max"],
                                              title=layer_name,
                                              xname="step")
                stats[f"safe_regions_stats/{layer_name}/multi_line_plot"] = plot

            layer_dir = os.path.join(run_dir, 'plots', layer_name)
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)

            # matplotlib plot
            ax = fig.add_subplot()
            ax.plot(xs, ys_max, label="running_max", linestyle="-")
            ax.plot(xs, ys_mean, label="running_mean", linestyle="-")
            ax.plot(xs, ys_min, label="running_min", linestyle="-")
            ax.set_xlabel('neural unit')
            ax.set_ylabel('recorded value')
            ax.set_title(f'Safe region per neural unit at step {step}')
            max_bound = 20
            min_bound = 20
            ax.set_ylim((-min_bound, max_bound))
            ax.legend()
            fig_name = os.path.join(layer_dir, f"{str(step).zfill(7)}.png")
            fig.savefig(fig_name)
            plt.clf()
            idx += 1
    return stats


def collect_safe_regions_test_stats(model, boundary_definition):
    temp_model = model
    if type(model) is DistributedDataParallel:
        temp_model = model.module

    stats = dict()
    idx = 0

    for module in temp_model.modules():
        if isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion):
            layer_name = 'layer' + str(idx).zfill(3)

            # wandb plot
            num_units = module.num_features
            xs = np.arange(num_units)
            ys_mean = module.running_mean.detach().cpu().numpy()
            ys_std = np.sqrt(module.running_var.detach().cpu().numpy())
            ys_min = module.running_min.detach().cpu().numpy()
            ys_max = module.running_max.detach().cpu().numpy()
            sample_x_mean = module.last_x_mean.detach().cpu().numpy()
            sample_x_std = np.sqrt(module.last_x_var.detach().cpu().numpy())
            sample_x_max = module.last_x_max.detach().cpu().numpy()
            sample_x_min = module.last_x_min.detach().cpu().numpy()

            if boundary_definition == 'minmax':
                upper_bound = ys_max
                lower_bound = ys_min
            elif boundary_definition == 'gaussian':
                param = 1
                upper_bound = ys_mean + param * ys_std
                lower_bound = ys_mean - param * ys_std
            else:
                exit('unsupported boundary definition')

            x = module.last_x.detach().cpu()
            upper_bound_tensor = torch.from_numpy(upper_bound)
            upper_bound_tensor = upper_bound_tensor[None, :, None, None]
            lower_bound_tensor = torch.from_numpy(lower_bound)
            lower_bound_tensor = lower_bound_tensor[None, :, None, None]
            in_out = torch.logical_or(torch.gt(x, upper_bound_tensor), torch.lt(x, lower_bound_tensor))
            stats[f'saferegion/{layer_name}/out_of_bounds_percentage'] = (torch.sum(in_out) * 100) / x.numel()
            stats[f'saferegion/{layer_name}/distance_upper'] = torch.mean(torch.abs(upper_bound_tensor - x))
            stats[f'saferegion/{layer_name}/distance_lower'] = torch.mean(torch.abs(lower_bound_tensor - x))

            idx += 1

    return stats
