import os
import imageio

run_dir = '/home/andrijazz/storage/saferegions/log/wandb/run-20210818_202909-rjx1cbl9/files'

plots_dir = os.path.join(run_dir, 'plots')
for f in os.scandir(plots_dir):
    if not os.path.isdir(f):
        continue
    layer_name = f.name
    layer_path = os.path.join(plots_dir, layer_name)
    frames = sorted(os.listdir(layer_path))
    video_file = os.path.join(layer_path, layer_name + ".mp4")
    writer = imageio.get_writer(video_file, fps=2)
    for frame in frames:
        writer.append_data(imageio.imread(os.path.join(layer_path, frame)))
    writer.close()

    # if self.config['log_video']:
    #     log_dict[f'saferegions/{layer_name}/video'] = wandb.Video(video_file)


def log_video():
    # if last epoch log videos
    if self.config['train_num_epochs'] - 1 == info['epoch_idx']:
        plots_dir = os.path.join(wandb.run.dir, 'plots')
        for f in os.scandir(plots_dir):
            if not os.path.isdir(f):
                continue
            layer_name = f.name
            layer_path = os.path.join(plots_dir, layer_name)
            frames = sorted(os.listdir(layer_path))
            video_file = os.path.join(layer_path, layer_name + ".mp4")
            writer = imageio.get_writer(video_file, fps=2)
            for frame in frames:
                writer.append_data(imageio.imread(os.path.join(layer_path, frame)))
            writer.close()

            if self.config['log_video']:
                log_dict[f'saferegions/{layer_name}/video'] = wandb.Video(video_file)

