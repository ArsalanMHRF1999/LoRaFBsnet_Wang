import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Sampler
from decord import VideoReader, cpu, bridge
import numpy as np
import random as python_random
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
from Dataset.path_reader import get_spatial_params, sample_frame_indices

bridge.set_bridge("torch")
class HADSportArtDailyLifeDataset(Dataset):

    def __init__(self, df, num_frames=16, frame_size=128, grayScaleUsage=True, seed=42):
        self.seed = seed
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.grayScale = grayScaleUsage
        self.paths = self.df["video_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()

        # Normalization
        if grayScaleUsage:
            self.mean = torch.tensor([0.456]).view(1, 1, 1, 1)
            self.std = torch.tensor([0.224]).view(1, 1, 1, 1)
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        video_path = self.paths[idx]
        label = self.labels[idx]

        local_seed = self.seed + idx
        py_state = python_random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        python_random.seed(local_seed)
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)

        try:

            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            frame_idx = sample_frame_indices(
                total_frames=total_frames, num_frames=self.num_frames)

            frames = vr.get_batch(frame_idx)

            clip = frames.permute(0, 3, 1, 2).float() / 255.0

            if self.grayScale:
                clip = TF.rgb_to_grayscale(clip, num_output_channels=1)
                clip = clip.clamp(0.0, 1.0)

            T, C, H, W = clip.shape
            (new_h, new_w, top, left, flip) = get_spatial_params(
                H, W, self.frame_size)

            clip = F.interpolate(clip, size=(new_h, new_w),
                                 mode="bilinear", align_corners=False)

            clip = clip[:, :, top:top + self.frame_size,
                   left:left + self.frame_size]

            if flip:
                clip = clip.flip(-1)

            clip = (clip - self.mean) / self.std

        finally:
            python_random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)

        return clip, label


class FixedEpochSampler(Sampler):
    def __init__(self, dataset_size, epochs, seed=42):
        self.dataset_size = dataset_size
        self.epochs = epochs
        self.seed = seed
        self.epoch_orders = []
        for epoch in range(epochs):
            generator = torch.Generator()
            generator.manual_seed(seed + epoch)
            order = torch.randperm(dataset_size, generator=generator).tolist()
            self.epoch_orders.append(order)

        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        return iter(self.epoch_orders[self.current_epoch])

    def __len__(self):
        return self.dataset_size


class HADTestDataset(Dataset):
    def __init__(
            self,
            df,
            grayScaleUsage=True,
            num_frames=16,
            frame_size=128,
            shuffle_prob=0.8
    ):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.grayScale = grayScaleUsage
        self.paths = self.df["video_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.shuffle_prob = shuffle_prob

        # Normalization
        if self.grayScale:
            self.mean = torch.tensor([0.456]).view(1, 1, 1, 1)
            self.std = torch.tensor([0.224]).view(1, 1, 1, 1)
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def _sample_indices(self, total_frames):
        if total_frames <= 0:
            return np.zeros(self.num_frames, dtype=int)

        if total_frames < self.num_frames:
            idx = np.tile(np.arange(total_frames), int(np.ceil(self.num_frames / total_frames)))[:self.num_frames]
            return idx

        max_start = total_frames - self.num_frames
        start_idx = python_random.randint(0, max_start)
        idx = np.arange(start_idx, start_idx + self.num_frames)
        return idx

    def __getitem__(self, idx):
        rel_path = str(self.paths[idx]).lstrip("/")
        video_path = self.paths[idx]
        label = self.labels[idx]

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)

        frame_idx = self._sample_indices(total_frames)
        frames = vr.get_batch(frame_idx)

        if isinstance(frames, torch.Tensor):
            clip = frames.detach().cpu().float() / 255.0
        else:
            clip = torch.from_numpy(frames.asnumpy()).float() / 255.0

        clip = clip.permute(0, 3, 1, 2)
        T, C, H, W = clip.shape

        clip = F.interpolate(
            clip,
            size=(self.frame_size, self.frame_size),
            mode="bilinear",
            align_corners=False
        )
        clip = TF.rgb_to_grayscale(clip, num_output_channels=1)

        noise_prob = python_random.random()
        if noise_prob < 0.25:
            noise_var = 0.10
        elif 0.25 <= noise_prob < 0.5:
            noise_var = 0.15
        elif 0.5 <= noise_prob < 0.75:
            noise_var = 0.20
        else:
            noise_var = 0.25

        noise = torch.randn_like(clip) * (noise_var ** 0.5)
        clip = torch.clamp(clip + noise, 0.0, 1.0)

        is_occluded_prob = python_random.random()
        if is_occluded_prob < 1.0:
            occlusion_length = round(0.25 * self.num_frames)
            if occlusion_length > 0 and self.num_frames >= occlusion_length:
                start_occlusion = python_random.randint(0, self.num_frames - occlusion_length)
                clip[start_occlusion: start_occlusion + occlusion_length, :, :, :] = 127.0 / 255.0

        is_shuffled = python_random.random() < self.shuffle_prob
        if is_shuffled:
            shuffled_indices = torch.randperm(T).tolist()
            clip = clip[shuffled_indices]

        clip = (clip - self.mean) / self.std

        return clip, label

def show_had_clip_sample(dataset, idx, sample_idx=0, n_show=8, title_prefix="HAD Clip", grayScale=True):
    clip, label = dataset[sample_idx]
    T = clip.shape[0]
    frame_ids = np.linspace(0, T - 1, n_show).astype(int)

    if grayScale:
        mean = torch.tensor([0.456]).view(1, 1, 1)
        std = torch.tensor([0.224]).view(1, 1, 1)

    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(
        1, len(frame_ids), figsize=(3 * len(frame_ids), 3))
    if len(frame_ids) == 1:
        axes = [axes]

    for ax, t in zip(axes, frame_ids):

        img = (clip[t] * std + mean)

        img = img.clamp(0.0, 1.0)

        if grayScale:
            img = img[0].cpu().numpy()

            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)

        else:
            img = img.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img, vmin=0.0, vmax=1.0)

        ax.set_title(f"t={t}")

        ax.axis("off")

    class_name = idx[int(label)]

    plt.suptitle(
        f"{title_prefix} | "
        f"label={label} | "
        f"class={class_name}"
    )

    plt.tight_layout()
    plt.show()