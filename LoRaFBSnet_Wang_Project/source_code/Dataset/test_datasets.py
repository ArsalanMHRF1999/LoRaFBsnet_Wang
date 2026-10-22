import torch
from Dataset.Dataloaders import HADTestDataset, HADSportArtDailyLifeDataset, FixedEpochSampler, show_had_clip_sample
from Dataset.path_reader import datasetSpecificsExtractor
from pathlib import Path
from torch.utils.data import DataLoader
NUM_FRAMES = 16
FRAME_SIZE = 128
BATCH_SIZE = 4
LABEL_MAP = {
    "sport": 0,
    "art": 1,
    "dailyLife": 2
}
IDX_TO_LABEL = {
    0: "sport",
    1: "art",
    2: "dailyLife"
}

DATA_ROOT = Path("./../sport_art_dailyLife")

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

catagory_label = [
    ("sport", 0), ("art", 1), ("dailyLife", 2)
]


train_df, val_df, test_df = datasetSpecificsExtractor(data_path=DATA_ROOT,
                                                      extentions=VIDEO_EXTENSIONS,
                                                      catagory_label=catagory_label)
NUM_EPOCHS = 100
train_ds = HADSportArtDailyLifeDataset(
    train_df,
    num_frames=16,
    frame_size=128,
    grayScaleUsage=True
)

val_ds = HADSportArtDailyLifeDataset(
    val_df,
    num_frames=16,
    frame_size=128,
    grayScaleUsage=True
)

test_ds = HADTestDataset(
    test_df,
    grayScaleUsage=True,
    num_frames=16,
    frame_size=128,
)

fixed_train_sampler = FixedEpochSampler(dataset_size=len(train_ds), epochs=NUM_EPOCHS, seed=42)
fixed_train_video_loader = DataLoader(
    train_ds,
    batch_size=8,
    sampler=fixed_train_sampler,
    num_workers=0,
    drop_last=False
)

fixed_val_video_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

test_video_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

show_had_clip_sample(
    fixed_train_video_loader.dataset,
    idx= IDX_TO_LABEL,
    sample_idx=3,
    n_show=8,
    title_prefix="HAD Sport-Art-DailyLife",
    grayScale=True
)
x1, y1 = train_ds[0]
x2, y2 = train_ds[0]

print("Label:", y1 == y2)
print("Shape:", x1.shape)
print("Identical:", torch.equal(x1, x2))
print(
    "Max Difference:",
    (x1 - x2).abs().max().item()
)