import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def datasetSpecificsExtractor(data_path, extentions, catagory_label):
    rows = []
    for category, label in catagory_label:
        category_dir = data_path / category
        for class_dir in sorted(category_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for video_path in class_dir.rglob("*"):
                if (
                    video_path.is_file()
                    and video_path.suffix.lower() in extentions
                ):
                    rows.append({
                        "video_path": str(video_path),
                        "category": category,
                        "class": class_dir.name,
                        "label": label
                    })

    df = pd.DataFrame(rows)

    print("Total videos:", len(df))
    print()
    print("Category counts:")
    print(df["category"].value_counts())
    print()
    print("Class counts:")
    print(df.groupby(["category", "class"]).size())

    df["source_class"] = (
        df["category"].astype(str) + "_" +
        df["class"].astype(str)
    )

    print(df["source_class"].value_counts())

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["source_class"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["source_class"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df))

    print("=== TRAIN ===")
    print(train_df["category"].value_counts())
    print(train_df.groupby("source_class").size())

    print("\n=== VALIDATION ===")
    print(val_df["category"].value_counts())
    print(val_df.groupby("source_class").size())

    print("\n=== TEST ===")
    print(test_df["category"].value_counts())
    print(test_df.groupby("source_class").size())
    return train_df, val_df, test_df


def sample_frame_indices(total_frames, num_frames):
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)

    if total_frames < num_frames:
        idx = np.arange(total_frames, dtype=np.int64)
        pad = np.full(num_frames - total_frames, total_frames - 1,dtype=np.int64)
        return np.concatenate([idx, pad])
    idx = np.linspace(0, total_frames - 1, num_frames)
    return np.clip(idx, 0, total_frames - 1).astype(np.int64)


def get_spatial_params(h, w, frame_size):
    short = min(h, w)
    scale = frame_size / short
    new_h = max(int(round(h * scale)), frame_size)
    new_w = max(int(round(w * scale)), frame_size)
    top = (new_h - frame_size) // 2
    left = (new_w - frame_size) // 2
    flip = False
    return (new_h, new_w, top, left, flip)