import os
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.audio import load_audio, mix_at_snr, extract_segment

# Category mapping for the 7 classes of the Military Audio Dataset
MAD_CATEGORIES = {
    0: "Communication",
    1: "Gunshot",
    2: "Footsteps",
    3: "Shelling",
    4: "Vehicle",
    5: "Helicopter",
    6: "Fighter"
}


def discover_mad_files(data_dir: str, csv_path: str = None) -> pd.DataFrame:
    """
    Discover audio files in MAD dataset either through CSV or recursive search.
    Returns DataFrame with columns ['path', 'label', 'category', 'group_id'].
    """
    records = []

    # Priority 1: Check CSV path
    if csv_path and os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        base_dir = os.path.dirname(os.path.abspath(csv_path))

        for _, row in df_csv.iterrows():
            rel_path = str(row["path"]).replace("\\", "/")
            full_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(full_path):
                # Try relative to data_dir
                full_path = os.path.join(data_dir, rel_path)

            if os.path.exists(full_path):
                label = int(row["label"])
                # Extract group ID from parent folder (e.g. '398' in 'training/398/0.wav')
                parts = rel_path.split("/")
                group_id = parts[-2] if len(parts) >= 2 else "default"
                records.append({
                    "path": full_path,
                    "label": label,
                    "category": MAD_CATEGORIES.get(label, f"Class_{label}"),
                    "group_id": group_id
                })

    # Priority 2: Recursive file discovery if CSV is unavailable or empty
    if not records and os.path.exists(data_dir):
        print(f"[*] Discovering WAV files recursively in {data_dir}...")
        wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        for p in wav_files:
            p_norm = p.replace("\\", "/")
            parts = p_norm.split("/")
            group_id = parts[-2] if len(parts) >= 2 else "default"
            
            # Infer label from folder or category name if present
            label = 0 if "communication" in p_norm.lower() else 1
            for k, cat_name in MAD_CATEGORIES.items():
                if cat_name.lower() in p_norm.lower():
                    label = k
                    break

            records.append({
                "path": p,
                "label": label,
                "category": MAD_CATEGORIES.get(label, f"Class_{label}"),
                "group_id": group_id
            })

    if not records:
        raise RuntimeError(f"No audio files found in data_dir='{data_dir}' or csv_path='{csv_path}'")

    return pd.DataFrame(records)


def split_mad_dataset(df: pd.DataFrame, val_split: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and validation sets grouped by 'group_id' (recording/folder)
    to prevent data leakage between training and validation.
    """
    rng = random.Random(seed)
    groups = df["group_id"].unique().tolist()
    rng.shuffle(groups)

    n_val = int(len(groups) * val_split)
    val_groups = set(groups[:n_val])
    train_groups = set(groups[n_val:])

    train_df = df[df["group_id"].isin(train_groups)].reset_index(drop=True)
    val_df = df[df["group_id"].isin(val_groups)].reset_index(drop=True)

    # Ensure both splits have clean speech (label 0) and noise (label > 0)
    if len(train_df[train_df["label"] == 0]) == 0 or len(val_df[val_df["label"] == 0]) == 0:
        # Fallback to stratified random split if group split starved clean speech
        clean_df = df[df["label"] == 0].sample(frac=1.0, random_state=seed)
        noise_df = df[df["label"] != 0].sample(frac=1.0, random_state=seed)

        n_clean_val = int(len(clean_df) * val_split)
        n_noise_val = int(len(noise_df) * val_split)

        val_df = pd.concat([clean_df.iloc[:n_clean_val], noise_df.iloc[:n_noise_val]]).reset_index(drop=True)
        train_df = pd.concat([clean_df.iloc[n_clean_val:], noise_df.iloc[n_noise_val:]]).reset_index(drop=True)

    return train_df, val_df


class MADDataset(Dataset):
    """
    PyTorch Dataset for MAD Speech Enhancement.
    Dynamically loads clean speech (Communication, label 0) and defence noise (labels 1-6),
    mixing them on-the-fly at random SNR levels.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        segment_seconds: float = 3.0,
        snr_levels: list[float] = [-10, -5, 0, 5, 10, 15],
        is_training: bool = True,
        fixed_noise_df: pd.DataFrame = None
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_seconds)
        self.snr_levels = snr_levels
        self.is_training = is_training

        # Separate clean speech and defence noise
        self.clean_df = df[df["label"] == 0].reset_index(drop=True)
        
        if fixed_noise_df is not None:
            self.noise_df = fixed_noise_df.reset_index(drop=True)
        else:
            self.noise_df = df[df["label"] != 0].reset_index(drop=True)

        if len(self.clean_df) == 0:
            raise ValueError("No clean speech samples (label 0) found in dataset split!")
        if len(self.noise_df) == 0:
            raise ValueError("No defence noise samples (labels 1-6) found in dataset split!")

    def __len__(self) -> int:
        return len(self.clean_df)

    def __getitem__(self, idx: int) -> dict:
        clean_row = self.clean_df.iloc[idx]
        clean_path = clean_row["path"]

        # Select noise: random during training, deterministic hash during eval
        if self.is_training:
            noise_idx = random.randint(0, len(self.noise_df) - 1)
            snr = float(random.choice(self.snr_levels))
        else:
            noise_idx = idx % len(self.noise_df)
            snr = float(self.snr_levels[idx % len(self.snr_levels)])

        noise_row = self.noise_df.iloc[noise_idx]
        noise_path = noise_row["path"]
        noise_label = int(noise_row["label"])
        noise_category = str(noise_row["category"])

        # Load audio on the fly
        clean_audio = load_audio(clean_path, target_sr=self.sample_rate)
        noise_audio = load_audio(noise_path, target_sr=self.sample_rate)

        # Extract fixed segment length
        clean_seg = extract_segment(clean_audio, self.segment_len)
        noise_seg = extract_segment(noise_audio, self.segment_len)

        # Mix dynamically at selected SNR
        clean_mixed, noisy_mixed = mix_at_snr(clean_seg, noise_seg, snr)

        return {
            "clean": torch.from_numpy(clean_mixed),
            "noisy": torch.from_numpy(noisy_mixed),
            "snr": torch.tensor(snr, dtype=torch.float32),
            "noise_label": torch.tensor(noise_label, dtype=torch.long),
            "noise_category": noise_category,
            "clean_path": clean_path,
            "noise_path": noise_path
        }


def create_dataloaders(
    train_csv: str = "training.csv",
    test_csv: str = "test.csv",
    data_dir: str = "./",
    sample_rate: int = 16000,
    segment_seconds: float = 3.0,
    snr_levels: list[float] = [-10, -5, 0, 5, 10, 15],
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create Train, Validation, and Test DataLoader instances.
    """
    train_df_raw = discover_mad_files(data_dir=data_dir, csv_path=train_csv)
    test_df_raw = discover_mad_files(data_dir=data_dir, csv_path=test_csv)

    train_df, val_df = split_mad_dataset(train_df_raw, val_split=val_split, seed=seed)

    print(f"[*] Dataset Split: Train Clean={len(train_df[train_df['label']==0])}, Train Noise={len(train_df[train_df['label']!=0])} | "
          f"Val Clean={len(val_df[val_df['label']==0])}, Val Noise={len(val_df[val_df['label']!=0])} | "
          f"Test Clean={len(test_df_raw[test_df_raw['label']==0])}, Test Noise={len(test_df_raw[test_df_raw['label']!=0])}")

    train_dataset = MADDataset(
        train_df,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        snr_levels=snr_levels,
        is_training=True
    )

    val_dataset = MADDataset(
        val_df,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        snr_levels=snr_levels,
        is_training=False
    )

    test_dataset = MADDataset(
        test_df_raw,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        snr_levels=snr_levels,
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
