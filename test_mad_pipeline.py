import os
import glob
import time
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import librosa

# Imports from project
from utils.audio import load_audio, mix_at_snr, calculate_rms
from utils.stft import STFTModule
from models.crn import CRN
from datasets.mad_dataset import discover_mad_files, split_mad_dataset, MADDataset, MAD_CATEGORIES
from losses.si_snr import calculate_si_snr, SISNRLoss
from losses.spectral_loss import SpectralLoss

def run_verification():
    print("=" * 70)
    print("      MAD DATASET PIPELINE VERIFICATION SUITE")
    print("=" * 70)
    
    checks = []
    
    # 1. Folder Structure & Path Discovery
    mad_root = os.path.abspath("..") # c:\anc head set\venv\MAD_dataset
    train_dir = os.path.join(mad_root, "training")
    test_dir = os.path.join(mad_root, "test")
    train_csv = os.path.join(mad_root, "training.csv")
    test_csv = os.path.join(mad_root, "test.csv")
    
    print(f"\n[1] Checking MAD Root Path: {mad_root}")
    print(f"  - Training dir exists: {os.path.exists(train_dir)} ({train_dir})")
    print(f"  - Testing dir exists:  {os.path.exists(test_dir)} ({test_dir})")
    print(f"  - Training CSV exists: {os.path.exists(train_csv)}")
    print(f"  - Testing CSV exists:  {os.path.exists(test_csv)}")
    
    # 2. Recursive WAV file discovery
    train_wavs = glob.glob(os.path.join(train_dir, "**", "*.wav"), recursive=True)
    test_wavs = glob.glob(os.path.join(test_dir, "**", "*.wav"), recursive=True)
    
    train_folders = set(os.path.dirname(p) for p in train_wavs)
    test_folders = set(os.path.dirname(p) for p in test_wavs)
    
    print(f"\n[2] WAV Files Discovery:")
    print(f"  - Training WAV files: {len(train_wavs):,} in {len(train_folders):,} subfolders")
    print(f"  - Testing WAV files:  {len(test_wavs):,} in {len(test_folders):,} subfolders")
    print(f"  - Total WAV files:    {len(train_wavs) + len(test_wavs):,}")
    
    if len(train_wavs) > 0 and len(test_wavs) > 0:
        checks.append(("WAV File Discovery", "PASS", f"{len(train_wavs)+len(test_wavs)} files discovered across {len(train_folders)+len(test_folders)} folders"))
    else:
        checks.append(("WAV File Discovery", "FAIL", "No WAV files found in Training/Testing folders"))

    print("\n  Example Training paths:")
    for p in train_wavs[:3]:
        print(f"    * {os.path.relpath(p, mad_root)}")
    print("  Example Testing paths:")
    for p in test_wavs[:3]:
        print(f"    * {os.path.relpath(p, mad_root)}")
        
    # 3. Audio Properties Analysis (Sample rate, channels, min/max/avg duration)
    sample_files = (train_wavs[:50] + test_wavs[:50]) if (train_wavs and test_wavs) else []
    sample_rates = set()
    channels = set()
    durations = []
    
    decode_success = True
    decode_error_count = 0
    
    for p in sample_files:
        try:
            info = sf.info(p)
            sample_rates.add(info.samplerate)
            channels.add(info.channels)
            durations.append(info.duration)
        except Exception as e:
            decode_success = False
            decode_error_count += 1
            
    if durations:
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = np.mean(durations)
        print(f"\n[3] Audio Format & Properties (sampled {len(sample_files)} real files):")
        print(f"  - Sample Rates: {sorted(list(sample_rates))} Hz")
        print(f"  - Channels: {sorted(list(channels))} ({'Mono' if channels == {1} else 'Stereo/Multi'})")
        print(f"  - Duration: min={min_dur:.2f}s, max={max_dur:.2f}s, avg={avg_dur:.2f}s")
        checks.append(("Audio Format & Stats", "PASS", f"SR={list(sample_rates)}, Channels={list(channels)}, Dur=[{min_dur:.1f}s - {max_dur:.1f}s]"))
    else:
        checks.append(("Audio Format & Stats", "FAIL", "Could not read audio file properties"))

    # 4. Decoding Test on Real Files
    real_decode_samples = train_wavs[:10]
    decoded_tensors = []
    for p in real_decode_samples:
        try:
            aud = load_audio(p, target_sr=16000)
            assert isinstance(aud, np.ndarray) and aud.dtype == np.float32
            assert not np.isnan(aud).any() and not np.isinf(aud).any()
            decoded_tensors.append(aud)
        except Exception as e:
            decode_success = False
            print(f"  [!] Decode failed on {p}: {e}")
            
    if decode_success and len(decoded_tensors) == len(real_decode_samples):
        checks.append(("Real Audio Decoding", "PASS", f"Decoded {len(decoded_tensors)} real MAD WAVs to 16kHz float32"))
    else:
        checks.append(("Real Audio Decoding", "FAIL", f"Failed decoding ({decode_error_count} errors)"))

    # 5. Data Leakage & Separation Check
    train_parent_ids = set(os.path.basename(os.path.dirname(p)) for p in train_wavs)
    test_parent_ids = set(os.path.basename(os.path.dirname(p)) for p in test_wavs)
    overlap = train_parent_ids.intersection(test_parent_ids)
    print(f"\n[4] Train / Test Leakage Check:")
    print(f"  - Unique Train Folders: {len(train_parent_ids)}")
    print(f"  - Unique Test Folders:  {len(test_parent_ids)}")
    print(f"  - Overlapping Folders:  {len(overlap)} ({overlap if overlap else 'None'})")
    if len(overlap) == 0:
        checks.append(("Train/Test Separation", "PASS", "Zero folder overlap between Training and Testing sets"))
    else:
        checks.append(("Train/Test Separation", "FAIL", f"Found {len(overlap)} overlapping folder IDs"))

    # 6. CSV & Label Mapping Analysis
    df_train_csv = pd.read_csv(train_csv) if os.path.exists(train_csv) else None
    print(f"\n[5] Inspecting CSV Metadata & Labels:")
    if df_train_csv is not None:
        print(f"  - CSV Columns: {list(df_train_csv.columns)}")
        labels_found = sorted(df_train_csv['label'].unique().tolist())
        print(f"  - Numeric Labels present: {labels_found}")
        print(f"  - Counts per label:")
        for l, count in df_train_csv['label'].value_counts().sort_index().items():
            cat_name = MAD_CATEGORIES.get(l, f"Label_{l}")
            print(f"      Label {l} ({cat_name:13s}): {count:5d} clips")
        checks.append(("CSV Label Parsing", "PASS", f"Found {len(labels_found)} numeric classes (0 to 6)"))
    else:
        checks.append(("CSV Label Parsing", "FAIL", "training.csv not found"))

    # 7. Dataset Loader & Dynamic SNR Mixing Test
    print(f"\n[6] Testing MAD Dataset Loader & Dynamic SNR Mixing:")
    try:
        train_df_discovered = discover_mad_files(data_dir=mad_root, csv_path=train_csv)
        train_split, val_split = split_mad_dataset(train_df_discovered, val_split=0.2, seed=42)
        
        dataset = MADDataset(
            train_split,
            sample_rate=16000,
            segment_seconds=3.0,
            snr_levels=[5.0], # Test fixed 5 dB
            is_training=True
        )
        sample = dataset[0]
        clean_t = sample["clean"]
        noisy_t = sample["noisy"]
        snr_t = sample["snr"]
        noise_cat = sample["noise_category"]
        
        print(f"  - Sample clean tensor shape: {clean_t.shape}, dtype: {clean_t.dtype}")
        print(f"  - Sample noisy tensor shape: {noisy_t.shape}, dtype: {noisy_t.dtype}")
        print(f"  - Target SNR: {snr_t.item():.1f} dB | Noise Category: {noise_cat}")
        print(f"  - Clean source: {os.path.relpath(sample['clean_path'], mad_root)}")
        print(f"  - Noise source: {os.path.relpath(sample['noise_path'], mad_root)}")
        
        # Verify measured SNR: SNR = 20 log10(RMS(clean) / RMS(noise))
        noise_extracted = (noisy_t - clean_t).numpy()
        rms_clean = calculate_rms(clean_t.numpy())
        rms_noise = calculate_rms(noise_extracted)
        measured_snr = 20.0 * np.log10(rms_clean / (rms_noise + 1e-8))
        print(f"  - Measured Mixed SNR: {measured_snr:.2f} dB (Target: 5.0 dB)")
        
        assert clean_t.shape == torch.Size([48000]), f"Unexpected shape {clean_t.shape}"
        assert abs(measured_snr - 5.0) < 0.5, f"Measured SNR {measured_snr} differs from target 5.0 dB"
        
        checks.append(("DataLoader & SNR Mixing", "PASS", f"Clean+Noise mixed at target 5.0 dB (measured {measured_snr:.2f} dB)"))
    except Exception as e:
        print(f"  [!] Dataset loader failed: {e}")
        checks.append(("DataLoader & SNR Mixing", "FAIL", str(e)))

    # 8. STFT Pipeline Test
    print(f"\n[7] Testing STFT / iSTFT on Mixed Real Audio:")
    try:
        stft_mod = STFTModule(n_fft=512, hop_length=128, win_length=512)
        spec = stft_mod.stft(noisy_t.unsqueeze(0))
        reconstructed = stft_mod.istft(spec, length=48000)
        
        stft_err = torch.mean(torch.abs(noisy_t.unsqueeze(0) - reconstructed)).item()
        print(f"  - STFT Complex shape: {spec.shape} (F=257, T_frames={spec.shape[-1]})")
        print(f"  - iSTFT Reconstructed shape: {reconstructed.shape}")
        print(f"  - STFT <-> iSTFT reconstruction MAE: {stft_err:.2e}")
        
        assert reconstructed.shape == noisy_t.unsqueeze(0).shape
        assert stft_err < 1e-5, f"STFT reconstruction error too large: {stft_err}"
        checks.append(("STFT & iSTFT Pipeline", "PASS", f"STFT shape [1, 257, {spec.shape[-1]}], perfect iSTFT inversion (MAE={stft_err:.1e})"))
    except Exception as e:
        print(f"  [!] STFT pipeline failed: {e}")
        checks.append(("STFT & iSTFT Pipeline", "FAIL", str(e)))

    # 9. Model Forward & Backward with Real MAD Audio Batch
    print(f"\n[8] Testing CRN Forward & Backward on Real MAD Audio Batch:")
    try:
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        
        real_clean = batch["clean"] # [4, 48000]
        real_noisy = batch["noisy"] # [4, 48000]
        
        crn = CRN(n_fft=512, hop_length=128, win_length=512, lstm_layers=2, hidden_size=256)
        crn.train()
        
        enhanced_audio, enh_mag = crn(real_noisy)
        print(f"  - Real Noisy Input Batch:   {real_noisy.shape}")
        print(f"  - CRN Enhanced Audio Batch: {enhanced_audio.shape}")
        print(f"  - CRN Enhanced Mag Batch:   {enh_mag.shape}")
        
        # Loss computation on real batch
        si_loss = SISNRLoss()(enhanced_audio, real_clean)
        target_spec = crn.stft_module.stft(real_clean)
        target_mag = crn.stft_module.to_magnitude(target_spec)
        spec_loss = SpectralLoss()(enh_mag, target_mag)
        
        total_loss = si_loss + 0.5 * spec_loss
        total_loss.backward()
        
        print(f"  - Computed Loss on Real MAD Data: {total_loss.item():.4f}")
        print(f"  - Gradient on enc1 conv weight norm: {crn.enc1.conv.weight.grad.norm().item():.4f}")
        
        assert enhanced_audio.shape == real_clean.shape
        assert not torch.isnan(total_loss)
        assert crn.enc1.conv.weight.grad is not None
        
        checks.append(("CRN Real Audio Forward/Backward", "PASS", f"Batch of 4 processed, Loss={total_loss.item():.2f}, Gradients computed"))
    except Exception as e:
        print(f"  [!] CRN test failed: {e}")
        checks.append(("CRN Real Audio Forward/Backward", "FAIL", str(e)))

    # Summary Table
    print("\n" + "=" * 75)
    print("                      PIPELINE VERIFICATION SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Check':<32} | {'Status':<8} | {'Result'}")
    print("-" * 75)
    all_pass = True
    for name, status, res in checks:
        print(f"{name:<32} | {status:<8} | {res}")
        if status != "PASS":
            all_pass = False
    print("=" * 75)
    
    print(f"\n>>> OVERALL DATASET STATUS: {'PASS' if all_pass else 'FAIL'} <<<\n")

if __name__ == "__main__":
    run_verification()
