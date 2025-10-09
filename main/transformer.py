import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

sys.path.append('/mnt/d/GIST/Project/Project 1/ecog_word')

from model.transformer import Transformer
from utils.data import main as data_main
from utils.plot import plot_mspec

from torchmetrics import MeanSquaredError, PearsonCorrCoef
from torchmetrics.text import CharErrorRate, WordErrorRate

from nemo.collections.tts.models import HifiGanModel
from transformers import AutoProcessor, HubertForCTC

import random 


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0


def val_transformer_model(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for ecog, mspec in val_loader:
            ecog, mspec = ecog.to(device), mspec.to(device)

            mspec_in = torch.zeros_like(mspec)
            mspec_in[:, 1:, :] = mspec[:, :-1, :]

            output, _ = model(ecog, mspec_in) 
            loss = criterion(output, mspec) 
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_transformer_model(model, train_loader, val_loader, device, epochs, optimizer, criterion, scheduler, early_stopping):
    train_losses, val_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        epoch_balance_loss = 0.0

        for idx, (ecog, mspec) in enumerate(train_loader):
            ecog, mspec = ecog.to(device), mspec.to(device)

            optimizer.zero_grad()

            mspec_in = torch.zeros_like(mspec)
            mspec_in[:, 1:, :] = mspec[:, :-1, :]

            output, load_balance_loss = model(ecog, mspec_in)
            primary_loss = criterion(output, mspec)

            total_balance_loss = sum(load_balance_loss) if isinstance(load_balance_loss, list) else load_balance_loss
            total_loss = primary_loss + total_balance_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += primary_loss.item()
            epoch_balance_loss += total_balance_loss.item()

            # if idx % 50 == 0:
            #     print(f"  Batch {idx}/{len(train_loader)} - Loss: {primary_loss.item():.4f}, Balance Loss: {total_balance_loss.item():.4f}")

        train_loss = epoch_loss / len(train_loader)
        avg_balance_loss = epoch_balance_loss / len(train_loader)
        val_loss = val_transformer_model(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Balance Loss: {avg_balance_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}")
        print("-" * 50)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    os.makedirs("outputs/output_transformer", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/output_transformer/transformer_model_16.pth")
    print("Model saved.")
    return model


def test_transformer_model(model, test_loader, device):
    model.eval()
    gen_mspec, target_mspec = [], []

    with torch.no_grad():
        for ecog, mspec in test_loader:
            ecog, mspec = ecog.to(device), mspec.to(device)

            mspec_in = torch.zeros_like(mspec)
            mspec_in[:, 1:, :] = mspec[:, :-1, :]

            output, _ = model(ecog, mspec_in)

            gen_mspec.append(output.cpu())
            target_mspec.append(mspec.cpu())
    
    gen_mspec = torch.cat(gen_mspec)
    target_mspec = torch.cat(target_mspec)

    # Compute evaluation metrics
    mse = MeanSquaredError()(gen_mspec, target_mspec).item()
    rmse = np.sqrt(mse)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Flatten for PCC calculation
    gen_mspec_flat = gen_mspec.flatten()
    target_mspec_flat = target_mspec.flatten()
    pcc = PearsonCorrCoef()(gen_mspec_flat, target_mspec_flat).item()
    print(f"Test PCC: {pcc:.4f}")

    return gen_mspec, target_mspec


def mspec2audio2text_hubert(mspec, hifigan, processor, model):
    """HuBERT-specific audio to text conversion"""
    try:
        device = next(hifigan.parameters()).device
        mspec_tensor = torch.tensor(mspec, dtype=torch.float32)

        if len(mspec_tensor.shape) == 2:
            if mspec_tensor.shape[1] == 80: 
                mspec_tensor = mspec_tensor.transpose(0, 1)
            mspec_tensor = mspec_tensor.unsqueeze(0)
        mspec_tensor = mspec_tensor.to(device)

        with torch.no_grad():
            audio = hifigan.convert_spectrogram_to_audio(spec=mspec_tensor)
        audio_np = audio.squeeze().cpu().numpy()

        # Resample to 16kHz for HuBERT
        audio_resample = librosa.resample(audio_np, orig_sr=22050, target_sr=16000)
        
        # Normalize audio
        audio_resample = audio_resample / np.max(np.abs(audio_resample) + 1e-8)
        
        # Process with HuBERT
        inputs = processor(audio_resample, return_tensors="pt", sampling_rate=16000).input_values.to(device)

        with torch.no_grad():
            logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        return audio_resample, transcription
    
    except Exception as e:
        print(f"[ERROR mspec->audio HuBERT] {e}")
        return np.zeros(16000), "[ERROR]"
    

class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, n_ffts=[1024, 2048, 512], hop_lengths=None, win_lengths=None):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths or [n // 4 for n in n_ffts]
        self.win_lengths = win_lengths or n_ffts
        
    def stft_loss(self, pred, target, n_fft, hop_length, win_length):
        # Convert mel-spec back to linear for STFT
        pred_stft = torch.stft(pred.flatten(1), n_fft=n_fft, hop_length=hop_length, 
                              win_length=win_length, return_complex=True)
        target_stft = torch.stft(target.flatten(1), n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length, return_complex=True)
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        
        # Phase loss
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        phase_loss = F.l1_loss(pred_phase, target_phase)
        
        return mag_loss + 0.1 * phase_loss
    
    def forward(self, pred, target):
        total_loss = 0
        for n_fft, hop_length, win_length in zip(self.n_ffts, self.hop_lengths, self.win_lengths):
            total_loss += self.stft_loss(pred, target, n_fft, hop_length, win_length)
        return total_loss / len(self.n_ffts)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = MultiScaleSpectralLoss()
        
    def forward(self, pred, target):
        # Reconstruction losses
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)

        # Spectral loss for better audio quality
        spectral = self.spectral_loss(pred, target)
        
        # Perceptual loss (difference in mel-scale)
        mel_diff = torch.mean(torch.abs(pred - target) * torch.log(1 + torch.abs(target)))
        
        return l1 + 0.5 * mse + 2.0 * spectral + 0.3 * mel_diff


def main():
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    feature_state = True
    train_loader, val_loader, test_loader = data_main(diffusion_ratio=1.0, feature_state=feature_state)
    print('\nAugmented data ready.')

    if feature_state:
        transformer_model = Transformer(src_dim=80, tgt_dim=80, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=101, dropout=0.1).to(device)
    else:
        transformer_model = Transformer(src_dim=256, tgt_dim=80, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=101, dropout=0.1).to(device)

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=15, delta=0.01)  

    model_path = f'outputs/output_transformer/transformer_model_16.pth'
    if os.path.exists(model_path):
        print("\nLoading existing augmented model...")
        transformer_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("\nTraining transformer with augmented data...")
        transformer_model = train_transformer_model(transformer_model, train_loader, val_loader, device, 150, optimizer, criterion, scheduler, early_stopping)

    gen_mspec, target_mspec = test_transformer_model(transformer_model, test_loader, device)

    try:
        hifigan = HifiGanModel.from_pretrained("nvidia/tts_hifigan").to(device)
        print("✓ HiFi-GAN loaded successfully")

        processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
        print("✓ HuBERT model loaded successfully")

    except Exception as e:
        print(f"[ERROR] Failed to load pretrained models: {e}")
        hifigan, processor, asr_model = None, None, None

    if hifigan and processor and asr_model:
        gen_texts, target_texts = [], []
        for i in tqdm(range(len(gen_mspec)), desc=f"Processing audio"):
            _, gen_text = mspec2audio2text_hubert(gen_mspec[i], hifigan, processor, asr_model)
            _, target_text = mspec2audio2text_hubert(target_mspec[i], hifigan, processor, asr_model)
            gen_texts.append(gen_text)
            target_texts.append(target_text)
        
        cer = CharErrorRate()(gen_texts, target_texts).item()
        wer = WordErrorRate()(gen_texts, target_texts).item()
        # print(f"Gen_texts: {gen_texts}")
        # print(f"Target_texts: {target_texts}")
        print(f"Character Error Rate (CER): {cer*100:.4f}")
        print(f"Word Error Rate (WER): {wer*100:.4f}")
    
    else:
        print("Skipping audio generation due to HiFi-GAN loading failure.")

    # plot_mspec(target_mspec[0].squeeze().cpu().numpy(), gen_mspec[0].squeeze().cpu().numpy(), title=None, save_path="outputs/output_transformer/mspec_comparison.png")
    # gen_audio, _ = mspec2audio2text_hubert(gen_mspec[0].squeeze().cpu().numpy(), hifigan, processor, asr_model)
    # target_audio, _ = mspec2audio2text_hubert(target_mspec[0].squeeze().cpu().numpy(), hifigan, processor, asr_model)
    # sf.write("outputs/output_transformer/target_audio.wav", target_audio, 16000)
    # sf.write("outputs/output_transformer/gen_audio.wav", gen_audio, 16000)


if __name__ == "__main__":
    main()