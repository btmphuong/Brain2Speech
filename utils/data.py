import numpy as np
import scipy.io as sio
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils.melSpectrogram import load_audio

import torch.nn as nn
import torch.nn.functional as F

from utils.align import align_from_distances
from model.diffusion import ECoGDiffusionModel, ConditionalECoGDiffusion
from torchmetrics import MeanSquaredError, PearsonCorrCoef

import random 


def meanSubtract(dat, area='6v'):
    """Subtracts the mean of each block from the neural features."""
    if area == '6v':
        dat['feat'] = np.concatenate([dat['tx2'][:, :128], dat['spikePow'][:, :128]], axis=1).astype(np.float32)
    elif area == '44':
        dat['feat'] = np.concatenate([dat['tx2'][:, 128:], dat['spikePow'][:, 128:]], axis=1).astype(np.float32)

    for b in np.unique(dat['blockNum']):
        idx = dat['blockNum'].squeeze() == b
        dat['feat'][idx] -= np.mean(dat['feat'][idx], axis=0, keepdims=True)

    return dat


def load_data(filename, area='6v'):
    """Loads and processes ECoG and spectrogram data from MATLAB file."""
    data = sio.loadmat(filename)
    data = meanSubtract(data, area)

    goTrialEpochs = data['goTrialEpochs'] - 1
    cueList = data['cueList'][0, 1:51]
    trialCues = data['trialCues'] - 1

    ecog_data = []
    valid_trialCues = []

    # Extract segments based on goTrialEpochs and trialCues
    for idx, (start, end) in enumerate(goTrialEpochs):
        if trialCues[idx] != 0: # except do_nothing
            segment = data['feat'][start:end, :]
            ecog_data.append(segment)
            valid_trialCues.append(trialCues[idx] - 1) 
    trialCues = np.array(valid_trialCues).flatten()

    # Padding
    max_len = max(segment.shape[0] for segment in ecog_data)
    n_trials = len(ecog_data)
    n_channels = ecog_data[0].shape[1]

    ecog_data_padded = np.zeros((n_trials, max_len, n_channels))
    for i, segment in enumerate(ecog_data):
        length = segment.shape[0]
        ecog_data_padded[i, :length, :] = segment
    ecog_data = np.array(ecog_data_padded, dtype=np.float32)

    # Create map between label and text
    label_text_dict = {}
    for i in range(len(trialCues)):
        label = trialCues[i]
        label_text_dict[label] = cueList[label]

    fixedlabels = []
    for x in range(len(cueList)):
        fixedlabels.append(cueList[x])
    
    # Load mel spectrograms and audio
    all_mspec, all_audio = [], []
    for i in tqdm(range(len(fixedlabels))):
        text = fixedlabels[i][0]
        mspec, audio = load_audio(text)
        all_mspec.append(mspec)
        all_audio.append(audio)
    
    # Make dictionary
    label_text_spectrogram_dict = {}
    for label in label_text_dict:
        label_text_spectrogram_dict[label] = {
            "text": label_text_dict[label],
            "mspec": all_mspec[label],
        }
    sub_dict = [label_text_spectrogram_dict[label] for label in trialCues]
    
    texts = [item['text'] for item in sub_dict]
    mspecs = [item['mspec'] for item in sub_dict]
    
    dictionary = {
        'text': texts,
        'mspec': mspecs,
        'ecog': ecog_data,
        'label': trialCues
    }
    return dictionary


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def prepare_dataloader_custom(ecog, mspec, labels, test_size=0.1, val_size=0.1, exclude_labels=None):
    """
    Custom dataloader preparation that excludes certain labels from training and validation sets
    and adds them to the test set
    """
    if exclude_labels is not None:
        # Separate excluded labels from the rest
        exclude_mask = np.isin(labels, exclude_labels)
        include_mask = ~exclude_mask
        
        # Data with excluded labels (goes to test set)
        ecog_excluded = [ecog[i] for i in range(len(ecog)) if exclude_mask[i]]
        mspec_excluded = [mspec[i] for i in range(len(mspec)) if exclude_mask[i]]
        labels_excluded = labels[exclude_mask]
        
        # Data without excluded labels (for train/val split)
        ecog_included = [ecog[i] for i in range(len(ecog)) if include_mask[i]]
        mspec_included = np.array([mspec[i] for i in range(len(mspec)) if include_mask[i]])
        labels_included = labels[include_mask]
        
        # Split the remaining data into train and validation
        if len(ecog_included) > 0:
            X_train, X_val, y_train, y_val, label_train, label_val = train_test_split(
                ecog_included, mspec_included, labels_included, 
                test_size=val_size, random_state=42, stratify=labels_included
            )
        else:
            X_train, X_val, y_train, y_val = [], [], np.array([]), np.array([])
            label_train, label_val = np.array([]), np.array([])
        
        # Combine excluded samples with any additional test samples
        if test_size > 0 and len(ecog_included) > 0:
            # Take additional samples for test set from the remaining data
            X_temp, X_test_extra, y_temp, y_test_extra, label_temp, label_test_extra = train_test_split(
                X_train, y_train, label_train, 
                test_size=test_size, random_state=42, stratify=label_train
            )
            # Update train set
            X_train, y_train, label_train = X_temp, y_temp, label_temp
            
            # Combine excluded samples with extra test samples
            X_test = ecog_excluded + list(X_test_extra)
            y_test = np.concatenate([mspec_excluded, y_test_extra])
            label_test = np.concatenate([labels_excluded, label_test_extra])
        else:
            # Only excluded samples go to test set
            X_test = ecog_excluded
            y_test = np.array(mspec_excluded)
            label_test = labels_excluded
    
    else:
        # Original behavior when no labels are excluded
        X_temp, X_test, y_temp, y_test, label_temp, label_test = train_test_split(
            ecog, mspec, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, label_train, label_val = train_test_split(
            X_temp, y_temp, label_temp, test_size=val_size_adjusted, random_state=42, stratify=label_temp
        )
    
    # Convert lists to numpy arrays
    X_train = np.array(X_train) if len(X_train) > 0 else np.array([])
    X_val = np.array(X_val) if len(X_val) > 0 else np.array([])
    X_test = np.array(X_test) if len(X_test) > 0 else np.array([])

    return X_train, X_val, X_test, y_train, y_val, y_test, label_train, label_val, label_test


def train_diffusion_model(ecog, label, num_epochs, batch_size, learning_rate, device):
    # Convert to numpy arrays if needed
    if isinstance(ecog, list):
        ecog = np.array(ecog)
    if isinstance(label, list):
        label = np.array(label)
    
    # CRITICAL: Remap labels to start from 0
    unique_labels = np.unique(label)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    reverse_mapping = {new_label: old_label for old_label, new_label in label_mapping.items()}
    
    # Apply mapping
    remapped_labels = np.array([label_mapping[l] for l in label])
    
    channels = ecog.shape[-1]
    seq_len = ecog.shape[1]
    num_labels = len(unique_labels)  

    model = ECoGDiffusionModel(
        channels=channels,
        seq_len=seq_len,
        dim=128,
        time_emb_dim=256, 
        num_blocks=6
    ).to(device)

    diffusion = ConditionalECoGDiffusion(
        model=model,
        num_labels=num_labels  
    )
    
    # Store mappings for later use
    diffusion.label_mapping = label_mapping
    diffusion.reverse_mapping = reverse_mapping
    
    # Move diffusion tensors to device
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.label_embedding = diffusion.label_embedding.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(diffusion.label_embedding.parameters()), 
        lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Use remapped labels for dataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(ecog).transpose(1, 2),  # [batch, channels, seq_len]
        torch.LongTensor(remapped_labels)  # Use remapped labels
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    diffusion.label_embedding.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, (x0, batch_labels) in enumerate(dataloader):
            x0, batch_labels = x0.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            loss = diffusion.training_step(x0, batch_labels)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at epoch {epoch}, batch {batch_idx}: {loss}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(diffusion.label_embedding.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1

        scheduler.step()
        avg_loss = total_loss / valid_batches
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return model, diffusion


def generate_synthetic_ecog(model, diffusion, original_labels, num_samples_per_label=5):
    """Generate synthetic ECoG data with proper label handling"""
    device = next(model.parameters()).device
    synthetic_data = []
    synthetic_labels = []
    
    unique_labels = np.unique(original_labels)
    
    # Ensure diffusion components are on correct device
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.label_embedding = diffusion.label_embedding.to(device)
    
    model.eval()
    diffusion.label_embedding.eval()

    for original_label in unique_labels:
        # Map to training label space
        if hasattr(diffusion, 'label_mapping') and original_label in diffusion.label_mapping:
            mapped_label = diffusion.label_mapping[original_label]
        else:
            # Fallback: find index in unique labels
            mapped_label = np.where(unique_labels == original_label)[0][0]
        
        # Verify mapped label is valid
        if mapped_label >= diffusion.label_embedding.num_embeddings:
            print(f"ERROR: Mapped label {mapped_label} exceeds embedding size {diffusion.label_embedding.num_embeddings}")
            continue
        
        # Generate in small batches to avoid memory issues
        batch_size = min(num_samples_per_label, 3)
        for start_idx in range(0, num_samples_per_label, batch_size):
            current_batch_size = min(batch_size, num_samples_per_label - start_idx)
            labels_batch = torch.full((current_batch_size,), mapped_label, device=device, dtype=torch.long)
            
            try:
                with torch.no_grad():
                    generated = diffusion.sample_conditional(labels_batch, device)
                    generated = generated.transpose(1, 2).cpu().numpy()  # Back to [batch, seq_len, channels]
                
                synthetic_data.extend(generated)
                # Use original labels for consistency with training data
                synthetic_labels.extend([original_label] * current_batch_size)
                
            except RuntimeError as e:
                print(f"Error generating for label {original_label}: {e}")
                continue
    
    print(f"Generated {len(synthetic_data)} synthetic samples")
    return np.array(synthetic_data), np.array(synthetic_labels)


class ConvLSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size=256, conv_channels=64, lstm_hidden=128, 
                 output_size=80, dropout=0.2):
        super(ConvLSTMFeatureExtractor, self).__init__()
        
        # Multi-scale convolutional feature extraction
        self.conv_block1 = self._make_conv_block(input_size, conv_channels, 3)
        self.conv_block2 = self._make_conv_block(conv_channels, conv_channels*2, 5)
        self.conv_block3 = self._make_conv_block(conv_channels*2, conv_channels*2, 7)
        
        # Attention for spatial (channel) selection
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(conv_channels*2, conv_channels//2, 1),
            nn.ReLU(),
            nn.Conv1d(conv_channels//2, conv_channels*2, 1),
            nn.Sigmoid()
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=conv_channels*2,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        # Self-attention for temporal relationships
        lstm_output_size = lstm_hidden * 2
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_size)
        )
        
        # Residual connection
        self.residual = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        
    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        residual = self.residual(x)
        
        # Convolutional feature extraction
        x_conv = x.transpose(1, 2)  # [batch, input_size, seq_len]
        conv1 = self.conv_block1(x_conv)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        
        # Apply channel attention
        attention = self.channel_attention(conv3)
        conv_features = conv3 * attention
        conv_features = conv_features.transpose(1, 2)  # [batch, seq_len, conv_channels*2]
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_features)
        
        # Temporal attention
        attended_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        attended_out = lstm_out + attended_out  # Residual connection
        
        # Output projection
        output = self.output_proj(attended_out)
        output = output + residual
        output = self.layer_norm(output)
        
        return output


def main(diffusion_ratio, feature_state=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('\nLoading data...')
    dictionary = load_data('/mnt/d/GIST/Project/Project 1/ecog_word/dataset/t12.2022.05.03_fiftyWordSet.mat')
    ecog, mspec, label = dictionary['ecog'], dictionary['mspec'], dictionary['label']
    
    print('\nAligning ECoG and mel spectrograms...')
    aligned_mspec = [mspec[i][align_from_distances(ecog[i], mspec[i], debug=False)] for i in tqdm(range(len(mspec)))]
    aligned_mspec = np.array(aligned_mspec, dtype=np.float32)

    X_train, X_val, X_test, y_train, y_val, y_test, label_train, label_val, label_test = prepare_dataloader_custom(
        ecog, aligned_mspec, label,
        test_size=0.1, val_size=0.1,
        exclude_labels=[4, 13, 40, 26, 48, 10, 29, 18, 28, 6]
    )
    print(f"\nData split summary:")
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    num_samples_per_label = int(round((len(X_train) / len(np.unique(label_train))) * diffusion_ratio))

    if num_samples_per_label > 0:
        print('\nTraining diffusion model...')
        model, diffusion = train_diffusion_model(
            X_train, label_train,
            num_epochs=100, batch_size=4, learning_rate=1e-4, 
            device=device
        )

        print('\nGenerating synthetic ECoG data...')
        synthetic_ecog, synthetic_labels = generate_synthetic_ecog(
            model, diffusion, label_train, num_samples_per_label=num_samples_per_label
        )
        print(f'Original train set size: {X_train.shape}')
        print(f'Synthetic train set size: {synthetic_ecog.shape}')

        # Create synthetic mel spectrograms
        synthetic_mspec = []
        for syn_label in synthetic_labels:
            matching_indices = np.where(label_train == syn_label)[0]
            if len(matching_indices) > 0:
                idx = np.random.choice(matching_indices)
                synthetic_mspec.append(y_train[idx])  
            else:
                synthetic_mspec.append(y_train[0])
        synthetic_mspec = np.array(synthetic_mspec, dtype=np.float32)

        # Combine data
        augmented_ecog = np.concatenate((X_train, synthetic_ecog), axis=0)
        augmented_mspec = np.concatenate((y_train, synthetic_mspec), axis=0)
    else:
        print('\nSkipping diffusion model and synthetic data generation.')
        augmented_ecog = X_train
        augmented_mspec = y_train

    if feature_state:
        feature_extractor = ConvLSTMFeatureExtractor(input_size=256, conv_channels=64, lstm_hidden=128, output_size=80).to(device)
        feature_extractor.eval() 
        print('\nExtracting features from augmented ECoG data...')

        # Extract features for training data (augmented)
        with torch.no_grad():
            train_features = feature_extractor(
                torch.tensor(augmented_ecog, dtype=torch.float32).to(device)
            ).cpu().detach()  # Move back to CPU for DataLoader
        
        # Extract features for validation data  
        with torch.no_grad():
            val_features = feature_extractor(
                torch.tensor(X_val, dtype=torch.float32).to(device)
            ).cpu().detach()
        
        # Extract features for test data
        with torch.no_grad():
            test_features = feature_extractor(
                torch.tensor(X_test, dtype=torch.float32).to(device)
            ).cpu().detach()
        
        # Create dataloaders with extracted features
        train_loader = DataLoader(
            TensorDataset(
                train_features,  
                torch.tensor(augmented_mspec, dtype=torch.float32)
            ),
            batch_size=4, shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(
                val_features,  
                torch.tensor(y_val, dtype=torch.float32)
            ),
            batch_size=4, shuffle=False
        )
        
        test_loader = DataLoader(
            TensorDataset(
                test_features,  
                torch.tensor(y_test, dtype=torch.float32)
            ),
            batch_size=4, shuffle=False
        )
        
    else:
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(augmented_ecog, dtype=torch.float32),
                torch.tensor(augmented_mspec, dtype=torch.float32)
            ),
            batch_size=4, shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            ),
            batch_size=4, shuffle=False
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            ),
            batch_size=4, shuffle=False
        )
    
    print(f'\nFinal dataset sizes:')
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    return train_loader, val_loader, test_loader