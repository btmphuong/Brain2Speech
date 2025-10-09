import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ECoGBlock(nn.Module):
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, dim)
        
        # Temporal convolution
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, max(dim // 4, 1), 1),  # Ensure at least 1 channel
            nn.ReLU(),
            nn.Conv1d(max(dim // 4, 1), dim, 1),
            nn.Sigmoid()
        )
        
        # Use LayerNorm instead of GroupNorm for more flexibility
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, time_emb):
        # x: [batch, dim, seq_len]
        # time_emb: [batch, time_emb_dim]
        
        # Transpose for LayerNorm: [batch, seq_len, dim]
        h = x.transpose(1, 2)
        h = self.norm1(h)
        h = h.transpose(1, 2)  # Back to [batch, dim, seq_len]
        h = F.relu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb_proj = self.time_mlp(time_emb)  # [batch, dim]
        h = h + time_emb_proj.unsqueeze(-1)  # Broadcast over seq_len
        
        # Second convolution
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Channel attention
        attention = self.channel_attention(h)
        h = h * attention
        
        return x + h  # Residual connection


class ECoGDiffusionModel(nn.Module):
    def __init__(self, channels=256, seq_len=101, dim=128, time_emb_dim=256, num_blocks=6):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(channels, dim, 1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            ECoGBlock(dim, time_emb_dim) for _ in range(num_blocks // 2)
        ])
        
        # Middle block
        self.middle_block = ECoGBlock(dim, time_emb_dim)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            ECoGBlock(dim, time_emb_dim) for _ in range(num_blocks // 2)
        ])
        
        # Output projection
        self.output_proj = nn.Conv1d(dim, channels, 1)
        
    def forward(self, x, t, label_emb=None):  # Added label_emb parameter
        """
        Args:
            x: [batch, channels, seq_len] - Noisy ECoG data
            t: [batch] - Time steps
            label_emb: [batch, emb_dim] - Label embeddings (optional)
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)
        
        # Add label embedding if provided
        if label_emb is not None:
            time_emb = time_emb + label_emb
        
        # Input projection
        h = self.input_proj(x)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for block in self.encoder_blocks:
            h = block(h, time_emb)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block(h, time_emb)
        
        # Decoder with skip connections
        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            h = h + skip
            h = block(h, time_emb)
        
        # Output
        output = self.output_proj(h)
        return output


class ECoGDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x0, t, noise=None):
        """Add noise to clean ECoG data"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod_t)
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps"""
        return torch.randint(0, self.timesteps, (batch_size,), device=device)
    
    def training_step(self, x0):
        """Single training step"""
        device = x0.device
        batch_size = x0.shape[0]
        
        # Sample timesteps and noise
        t = self.sample_timesteps(batch_size, device)
        noise = torch.randn_like(x0)
        
        # Add noise
        xt = self.add_noise(x0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(xt, t)
        
        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size, device, condition=None):
        """Generate new ECoG samples"""
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(batch_size, self.model.channels, self.model.seq_len, device=device)
        
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t_batch)
            
            # Denoise
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
        
        return x


class ConditionalECoGDiffusion(ECoGDiffusion):
    """Conditional diffusion model for label-specific ECoG generation"""
    
    def __init__(self, model, num_labels, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__(model, timesteps, beta_start, beta_end)
        self.num_labels = num_labels
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_labels, model.time_mlp[0].in_features)
        
    def training_step(self, x0, labels):
        """Conditional training step"""
        device = x0.device
        batch_size = x0.shape[0]
        
        # Sample timesteps and noise
        t = self.sample_timesteps(batch_size, device)
        noise = torch.randn_like(x0)
        
        # Add noise
        xt = self.add_noise(x0, t, noise)
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Predict noise (with conditioning)
        predicted_noise = self.model(xt, t, label_emb)
        
        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def sample_conditional(self, labels, device):
        """Generate ECoG samples conditioned on labels"""
        self.model.eval()
        batch_size = len(labels)
        
        # Start from pure noise
        x = torch.randn(batch_size, self.model.channels, self.model.seq_len, device=device)
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        for t in tqdm(reversed(range(self.timesteps))):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise with conditioning
            predicted_noise = self.model(x, t_batch, label_emb)
            
            # Denoise
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
        
        return x
    

