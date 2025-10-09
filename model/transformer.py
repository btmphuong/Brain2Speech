import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """Individual expert network in MoE architecture"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert network"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    """Gating network that routes inputs to appropriate experts"""
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gates: Softmax probabilities for top-k experts
            indices: Expert indices for routing
        """
        # Calculate gating scores
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Apply softmax to top-k gates
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        
        return top_k_gates, top_k_indices


class MixtureOfExperts(nn.Module):
    """Complete Mixture of Experts layer with load balancing"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = GatingNetwork(input_dim, num_experts, top_k)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer
        
        Returns:
            output: Weighted combination of expert outputs
            load_balance_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
        
        # Get gating decisions
        gates, expert_indices = self.gate(x_flat)  # [batch_size * seq_len, top_k]
        
        # Initialize output tensor
        output = torch.zeros_like(x_flat[:, :self.experts[0].fc2.out_features])
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == i).any(dim=-1)
            
            if expert_mask.sum() == 0:
                continue
                
            # Get tokens for this expert
            expert_input = x_flat[expert_mask]
            expert_output = self.experts[i](expert_input)
            
            # Get corresponding gates
            expert_gate_positions = (expert_indices == i).nonzero()
            expert_gates = gates[expert_gate_positions[:, 0], expert_gate_positions[:, 1]]
            
            # Weighted addition to output
            output[expert_mask] += expert_gates.unsqueeze(-1) * expert_output
        
        # Calculate load balancing loss
        load_balance_loss = self._calculate_load_balance_loss(gates, expert_indices)
        
        # Reshape output
        output = output.view(batch_size, seq_len, -1)
        
        return output, load_balance_loss
    
    def _calculate_load_balance_loss(
        self, 
        gates: torch.Tensor, 
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Calculate auxiliary loss to encourage load balancing"""
        # Count tokens per expert
        expert_counts = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.num_experts):
            expert_counts[i] = (expert_indices == i).sum().float()
        
        # Calculate fraction of tokens per expert
        total_tokens = expert_counts.sum()
        expert_fractions = expert_counts / (total_tokens + 1e-8)
        
        # Target uniform distribution
        target_fraction = 1.0 / self.num_experts
        
        # L2 loss from uniform distribution
        load_balance_loss = torch.sum((expert_fractions - target_fraction) ** 2)
        
        return self.load_balance_weight * load_balance_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = MixtureOfExperts(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            num_experts=8,
            top_k=1,
            load_balance_weight=0.01
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output, load_balance_loss = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, load_balance_loss


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = MixtureOfExperts(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            num_experts=8,
            top_k=1,
            load_balance_weight=0.01
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output, load_balance_loss = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x, load_balance_loss
    

class Transformer(nn.Module):
    def __init__(self, src_dim, tgt_dim, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        
        # Linear projections instead of embeddings for continuous data
        self.src_projection = nn.Linear(src_dim, d_model)
        self.tgt_projection = nn.Linear(tgt_dim, d_model)
        self.output_projection = nn.Linear(d_model, tgt_dim)
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, x):
        return (x.sum(dim=-1) == 0)
    
    def create_look_ahead_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, src, tgt):
        device = src.device
        batch_size = src.size(0)
        
        # Create padding masks
        src_padding_mask = self.create_padding_mask(src)  # [batch, src_len]
        tgt_padding_mask = self.create_padding_mask(tgt)  # [batch, tgt_len]
        
        # Create look-ahead mask
        tgt_seq_len = tgt.size(1)
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len, device)  # [tgt_len, tgt_len]
        
        # Project to model dimension
        src_embedded = self.dropout(self.positional_encoding(self.src_projection(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_projection(tgt)))

        # Encoder
        enc_output = src_embedded
        total_load_balance_loss = 0.0
        for enc_layer in self.encoder_layers:
            # Convert padding mask to attention mask format [batch, 1, 1, src_len]
            src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2) if src_padding_mask.any() else None
            enc_output, load_balance_loss = enc_layer(enc_output, src_mask)
            total_load_balance_loss += load_balance_loss
        
        # Decoder
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            # Source mask for cross-attention
            src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2) if src_padding_mask.any() else None
            
            # Target mask combining padding and look-ahead
            tgt_mask = tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
            if tgt_padding_mask.any():
                padding_mask = tgt_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
                tgt_mask = tgt_mask | padding_mask
            
            dec_output, load_balance_loss = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            total_load_balance_loss += load_balance_loss
        
        # Project back to target dimension
        output = self.output_projection(dec_output)
        return output, total_load_balance_loss