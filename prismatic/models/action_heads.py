"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class ActionTransformerBlock(nn.Module):
    """
    Transformer block for action expert, similar to OpenPI's Gemma action expert.
    Includes self-attention to model relationships between action tokens in a chunk.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, attn_mask=None):
        # x: (batch_size, seq_len, dim)
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + x
        
        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class DiffusionActionHead(nn.Module):
    """
    OpenPI-style Diffusion Action Head with Transformer-based action expert.
    
    Architecture:
    - Training: Full VLM forward, predict noise from action hidden states
    - Inference: Prefix forward once (cache KV), suffix forward per denoising step
    
    Key features:
    - action_in_proj: projects noisy actions to embedding space
    - action_expert: Transformer blocks (like OpenPI's Gemma action expert)
    - action_out_proj: projects to noise prediction
    
    Parameters:
    - With num_transformer_blocks=12, hidden_dim=1024: ~160M trainable parameters
    - With num_transformer_blocks=18, hidden_dim=1024: ~236M trainable parameters (closer to OpenPI's 300M)
    """

    def __init__(
        self,
        input_dim=4096,          # VLM hidden dimension (llm_dim)
        hidden_dim=1024,         # Transformer hidden dimension (like OpenPI's gemma_300m width=1024)
        action_dim=7,            # Action dimensionality
        num_diffusion_steps=100, # Number of diffusion steps
        num_transformer_blocks=12,# Number of transformer blocks (OpenPI uses 18 for gemma_300m)
        num_heads=8,             # Number of attention heads
    ):
        super().__init__()
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.num_transformer_blocks = num_transformer_blocks
        
        # === Input projection: noisy actions -> transformer hidden dim ===
        self.action_in_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # === Timestep encoding and projection ===
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # === Condition projection: VLM hidden states -> transformer hidden dim ===
        self.condition_proj = nn.Linear(input_dim, hidden_dim)
        
        # === Action Expert: Transformer blocks (like OpenPI's Gemma action expert) ===
        # This allows action tokens to attend to each other, modeling temporal relationships
        self.transformer_blocks = nn.ModuleList([
            ActionTransformerBlock(
                dim=hidden_dim, 
                num_heads=num_heads, 
                mlp_ratio=4,
                dropout=0.0,
            )
            for _ in range(num_transformer_blocks)
        ])
        
        # === Output projection: transformer hidden dim -> action dim ===
        self.action_out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Noise scheduler for diffusion
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps, 
            beta_schedule="squaredcos_cap_v2"
        )
        
        # Print parameter count
        self._print_param_count()
    
    def _print_param_count(self):
        """Print the number of parameters in this module."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DiffusionActionHead initialized:")
        print(f"  - Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  - Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  - num_transformer_blocks: {self.num_transformer_blocks}")
        print(f"  - hidden_dim: {self.hidden_dim}")

    def embed_actions(self, noisy_actions, timesteps):
        """
        Embed noisy actions and timesteps into transformer hidden space.
        
        Args:
            noisy_actions: (batch_size, chunk_len, action_dim)
            timesteps: (batch_size,) diffusion timesteps
            
        Returns:
            action_embeddings: (batch_size, chunk_len, hidden_dim)
            time_emb: (batch_size, hidden_dim)
        """
        # Project actions to hidden dimension
        action_tokens = self.action_in_proj(noisy_actions)  # (B, chunk_len, hidden_dim)
        
        # Encode timestep
        time_emb = self.time_encoder(timesteps.float())  # (B, hidden_dim)
        time_emb = time_emb.to(action_tokens.dtype)
        
        # Process timestep through MLP
        time_emb = self.time_mlp(time_emb)  # (B, hidden_dim)
        
        # Add timestep to each action token
        time_emb_expanded = time_emb.unsqueeze(1)  # (B, 1, hidden_dim)
        action_embeddings = action_tokens + time_emb_expanded  # (B, chunk_len, hidden_dim)
        
        return action_embeddings, time_emb

    def predict_from_hidden_states(self, actions_hidden_states):
        """
        Predict noise/velocity from VLM hidden states using Transformer action expert.
        
        Args:
            actions_hidden_states: (batch_size, chunk_len * action_dim, llm_dim) or
                                   (batch_size, chunk_len, llm_dim)
                                   
        Returns:
            prediction: (batch_size, chunk_len, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        
        # Handle different input shapes
        if actions_hidden_states.shape[1] == NUM_ACTIONS_CHUNK * ACTION_DIM:
            # Reshape: (B, chunk_len * action_dim, llm_dim) -> (B, chunk_len, action_dim * llm_dim)
            # Then take mean across the action_dim grouping
            reshaped = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, -1)
            # Mean over action_dim to get (B, chunk_len, llm_dim)
            x = reshaped.mean(dim=2)
        else:
            # Already (B, chunk_len, llm_dim)
            x = actions_hidden_states
        
        # Project VLM hidden states to transformer hidden dimension
        x = self.condition_proj(x)  # (B, chunk_len, hidden_dim)
        
        # Process through Transformer blocks (action expert)
        # This allows action tokens to attend to each other
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final projection to action dimension
        prediction = self.action_out_proj(x)  # (B, chunk_len, action_dim)
        
        return prediction

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Sample noise and create noisy actions for training.
        
        Args:
            ground_truth_actions: (batch_size, chunk_len, action_dim)
            
        Returns:
            dict with noise, noisy_actions, timesteps
        """
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        dtype = ground_truth_actions.dtype
        
        # Sample random noise
        noise = torch.randn(
            size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), 
            device=device, 
            dtype=dtype
        )
        
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            low=0, 
            high=self.noise_scheduler.config.num_train_timesteps, 
            size=(batch_size,), 
            device=device
        )
        
        # Add noise via forward diffusion
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)
        
        return {
            "noise": noise,
            "noisy_actions": noisy_actions,
            "timesteps": timesteps,
        }

    def sample_actions(self, condition_hidden_states, num_inference_steps=10):
        """
        Sample actions via iterative denoising with Transformer action expert.
        
        Args:
            condition_hidden_states: (batch_size, chunk_len * action_dim, llm_dim) 
                                     VLM hidden states as condition
            num_inference_steps: Number of denoising steps
            
        Returns:
            actions: (batch_size, chunk_len, action_dim) denoised actions
        """
        batch_size = condition_hidden_states.shape[0]
        device = condition_hidden_states.device
        dtype = condition_hidden_states.dtype
        
        # Set scheduler timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # Start from pure noise
        noisy_actions = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), 
            device=device, 
            dtype=dtype
        )
        
        # Pool and project condition hidden states once
        cond_reshaped = condition_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, -1)
        cond_pooled = cond_reshaped.mean(dim=2)  # (B, chunk_len, llm_dim)
        cond_projected = self.condition_proj(cond_pooled)  # (B, chunk_len, hidden_dim)
        
        # Iterative denoising
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Embed current noisy actions with timestep
            action_emb, _ = self.embed_actions(noisy_actions, timesteps)  # (B, chunk_len, hidden_dim)
            
            # Combine with condition
            x = action_emb + cond_projected
            
            # Process through Transformer blocks (action expert)
            for block in self.transformer_blocks:
                x = block(x)
            noise_pred = self.action_out_proj(x)  # (B, chunk_len, action_dim)
            
            # Denoise step
            noisy_actions = self.noise_scheduler.step(noise_pred, t, noisy_actions).prev_sample
        
        return noisy_actions
