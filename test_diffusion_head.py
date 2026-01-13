#!/usr/bin/env python3
"""
Test script for OpenPI-style DiffusionActionHead.
Run this to verify the implementation works correctly.
"""
import torch
import sys

# Test basic imports
print("Testing imports...")
try:
    from prismatic.models.action_heads import DiffusionActionHead, SinusoidalPositionalEncoding
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test DiffusionActionHead initialization
print("\nTesting DiffusionActionHead initialization...")
try:
    action_head = DiffusionActionHead(
        input_dim=4096,
        hidden_dim=1024,  # OpenPI action expert uses width=1024 (gemma_300m)
        action_dim=ACTION_DIM,
        num_diffusion_steps=100,
        num_transformer_blocks=8,  # 8 transformer blocks
        num_heads=8,
    )
    print(f"✓ DiffusionActionHead created successfully")
    print(f"  - action_dim: {action_head.action_dim}")
    print(f"  - input_dim: {action_head.input_dim}")
    print(f"  - hidden_dim: {action_head.hidden_dim}")
    print(f"  - num_diffusion_steps: {action_head.num_diffusion_steps}")
    print(f"  - num_transformer_blocks: {action_head.num_transformer_blocks}")
except Exception as e:
    print(f"✗ Error creating DiffusionActionHead: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test embed_actions
print("\nTesting embed_actions...")
try:
    batch_size = 2
    device = "cpu"
    dtype = torch.float32
    
    noisy_actions = torch.randn(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, device=device, dtype=dtype)
    timesteps = torch.randint(0, 100, (batch_size,), device=device)
    
    action_emb, time_emb = action_head.embed_actions(noisy_actions, timesteps)
    print(f"✓ embed_actions successful")
    print(f"  - noisy_actions shape: {noisy_actions.shape}")
    print(f"  - timesteps shape: {timesteps.shape}")
    print(f"  - action_emb shape: {action_emb.shape}")
    print(f"  - time_emb shape: {time_emb.shape}")
except Exception as e:
    print(f"✗ Error in embed_actions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test predict_from_hidden_states
print("\nTesting predict_from_hidden_states...")
try:
    # Test with (B, chunk_len * action_dim, llm_dim) shape
    hidden_states_flat = torch.randn(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, 4096, device=device, dtype=dtype)
    prediction_flat = action_head.predict_from_hidden_states(hidden_states_flat)
    print(f"✓ predict_from_hidden_states (flat) successful")
    print(f"  - hidden_states shape: {hidden_states_flat.shape}")
    print(f"  - prediction shape: {prediction_flat.shape}")
    
    # Test with (B, chunk_len, llm_dim) shape
    hidden_states_chunk = torch.randn(batch_size, NUM_ACTIONS_CHUNK, 4096, device=device, dtype=dtype)
    prediction_chunk = action_head.predict_from_hidden_states(hidden_states_chunk)
    print(f"✓ predict_from_hidden_states (chunk) successful")
    print(f"  - hidden_states shape: {hidden_states_chunk.shape}")
    print(f"  - prediction shape: {prediction_chunk.shape}")
except Exception as e:
    print(f"✗ Error in predict_from_hidden_states: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test sample_noisy_actions
print("\nTesting sample_noisy_actions...")
try:
    gt_actions = torch.randn(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, device=device, dtype=dtype)
    result = action_head.sample_noisy_actions(gt_actions)
    print(f"✓ sample_noisy_actions successful")
    print(f"  - gt_actions shape: {gt_actions.shape}")
    print(f"  - noise shape: {result['noise'].shape}")
    print(f"  - noisy_actions shape: {result['noisy_actions'].shape}")
    print(f"  - timesteps shape: {result['timesteps'].shape}")
except Exception as e:
    print(f"✗ Error in sample_noisy_actions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test sample_actions
print("\nTesting sample_actions...")
try:
    condition_hidden = torch.randn(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, 4096, device=device, dtype=dtype)
    sampled_actions = action_head.sample_actions(condition_hidden, num_inference_steps=5)  # Use fewer steps for quick test
    print(f"✓ sample_actions successful")
    print(f"  - condition_hidden shape: {condition_hidden.shape}")
    print(f"  - sampled_actions shape: {sampled_actions.shape}")
except Exception as e:
    print(f"✗ Error in sample_actions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed! DiffusionActionHead is working correctly.")
print("="*50)
