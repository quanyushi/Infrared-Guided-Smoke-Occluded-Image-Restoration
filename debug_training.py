#!/usr/bin/env python3
"""
Debug script to diagnose NaN issues in training
"""

import torch
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

def check_data_statistics(dataset, num_samples=10):
    """Check data statistics for anomalies"""
    print("=== Data Statistics ===")
    
    for i, data in enumerate(dataset):
        if i >= num_samples:
            break
            
        print(f"\nSample {i+1}:")
        
        # Check A (masked visible)
        A = data['A']
        print(f"  A shape: {A.shape}")
        print(f"  A range: [{A.min():.4f}, {A.max():.4f}]")
        print(f"  A mean: {A.mean():.4f}, std: {A.std():.4f}")
        print(f"  A has NaN: {torch.isnan(A).any()}")
        print(f"  A has Inf: {torch.isinf(A).any()}")
        
        # Check A_guide (infrared)
        A_guide = data['A_guide']
        print(f"  A_guide shape: {A_guide.shape}")
        print(f"  A_guide range: [{A_guide.min():.4f}, {A_guide.max():.4f}]")
        print(f"  A_guide mean: {A_guide.mean():.4f}, std: {A_guide.std():.4f}")
        print(f"  A_guide has NaN: {torch.isnan(A_guide).any()}")
        print(f"  A_guide has Inf: {torch.isinf(A_guide).any()}")
        
        # Check B (target)
        B = data['B']
        print(f"  B shape: {B.shape}")
        print(f"  B range: [{B.min():.4f}, {B.max():.4f}]")
        print(f"  B mean: {B.mean():.4f}, std: {B.std():.4f}")
        print(f"  B has NaN: {torch.isnan(B).any()}")
        print(f"  B has Inf: {torch.isinf(B).any()}")

def check_model_forward(model, data):
    """Check model forward pass for issues"""
    print("\n=== Model Forward Pass ===")
    
    # Set input
    model.set_input(data)
    
    print(f"Input A range: [{model.real_A.min():.4f}, {model.real_A.max():.4f}]")
    print(f"Input A_guide range: [{model.real_A_guide.min():.4f}, {model.real_A_guide.max():.4f}]")
    print(f"Target B range: [{model.real_B.min():.4f}, {model.real_B.max():.4f}]")
    
    # Forward pass
    try:
        model.forward()
        print(f"Generated fake_B range: [{model.fake_B.min():.4f}, {model.fake_B.max():.4f}]")
        print(f"Generated fake_B mean: {model.fake_B.mean():.4f}, std: {model.fake_B.std():.4f}")
        print(f"fake_B has NaN: {torch.isnan(model.fake_B).any()}")
        print(f"fake_B has Inf: {torch.isinf(model.fake_B).any()}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return False
    
    return True

def check_loss_computation(model):
    """Check loss computation for issues"""
    print("\n=== Loss Computation ===")
    
    try:
        # Discriminator backward
        model.set_requires_grad(model.netD, True)
        model.optimizer_D.zero_grad()
        model.backward_D()
        
        print(f"D_real loss: {model.loss_D_real.item():.6f}")
        print(f"D_fake loss: {model.loss_D_fake.item():.6f}")
        print(f"D total loss: {model.loss_D.item():.6f}")
        
        # Generator backward
        model.set_requires_grad(model.netD, False)
        model.optimizer_G.zero_grad()
        model.backward_G()
        
        print(f"G_GAN loss: {model.loss_G_GAN.item():.6f}")
        print(f"G_L1 loss: {model.loss_G_L1.item():.6f}")
        print(f"G_structure loss: {model.loss_G_structure.item():.6f}")
        print(f"G total loss: {model.loss_G.item():.6f}")
        
        # Check for NaN in losses
        losses = [model.loss_D_real, model.loss_D_fake, model.loss_D,
                 model.loss_G_GAN, model.loss_G_L1, model.loss_G_structure, model.loss_G]
        
        for i, loss in enumerate(losses):
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"WARNING: Loss {i} contains NaN or Inf!")
                return False
                
    except Exception as e:
        print(f"Error in loss computation: {e}")
        return False
    
    return True

def check_gradients(model):
    """Check gradients for issues"""
    print("\n=== Gradient Check ===")
    
    # Check generator gradients
    g_grad_norms = []
    g_nan_count = 0
    
    for name, param in model.netG.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            g_grad_norms.append(grad_norm)
            
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"WARNING: NaN/Inf gradient in G parameter: {name}")
                g_nan_count += 1
    
    if g_grad_norms:
        print(f"Generator gradient norms: min={min(g_grad_norms):.6f}, max={max(g_grad_norms):.6f}, mean={np.mean(g_grad_norms):.6f}")
        print(f"Generator parameters with NaN/Inf gradients: {g_nan_count}")
    
    # Check discriminator gradients
    d_grad_norms = []
    d_nan_count = 0
    
    for name, param in model.netD.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            d_grad_norms.append(grad_norm)
            
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"WARNING: NaN/Inf gradient in D parameter: {name}")
                d_nan_count += 1
    
    if d_grad_norms:
        print(f"Discriminator gradient norms: min={min(d_grad_norms):.6f}, max={max(d_grad_norms):.6f}, mean={np.mean(d_grad_norms):.6f}")
        print(f"Discriminator parameters with NaN/Inf gradients: {d_nan_count}")

def main():
    # Parse options
    opt = TrainOptions().parse()
    opt.netG = 'dual_branch_unet'
    opt.batch_size = 1  # Use small batch for debugging
    
    print("=== Debugging Dual Branch Training ===")
    print(f"Using dataset: {opt.dataroot}")
    print(f"Model: {opt.model}")
    print(f"Generator: {opt.netG}")
    
    # Create dataset
    dataset = create_dataset(opt)
    print(f"Dataset size: {len(dataset)}")
    
    # Check data statistics
    check_data_statistics(dataset, num_samples=3)
    
    # Create model
    model = create_model(opt)
    model.setup(opt)
    # Set model to training mode
    if hasattr(model, 'netG'):
        model.netG.train()
    if hasattr(model, 'netD'):
        model.netD.train()
    
    # Test with first batch
    data = next(iter(dataset))
    
    # Check model forward pass
    if not check_model_forward(model, data):
        print("FAILED: Model forward pass")
        return
    
    # Check loss computation
    if not check_loss_computation(model):
        print("FAILED: Loss computation")
        return
    
    # Check gradients
    check_gradients(model)
    
    print("\n=== Debug Complete ===")
    print("If no major issues were found, try training with lower learning rate and structure loss weight.")

if __name__ == '__main__':
    main()
