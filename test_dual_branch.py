#!/usr/bin/env python3
"""
Test script for the dual-branch UNet generator
Verifies that the new architecture can be instantiated and run forward pass
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import networks
from models.guided_pix2pix_model import GuidedPix2PixModel

class TestOptions:
    """Mock options class for testing"""
    def __init__(self):
        # Basic options
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netG = 'dual_branch_unet'
        self.netD = 'dual_branch_discriminator'
        self.n_layers_D = 3
        self.norm = 'batch'
        self.no_dropout = False
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.lr = 0.0002
        self.beta1 = 0.5
        self.direction = 'AtoB'
        self.dataset_mode = 'guided'
        self.gan_mode = 'vanilla'
        self.pool_size = 0
        
        # Dual branch specific options
        self.guide_nc = 3
        self.lambda_L1 = 100.0
        self.lambda_structure = 10.0
        self.visible_grad_weight = 1.0
        self.infrared_grad_weight = 0.5
        
        # Device
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.isTrain = True

        # Additional required attributes for BaseModel
        self.checkpoints_dir = './checkpoints'
        self.name = 'test_dual_branch'
        self.verbose = False
        self.preprocess = 'resize_and_crop'

def test_dual_branch_generator():
    """Test the dual branch generator standalone"""
    print("Testing Dual Branch UNet Generator...")
    
    # Create generator
    visible_nc = 3
    infrared_nc = 3
    output_nc = 3
    ngf = 64
    
    generator = networks.DualBranchUnetGenerator(
        visible_nc=visible_nc,
        infrared_nc=infrared_nc, 
        output_nc=output_nc,
        ngf=ngf,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False
    )
    
    print(f"Generator created successfully")
    print(f"Number of parameters: {sum(p.numel() for p in generator.parameters())}")
    
    # Test forward pass
    batch_size = 2
    height, width = 256, 256
    
    # Create test input (concatenated visible + infrared)
    test_input = torch.randn(batch_size, visible_nc + infrared_nc, height, width)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = generator(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, {output_nc}, {height}, {width}]")
    
    # Verify output shape
    expected_shape = (batch_size, output_nc, height, width)
    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
    
    print("✓ Dual Branch Generator test passed!")
    return True

def test_guided_pix2pix_model():
    """Test the complete guided pix2pix model with dual branch"""
    print("\nTesting Guided Pix2Pix Model with Dual Branch...")
    
    opt = TestOptions()
    
    # Create model
    model = GuidedPix2PixModel(opt)
    
    print(f"Model created successfully")
    print(f"Generator type: {type(model.netG)}")
    
    # Create test data
    batch_size = 2
    height, width = 256, 256
    
    test_data = {
        'A': torch.randn(batch_size, 3, height, width),  # Masked visible image
        'A_guide': torch.randn(batch_size, 3, height, width),  # Infrared guide
        'B': torch.randn(batch_size, 3, height, width),  # Target image
        'A_paths': ['test1.jpg', 'test2.jpg'],
        'A_guide_paths': ['guide1.jpg', 'guide2.jpg']
    }
    
    # Set input
    model.set_input(test_data)
    
    print(f"Input A shape: {model.real_A.shape}")
    print(f"Input A_guide shape: {model.real_A_guide.shape}")
    print(f"Target B shape: {model.real_B.shape}")
    
    # Forward pass
    model.forward()
    
    print(f"Generated output shape: {model.fake_B.shape}")
    
    # Test backward pass (if training)
    if model.isTrain:
        # Test discriminator backward
        model.set_requires_grad(model.netD, True)
        model.optimizer_D.zero_grad()
        model.backward_D()
        
        # Test generator backward  
        model.set_requires_grad(model.netD, False)
        model.optimizer_G.zero_grad()
        model.backward_G()
        
        print(f"Loss G_GAN: {model.loss_G_GAN.item():.4f}")
        print(f"Loss G_L1: {model.loss_G_L1.item():.4f}")
        print(f"Loss G_structure: {model.loss_G_structure.item():.4f}")
        print(f"Loss D_real: {model.loss_D_real.item():.4f}")
        print(f"Loss D_fake: {model.loss_D_fake.item():.4f}")
    
    print("✓ Guided Pix2Pix Model test passed!")
    return True

def test_individual_components():
    """Test individual components of the dual branch architecture"""
    print("\nTesting Individual Components...")
    
    batch_size = 2
    height, width = 256, 256
    ngf = 64
    
    # Test Visible Light Encoder
    print("Testing Visible Light Encoder...")
    visible_encoder = networks.VisibleLightEncoder(3, ngf, nn.BatchNorm2d)
    visible_input = torch.randn(batch_size, 3, height, width)
    vis_features = visible_encoder(visible_input)
    print(f"Visible features keys: {list(vis_features.keys())}")
    print(f"Bottleneck shape: {vis_features['bottleneck'].shape}")
    
    # Test Structure Extraction Encoder
    print("Testing Structure Extraction Encoder...")
    struct_encoder = networks.StructureExtractionEncoder(3, ngf, nn.BatchNorm2d)
    infrared_input = torch.randn(batch_size, 3, height, width)
    struct_features = struct_encoder(infrared_input)
    print(f"Structure features keys: {list(struct_features.keys())}")
    print(f"Edge3 shape: {struct_features['edge3'].shape}")
    
    # Test Cross-Modal Guidance Module
    print("Testing Cross-Modal Guidance Module...")
    guidance_module = networks.CrossModalGuidanceModule(ngf, nn.BatchNorm2d)
    guided_features = guidance_module(vis_features, struct_features)
    print(f"Guided features keys: {list(guided_features.keys())}")
    
    # Test Feature Fusion Decoder
    print("Testing Feature Fusion Decoder...")
    fusion_decoder = networks.FeatureFusionDecoder(ngf, 3, nn.BatchNorm2d)
    output = fusion_decoder(guided_features)
    print(f"Final output shape: {output.shape}")
    
    print("✓ Individual components test passed!")
    return True

if __name__ == "__main__":
    print("Starting Dual Branch Architecture Tests...")
    print("=" * 50)
    
    try:
        # Test individual components
        test_individual_components()
        
        # Test standalone generator
        test_dual_branch_generator()
        
        # Test complete model
        test_guided_pix2pix_model()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("The dual branch architecture is ready for training.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
