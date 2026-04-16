import torch
from .base_model import BaseModel
from . import networks


class GuidedPix2PixModel(BaseModel):
    """ 
    This class implements the guided pix2pix model for infrared-guided image reconstruction.
    
    The model takes two inputs:
    1. Masked visible image (A)
    2. Infrared guide image (A_guide)
    
    And generates a complete visible image (fake_B) that should match the target (real_B).
    
    The training objective is: GAN Loss + lambda_L1 * ||G(A, A_guide)-B||_1
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase.

        Returns:
            the modified parser.
        """
        # Set defaults for guided pix2pix with dual branch architecture - ULTRA BALANCED VERSION
        parser.set_defaults(norm='batch', netG='dual_branch_unet', netD='basic', dataset_mode='guided', init_gain=0.02)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla', lr=0.0002, batch_size=2)  # Reduce batch size for high-res
            parser.add_argument('--lambda_L1', type=float, default=50.0, help='weight for L1 loss')
            parser.add_argument('--lambda_structure', type=float, default=5.0, help='weight for structure guidance loss')
            parser.add_argument('--visible_grad_weight', type=float, default=1.0, help='gradient weight for visible branch')
            parser.add_argument('--infrared_grad_weight', type=float, default=0.3, help='gradient weight for infrared branch')
            parser.add_argument('--d_lr_ratio', type=float, default=0.2, help='discriminator learning rate ratio')
            parser.add_argument('--d_update_ratio', type=int, default=2, help='update discriminator every N iterations')

        # Add guide image specific options
        parser.add_argument('--guide_nc', type=int, default=3, help='# of guide image channels: 3 for RGB infrared, 1 for grayscale infrared')
        
        return parser

    def __init__(self, opt):
        """Initialize the guided pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out
        self.loss_names = ['G_GAN', 'G_L1', 'G_structure', 'D_real', 'D_fake']
        
        # specify the images you want to save/display
        self.visual_names = ['real_A', 'real_A_guide', 'fake_B', 'real_B']
        
        # specify the models you want to save to the disk
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            
        # Calculate input channels for generator: masked image + guide image
        generator_input_nc = opt.input_nc + getattr(opt, 'guide_nc', 3)
        
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(generator_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # Discriminator takes only masked input (A) and output (B), like original pix2pix
            # We don't include the infrared guide in discriminator input
            discriminator_input_nc = opt.input_nc + opt.output_nc
            self.netD = networks.define_D(discriminator_input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionStructure = torch.nn.MSELoss()  # For structure guidance loss

            # Store gradient weights for dual branch training
            self.visible_grad_weight = getattr(opt, 'visible_grad_weight', 1.0)
            self.infrared_grad_weight = getattr(opt, 'infrared_grad_weight', 0.5)

            # Initialize discriminator update counter for frequency control
            self.d_update_counter = 0
            self.d_update_ratio = getattr(opt, 'd_update_ratio', 2)

            # initialize optimizers with different learning rates for balance
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # Use much lower learning rate for discriminator to prevent it from becoming too strong
            d_lr = opt.lr * getattr(opt, 'd_lr_ratio', 0.5)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=d_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            print(f"Generator LR: {opt.lr}, Discriminator LR: {d_lr}")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        
        # Set the main inputs
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A_guide = input['A_guide'].to(self.device)  # Guide image is always A_guide
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        
        # Set paths for visualization
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.guide_paths = input['A_guide_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Check input for NaN or Inf
        if torch.isnan(self.real_A).any() or torch.isinf(self.real_A).any():
            print("Warning: NaN or Inf detected in real_A")
        if torch.isnan(self.real_A_guide).any() or torch.isinf(self.real_A_guide).any():
            print("Warning: NaN or Inf detected in real_A_guide")

        # Concatenate masked image and guide image as input to generator
        # Concatenate along channel dimension (dim=1 for batch format [N, C, H, W])
        generator_input = torch.cat([self.real_A, self.real_A_guide], dim=1)
        self.fake_B = self.netG(generator_input)  # G(A, A_guide)

        # Check output for NaN or Inf
        if torch.isnan(self.fake_B).any() or torch.isinf(self.fake_B).any():
            print("Warning: NaN or Inf detected in fake_B")
            print(f"fake_B stats: min={self.fake_B.min()}, max={self.fake_B.max()}, mean={self.fake_B.mean()}")

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # Only use masked input (A) and generated output (fake_B), no infrared guide
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        # Only use masked input (A) and real target (real_B), no infrared guide
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)

    def backward_G(self):
        """Calculate GAN, L1, and structure guidance loss for the generator"""
        # First, G(A, A_guide) should fake the discriminator
        # Discriminator only sees masked input (A) and generated output (fake_B)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A, A_guide) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third, structure guidance loss - encourage structural consistency
        # Use edge detection to compare structural features
        self.loss_G_structure = self.compute_structure_loss(self.fake_B, self.real_B, self.real_A_guide) * getattr(self.opt, 'lambda_structure', 10.0)

        # combine loss and calculate gradients with gradient weighting
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_structure
        self.loss_G.backward()

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)

    def compute_structure_loss(self, fake_B, real_B, guide_img):
        """
        Compute structure guidance loss using edge detection
        Args:
            fake_B: generated image
            real_B: target image
            guide_img: infrared guide image
        Returns:
            structure loss value
        """
        # Sobel edge detection
        def sobel_edges(img):
            # Convert to grayscale if needed
            if img.size(1) == 3:
                gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
            else:
                gray = img

            # Sobel kernels
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)

            # Apply convolution
            edge_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
            edge_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)

            # Compute edge magnitude with numerical stability
            edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)  # Add small epsilon for stability
            return edges

        # Extract edges from all images
        fake_edges = sobel_edges(fake_B)
        real_edges = sobel_edges(real_B)
        guide_edges = sobel_edges(guide_img)

        # Check for NaN or Inf values
        if torch.isnan(fake_edges).any() or torch.isinf(fake_edges).any():
            print("Warning: NaN or Inf detected in fake_edges")
            return torch.tensor(0.0, device=fake_B.device, requires_grad=True)

        if torch.isnan(real_edges).any() or torch.isinf(real_edges).any():
            print("Warning: NaN or Inf detected in real_edges")
            return torch.tensor(0.0, device=fake_B.device, requires_grad=True)

        if torch.isnan(guide_edges).any() or torch.isinf(guide_edges).any():
            print("Warning: NaN or Inf detected in guide_edges")
            return torch.tensor(0.0, device=fake_B.device, requires_grad=True)

        # Structure consistency loss: generated edges should match target edges
        edge_loss = self.criterionStructure(fake_edges, real_edges)

        # Structure guidance loss: generated edges should be guided by infrared edges
        guidance_loss = self.criterionStructure(fake_edges, guide_edges) * 0.5

        # Check final loss for NaN
        total_loss = edge_loss + guidance_loss
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print("Warning: NaN or Inf detected in structure loss")
            return torch.tensor(0.0, device=fake_B.device, requires_grad=True)

        return total_loss

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights with strict adaptive strategy"""
        self.forward()                   # compute fake images: G(A, A_guide)

        # First, compute current D losses to make informed decision
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()  # This computes current D losses

        # Multi-level discriminator update control
        d_loss_avg = (self.loss_D_real + self.loss_D_fake) / 2

        # Condition 1: Loss threshold (D shouldn't be too strong)
        loss_condition = d_loss_avg > 0.2

        # Condition 2: Update frequency (update D less frequently)
        self.d_update_counter += 1
        freq_condition = (self.d_update_counter % self.d_update_ratio) == 0

        # Condition 3: GAN loss check (if G_GAN is too high, pause D updates)
        gan_condition = True
        if hasattr(self, 'loss_G_GAN'):
            gan_condition = self.loss_G_GAN < 5.0

        # Update D only if ALL conditions are met
        update_d = loss_condition and freq_condition and gan_condition

        if update_d:
            self.optimizer_D.step()
            print(f"D updated: d_loss={d_loss_avg:.3f}, counter={self.d_update_counter}")
        # If not updating D, gradients are already computed for monitoring

        # Always update G (but update twice if GAN loss is very high)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Extra G update if struggling
        if hasattr(self, 'loss_G_GAN') and self.loss_G_GAN > 8.0:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
