import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class GuidedDataset(BaseDataset):
    """
    A dataset class for guided image reconstruction.
    
    This dataset loads:
    1. Masked visible images from trainA/testA folders
    2. Corresponding infrared guide images from trainA_guide/testA_guide folders
    3. Target complete visible images from trainB/testB folders
    
    The dataset supports both hierarchical and flat structures:
    - Hierarchical: dataroot/train/trainA, dataroot/train/trainA_guide, dataroot/train/trainB
    - Flat: dataroot/trainA, dataroot/trainA_guide, dataroot/trainB
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # Support both hierarchical structure (train/trainA) and flat structure (trainA)
        # First try hierarchical structure
        hierarchical_dir_A = os.path.join(opt.dataroot, opt.phase, opt.phase + 'A')
        hierarchical_dir_A_guide = os.path.join(opt.dataroot, opt.phase, opt.phase + 'A_guide')
        hierarchical_dir_B = os.path.join(opt.dataroot, opt.phase, opt.phase + 'B')
        
        # Then try flat structure
        flat_dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        flat_dir_A_guide = os.path.join(opt.dataroot, opt.phase + 'A_guide')
        flat_dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        # Check which structure exists and use it
        if (os.path.exists(hierarchical_dir_A) and 
            os.path.exists(hierarchical_dir_A_guide) and 
            os.path.exists(hierarchical_dir_B)):
            self.dir_A = hierarchical_dir_A
            self.dir_A_guide = hierarchical_dir_A_guide
            self.dir_B = hierarchical_dir_B
            print(f"Using hierarchical guided dataset structure:")
            print(f"  - Masked images: {self.dir_A}")
            print(f"  - Guide images: {self.dir_A_guide}")
            print(f"  - Target images: {self.dir_B}")
        elif (os.path.exists(flat_dir_A) and 
              os.path.exists(flat_dir_A_guide) and 
              os.path.exists(flat_dir_B)):
            self.dir_A = flat_dir_A
            self.dir_A_guide = flat_dir_A_guide
            self.dir_B = flat_dir_B
            print(f"Using flat guided dataset structure:")
            print(f"  - Masked images: {self.dir_A}")
            print(f"  - Guide images: {self.dir_A_guide}")
            print(f"  - Target images: {self.dir_B}")
        else:
            raise ValueError(f"Guided dataset structure not found. Please ensure you have:\n"
                           f"Hierarchical: {hierarchical_dir_A}, {hierarchical_dir_A_guide}, {hierarchical_dir_B}\n"
                           f"OR Flat: {flat_dir_A}, {flat_dir_A_guide}, {flat_dir_B}")

        # Load image paths
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A_guide_paths = sorted(make_dataset(self.dir_A_guide, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        # Verify that we have the same number of images in each directory
        self.A_size = len(self.A_paths)
        self.A_guide_size = len(self.A_guide_paths)
        self.B_size = len(self.B_paths)
        
        if not (self.A_size == self.A_guide_size == self.B_size):
            print(f"Warning: Mismatched dataset sizes - A: {self.A_size}, A_guide: {self.A_guide_size}, B: {self.B_size}")
            # Use the minimum size to avoid index errors
            min_size = min(self.A_size, self.A_guide_size, self.B_size)
            self.A_paths = self.A_paths[:min_size]
            self.A_guide_paths = self.A_guide_paths[:min_size]
            self.B_paths = self.B_paths[:min_size]
            self.A_size = self.A_guide_size = self.B_size = min_size
            print(f"Using minimum size: {min_size}")
        
        # Set up transforms
        # For visible images (A and B) - RGB
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        # For infrared guide images - typically grayscale, but we'll treat as RGB for consistency
        # You can modify this if your infrared images are single channel
        self.transform_A_guide = get_transform(self.opt, grayscale=False)  # Set to True if infrared is single channel

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns a dictionary that contains A, A_guide, B, and paths
            A (tensor)       -- masked visible image
            A_guide (tensor) -- infrared guide image
            B (tensor)       -- target complete visible image
            A_paths (str)    -- path to masked image
            A_guide_paths (str) -- path to guide image
            B_paths (str)    -- path to target image
        """
        # Make sure index is within range
        index = index % self.A_size
        
        # Get image paths
        A_path = self.A_paths[index]
        A_guide_path = self.A_guide_paths[index]
        B_path = self.B_paths[index]
        
        # Load images
        A_img = Image.open(A_path).convert('RGB')
        A_guide_img = Image.open(A_guide_path).convert('RGB')  # Change to 'L' if infrared is grayscale
        B_img = Image.open(B_path).convert('RGB')
        
        # Apply the same transform parameters to ensure spatial consistency
        transform_params = get_params(self.opt, A_img.size)
        
        # Get transforms with the same parameters
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.transform_A.transforms[0].__class__.__name__ == 'Grayscale' if len(self.transform_A.transforms) > 0 else False))
        A_guide_transform = get_transform(self.opt, transform_params, grayscale=False)  # Set to True if infrared is single channel
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.transform_B.transforms[0].__class__.__name__ == 'Grayscale' if len(self.transform_B.transforms) > 0 else False))
        
        # Apply transforms
        A = A_transform(A_img)
        A_guide = A_guide_transform(A_guide_img)
        B = B_transform(B_img)

        return {
            'A': A, 
            'A_guide': A_guide, 
            'B': B, 
            'A_paths': A_path, 
            'A_guide_paths': A_guide_path,
            'B_paths': B_path
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size
