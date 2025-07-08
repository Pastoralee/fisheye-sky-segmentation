import numpy as np
import cv2
import torch
import torchvision.transforms as T
import random
from torchvision.transforms import v2
from termcolor import colored
import os
import torchvision.transforms.functional as F

def decode_preprocessing(code):
    from main import PreProcessing
    """
    Decode a 10-bit integer into a PreProcessing object.

    Each bit in the integer represents the activation of a specific preprocessing step.
    The bit order is:
    [base_img, dark_channel, depth_estimation, hue_disparity, color_saturation,
     contrast_energy, exponential_transform, color_gradient, hsv, kmeans]

    Args:
        code (int): An integer (0–1023) encoding the preprocessing configuration.

    Returns:
        PreProcessing: An object with boolean fields indicating which preprocessing steps are enabled.
    """

    return PreProcessing(
        base_img=bool((code >> 9) & 1),
        dark_channel=bool((code >> 8) & 1),
        depth_estimation=bool((code >> 7) & 1),
        hue_disparity=bool((code >> 6) & 1),
        color_saturation=bool((code >> 5) & 1),
        contrast_energy=bool((code >> 4) & 1),
        exponential_transform=bool((code >> 3) & 1),
        color_gradient=bool((code >> 2) & 1),
        hsv=bool((code >> 1) & 1),
        kmeans=bool((code >> 0) & 1),
    )

def encode_preprocessing(preprocessing):
    """
    Encode a PreProcessing object into a 10-bit integer.

    Each preprocessing step is represented by a single bit in the output integer.
    The bit order is:
    [base_img, dark_channel, depth_estimation, hue_disparity, color_saturation,
     contrast_energy, exponential_transform, color_gradient, hsv, kmeans]

    Args:
        preprocessing (PreProcessing): An object with boolean flags for each preprocessing step.
            If None is passed, returns 0b1000000000 (base_img only enabled by default).

    Returns:
        int: An integer (0–1023) encoding the preprocessing configuration.
    """
    if preprocessing is None:
        return 0b1000000000
    return (
        (int(preprocessing.base_img) << 9) |
        (int(preprocessing.dark_channel) << 8) |
        (int(preprocessing.depth_estimation) << 7) |
        (int(preprocessing.hue_disparity) << 6) |
        (int(preprocessing.color_saturation) << 5) |
        (int(preprocessing.contrast_energy) << 4) |
        (int(preprocessing.exponential_transform) << 3) |
        (int(preprocessing.color_gradient) << 2) |
        (int(preprocessing.hsv) << 1) |
        (int(preprocessing.kmeans) << 0)
    )

def load_correct_state(path, device, input_channels):
    """
    Loads a model checkpoint from the given file path and returns the state dictionary.

    The function supports different checkpoint formats:
    - If the checkpoint is from SWAV-like pretraining, it is expected to have a 'state_dict' key.
    - If the checkpoint is a standard PyTorch checkpoint, it should have a 'model_state_dict' key.
    - Otherwise, the checkpoint is assumed to be a raw state dictionary.

    Args:
        path (str): Path to the checkpoint file.
        device (torch.device): Device on which to load the checkpoint.
        input_channels (int): Number of channels the model takes as input

    Returns:
        tuple:
            - is_swav (bool): Indicates whether the checkpoint is SWAV-like or not.
            - state_dict (dict): The extracted model state dictionary.

    Raises:
        ValueError: If the file is invalid or the format is not recognized.
    """
    
    valid_extensions = [".pth", ".pth.tar", ".pt"]
    lower_path = path.lower()
    if not os.path.isfile(path) or not any(lower_path.endswith(ext) for ext in valid_extensions):
        raise ValueError(colored(f"The specified file is not a valid checkpoint: {path}", "red"))

    print(colored(f"Loading pre-trained weights from: {path}", "green"))
    checkpoint = torch.load(path, weights_only=False, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(colored("Unexpected checkpoint format. Expected a dict with 'state_dict' or a raw state_dict.", "red"))

    if 'model_state_dict' in checkpoint:
        if input_channels != checkpoint['input_channels']:
            raise ValueError(colored(f"Pretrained model was training with {checkpoint['input_channels']} channels, but you specified {input_channels}", "red"))
        return False, checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        return True, checkpoint['state_dict']
    return True, checkpoint

def test_args_parameters(args):
    """
    Validates user-defined arguments to ensure consistency and correctness.

    Args:
        args (Args): Arguments to validate.

    Raises:
        ValueError: If parameters are incompatible or invalid.
    """

    if args.test is not None:
        if args.test < 0. or args.test > 1.:
            raise ValueError(colored("The test percentage must be between 0 and 1.", "red"))
    if args.validation is not None:
        if args.validation < 0. or args.validation > 1.:
            raise ValueError(colored("The validation percentage must be between 0 and 1.", "red"))
    if args.test is not None and args.validation is not None:
        if args.test + args.validation >= 1.:
            raise ValueError(colored("The sum of the test and validation percentages must be less than 1.", "red"))
    if args.test is not None and args.test <= 0:
        args.test = None
    if args.validation is not None and args.validation <= 0:
        args.validation = None
    if args.floatPrecision not in ['float32', 'float16', 'bfloat16']:
        raise ValueError(colored(f"Please chose either bfloat16, float16, or float32 as floatPrecision argument", "red"))
    
def compute_input_channels(pre_processing):
    """
    Computes the total number of input channels based on selected pre-processing methods.

    Args:
        pre_processing (object): An object with boolean attributes indicating 
                                 whether each pre-processing method is active.

    Returns:
        int: Total number of input channels.
    """
    
    channel_map = {
        'base_img': 3,
        'dark_channel': 1,
        'depth_estimation': 1,
        'hue_disparity': 1,
        'color_saturation': 1,
        'contrast_energy': 3,
        'exponential_transform': 1,
        'color_gradient': 1,
        'hsv': 3,
        'kmeans': 3,
    }

    total_channels = 0
    for key, channels in channel_map.items():
        if getattr(pre_processing, key, False):
            total_channels += channels

    return total_channels

def count_parameters(model):
    """
    Counts the total and trainable parameters of a given PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        tuple: Total parameters, trainable parameters.
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(colored(f"---------- MODEL: ----------", "green"))
    print(colored(f"Total parameters: {total_params:,}", "green"))
    print(colored(f"Trainable parameters: {trainable_params:,}", "green"))

def load_swav_pretrained_encoder_weights(model_encoder, model_proto, state_dict):
    """
    Loads pre-trained SwAV weights into encoder and prototype models.

    Args:
        model_encoder (torch.nn.Module): Encoder model to load backbone weights into.
        model_proto (torch.nn.Module): Prototype model to load the complete state_dict.
        state_dict (dict): Pre-trained SwAV weights, typically loaded from a checkpoint.

    Raises:
        ValueError: If the checkpoint file is invalid or incompatible.
    """

    model_proto.load_state_dict(state_dict)
    encoder_state_dict = {}
    for k, v in model_proto.state_dict().items():
            if "backbone" in k:
                new_key = k.replace("backbone.", "model.")
                encoder_state_dict[new_key] = v

    encoder_keys_model = set(model_encoder.state_dict().keys())
    encoder_keys_loaded = set(encoder_state_dict.keys())
    unexpected_keys = encoder_keys_loaded - encoder_keys_model
    matching_keys = encoder_keys_model & encoder_keys_loaded
    
    if unexpected_keys:
        print(colored(f"Unexpected keys in pretrained weights: {len(unexpected_keys)}", "yellow"))
    if len(matching_keys) < len(encoder_keys_model) * 0.5:
        print(colored("Warning: Less than 50% of weights were loaded. Make sure the encoder architecture matches.", "yellow"))

    model_encoder.load_state_dict(encoder_state_dict, strict=False)
    print(colored("Encoder weights loaded successfully.", "green"))

def fix_seed(seed):
    """
    Fixes random seeds for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed (int): The seed value to set.
    """
    
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(colored(f"Seed set to: {seed}", "green"))

#---------------------#
# ----- FISHEYE ----- #
#---------------------#

def fisheye_to_equirectangular(img: torch.Tensor) -> torch.Tensor:
    """
    Convert a square equidistant fisheye tensor [C, H, W] to equirectangular [C, H, 2W],
    only upper hemisphere (i.e., 0 ≤ θ ≤ π/2).

    Args:
        img (torch.Tensor): Input image.
    """

    C, H, W = img.shape
    output_height = H
    output_width = 2 * W
    half_height = output_height // 2

    theta = torch.linspace(0, np.pi / 2, half_height, device=img.device)
    phi = torch.linspace(-np.pi, np.pi, output_width, device=img.device)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    f = W / np.pi
    r = f * theta
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    px = (x + W/2).long()
    py = (y + H/2).long()
    mask = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (r <= W/2)

    output = torch.zeros((C, output_height, output_width), dtype=img.dtype, device=img.device)
    for c in range(C):
        equirect_half = torch.zeros((half_height, output_width), dtype=img.dtype, device=img.device)
        equirect_half[mask] = img[c, py[mask], px[mask]]
        output[c, :half_height] = equirect_half

    return output

def equirectangular_to_equisolid(img: torch.Tensor) -> torch.Tensor:
    """
    Convert equirectangular image tensor [C, H, W] (top-half only) into equisolid fisheye projection.

    Args:
        img (torch.Tensor): Input image.
    """

    C, H, W = img.shape
    half_H = H // 2
    img = img[:, :half_H, :] 

    xy = torch.linspace(-1, 1, W, device=img.device)
    x, y = torch.meshgrid(xy, xy, indexing='xy')
    r_norm = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    f = 1 / np.sqrt(2)

    theta = torch.zeros_like(r_norm)
    valid = r_norm <= 1
    theta[valid] = 2 * torch.arcsin(r_norm[valid] / (2 * f))
    lat = theta / (np.pi / 2)
    lon = (phi / (2 * np.pi)) + 0.5
    px = (lon * (W - 1)).long()
    py = (lat * (half_H - 1)).long()

    output = torch.zeros((C, W, W), dtype=img.dtype, device=img.device)
    for c in range(C):
        channel = torch.zeros((W, W), dtype=img.dtype, device=img.device)
        channel[valid] = img[c, py[valid], px[valid]]
        output[c] = channel

    return output

#-------------------------------#
# ----- DATA AUGMENTATION ----- #
#-------------------------------#

class CutMix(torch.nn.Module):
    """
    Applies CutMix augmentation to a batch of images and masks.
    """
    
    def __init__(self, beta1, beta2):
        super(CutMix, self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, imgs, masks):
        lamb = np.random.beta(self.beta1, self.beta2)
        W, H = imgs.shape[2], imgs.shape[3]
        cut_rat = np.sqrt(1. - lamb)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        rand_index = torch.randperm(imgs.size(0))
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        masks[:, :, bbx1:bbx2, bby1:bby2] = masks[rand_index, :, bbx1:bbx2, bby1:bby2]

        return imgs, masks

class AdjustGamma(torch.nn.Module):
    """
    Applies random gamma correction to images.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, image):
        return T.functional.adjust_gamma(image, random.uniform(0.5, 1.5))

class FisheyeDistortion(torch.nn.Module):
    """
    Applies fisheye lens distortion effect to an image.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, image, mask):
        epsilon = 1e-6
        image = image.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy()
        h, w, _ = image.shape
        K = np.array([[w, 0, w // 2], [0, w, h // 2], [0, 0, 1]], dtype=np.float32)
        D = np.random.normal(0.5, 0.5, 4) + epsilon
        D = np.clip(D, 0, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), None)
        distorted = cv2.fisheye.undistortImage(image, K, D=D, Knew=new_K)
        distorted_mask = cv2.fisheye.undistortImage(mask_np, K, D=D, Knew=new_K)
        if distorted.ndim == 2:
            distorted = np.expand_dims(distorted, axis=2)
        if distorted_mask.ndim == 2:
            distorted_mask = np.expand_dims(distorted_mask, axis=2)
        return torch.from_numpy(distorted).permute(2, 0, 1), torch.from_numpy(distorted_mask).permute(2, 0, 1)

class ChromaticAberration(torch.nn.Module):
    """
    Applies chromatic aberration by shifting color channels.
    """
    def __init__(self, max_shift=3):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, image):
        shift = random.uniform(1, self.max_shift)
        image = image.permute(1, 2, 0).numpy()
        h, w, _ = image.shape
        center_x, center_y = w // 2, h // 2
        r, g, b = cv2.split(image)

        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        norm_X = (X - center_x) / center_x
        norm_Y = (Y - center_y) / center_y
        radial = np.sqrt(norm_X**2 + norm_Y**2)

        dx = norm_X * radial * shift
        dy = norm_Y * radial * shift

        r_shifted = cv2.remap(r, (X + dx).astype(np.float32), (Y + dy).astype(np.float32), cv2.INTER_LINEAR)
        b_shifted = cv2.remap(b, (X - dx).astype(np.float32), (Y - dy).astype(np.float32), cv2.INTER_LINEAR)

        image = cv2.merge((r_shifted, g, b_shifted))
        return torch.from_numpy(image).permute(2, 0, 1)

class EquisolidDistortion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        equirectangular = fisheye_to_equirectangular(image)
        equisolid = equirectangular_to_equisolid(equirectangular)
        return equisolid

class CustomAugmentation(torch.nn.Module):
    """
    Apply a sequence of probabilistic augmentations to image-mask pairs

    This module performs random vertical and horizontal flips, geometric distortions, 
    and a series of color-based augmentations with configurable probabilities. 
    Augmentations are applied only to the image (not the mask), except for spatial 
    transformations which affect both.

    Args:
        prob_flip (float): Probability of applying vertical and/or horizontal flips.
        prob_augmentation (float): Probability of applying each color-based augmentation.
        prob_distorsion (float): Probability of applying geometric distortion (fisheye).

    Methods:
        forward(img, mask): Apply augmentations to the input image and corresponding mask.

    Returns:
        Tuple[Tensor, Tensor]: Augmented image and corresponding (possibly spatially 
        transformed) mask.
    """

    def __init__(self, prob_flip=0.5, prob_augmentation=0.5, prob_distorsion=0.5):
        super(CustomAugmentation, self).__init__()
        self.prob_flip = prob_flip
        self.prob_augmentation = prob_augmentation
        self.prob_distorsion = prob_distorsion
        
        self.brightness = v2.ColorJitter(brightness=0.5)
        self.contrast = v2.ColorJitter(contrast=0.5)
        self.saturation = v2.ColorJitter(saturation=0.5)
        self.hue = v2.ColorJitter(hue=0.2)
        self.sharpness = v2.RandomAdjustSharpness(sharpness_factor=3.0, p=1.0)
        self.gaussian_blur = v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2))
        self.adjust_gamma = AdjustGamma()
        self.chrom_aber = ChromaticAberration()
        self.fisheye = FisheyeDistortion()

        self.color_modifiers = [
            self.chrom_aber,
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
            self.sharpness,
            self.gaussian_blur,
            self.adjust_gamma,
        ]

    def forward(self, img, mask):
        if random.random() < self.prob_flip:
            img = F.vflip(img)
            mask = F.vflip(mask)
        if random.random() < self.prob_flip:
            img = F.hflip(img)
            mask = F.hflip(mask)
        
        if random.random() < self.prob_distorsion:
            img, mask = self.fisheye(img, mask)

        available_augmentations = self.color_modifiers.copy()
        while available_augmentations and random.random() < self.prob_augmentation:
            transform = random.choice(available_augmentations)
            available_augmentations.remove(transform)
            img = transform(img)

        return img, mask

#------------------------------------------------------------------------------#
# ------------------------------ PRE-PROCESSING ------------------------------ #
# https://www.researchgate.net/publication/324178136_Sky_Detection_in_Hazy_Image
#------------------------------------------------------------------------------#

def fix_circle(img, size=1024, save_type='float32'):
    """
    Masks an image with a centered circle (outside set to 0).

    Args:
        img (np.ndarray): Input image.
        size (int, optional): Size of the image. Defaults to 1024.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Masked image.
    """
    
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)
    dist2ori = np.sqrt(x**2 + y**2)

    img[dist2ori > 1] = 0
    return img.astype(save_type)

def min_max_norm(image, save_type='float32'):
    """
    Applies min-max normalization to the input image.

    Args:
        image (np.ndarray): Input image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Normalized image.
    """

    image = (image - image.min()) / (image.max() - image.min())
    return image.astype(save_type)

def compute_dark_channel(image, save_type='float32'):
    """
    Computes the dark channel of an image.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Dark channel map.
    """

    return np.min(image, axis=2).astype(save_type) / 255

def compute_depth_estimation(image, save_type='float32'):
    """
    Estimates depth from a hazy image using color properties.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Depth estimation map.
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    _, S, B = cv2.split(hsv_image)

    S = S.astype(np.float32) / 255.
    B = B.astype(np.float32) / 255.

    c0 = 0.121779
    c1 = 0.959710
    c2 = -0.780245

    return min_max_norm(c0 + c1 * B + c2 * S, save_type) 

def compute_hue_disparity(image, save_type='float32'):
    """
    Computes the hue disparity between the original and semi-inverse RGB images.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Hue disparity map.
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, _, _ = cv2.split(hsv_image)

    R, G, B = cv2.split(image.astype(save_type) / 255.)
    R_si = np.maximum(R, 1 - R)
    G_si = np.maximum(G, 1 - G)
    B_si = np.maximum(B, 1 - B)

    R_si = (R_si - 0.5) / 0.5
    G_si = (G_si - 0.5) / 0.5
    B_si = (B_si - 0.5) / 0.5

    semi_inverse_rgb = cv2.merge([R_si, G_si, B_si])
    semi_inverse_rgb = (semi_inverse_rgb * 255).astype(np.uint8)
    hsv_si = cv2.cvtColor(semi_inverse_rgb, cv2.COLOR_RGB2HSV)
    H_si, _, _ = cv2.split(hsv_si)

    return np.abs(H_si.astype(save_type) - H.astype(save_type)) / 255.

def compute_color_saturation(image, save_type='float32'):
    """
    Computes the color saturation of an image.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Saturation map.
    """

    R, G, B = cv2.split(image)

    min_val = np.minimum(np.minimum(R, G), B).astype(save_type)
    max_val = np.maximum(np.maximum(R, G), B).astype(save_type)
    max_val[max_val == 0] = 1

    return 1 - (min_val / max_val)

def compute_contrast_energy(image, save_type='float32'):
    """
    Computes the contrast energy across luminance and color opponent channels.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Contrast energy map.
    """

    image = image.astype(np.float32) / 255.0

    R, G, B = cv2.split(image)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    yb = 0.5 * (R + G) - B
    rg = R - G

    contrast_channels = {"gray": gray, "yb": yb, "rg": rg}
    tau_c = {"gray": 0.2353, "yb": 0.2287, "rg": 0.0528}
    k = 0.1

    # Second Derivative Operator (Laplacian)
    gh = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

    contrast_energy = {}
    for key, I_c in contrast_channels.items():
        I_gh = cv2.filter2D(I_c, -1, gh)
        Z_c = np.sqrt(I_gh**2)
        alpha = np.max(Z_c)
        CE_c = (alpha * Z_c) / (Z_c + alpha * k) - tau_c[key]
        contrast_energy[key] = CE_c

    return min_max_norm(cv2.merge([contrast_energy['gray'], contrast_energy['yb'], contrast_energy['rg']]), save_type)

#https://www.scitepress.org/papers/2017/60924/60924.pdf
def compute_exponential_transform(image, save_type='float32'):
    """
    Applies an exponential transformation to enhance dark regions.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Enhanced image.
    """

    image = image.astype(save_type) / 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    I_prime_min = np.min(image)
    I_prime_max = np.max(image)

    xi = 1 / (np.log(I_prime_max - I_prime_min + 1))
    image_et = np.exp((image) / xi) - 1 + I_prime_min

    return image_et

def compute_color_gradient(image, save_type='float32'):
    """
    Computes the combined color gradient magnitude of the image.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Gradient magnitude map.
    """

    R, G, B = cv2.split(image)

    sobel_r_x = cv2.Sobel(R, cv2.CV_64F, 1, 0, ksize=3)
    sobel_r_y = cv2.Sobel(R, cv2.CV_64F, 0, 1, ksize=3)
    sobel_g_x = cv2.Sobel(G, cv2.CV_64F, 1, 0, ksize=3)
    sobel_g_y = cv2.Sobel(G, cv2.CV_64F, 0, 1, ksize=3)
    sobel_b_x = cv2.Sobel(B, cv2.CV_64F, 1, 0, ksize=3)
    sobel_b_y = cv2.Sobel(B, cv2.CV_64F, 0, 1, ksize=3)

    return min_max_norm(np.sqrt(sobel_r_x**2 + sobel_r_y**2) + np.sqrt(sobel_g_x**2 + sobel_g_y**2) + np.sqrt(sobel_b_x**2 + sobel_b_y**2), save_type)

def compute_hsv(image, save_type='float32'):
    """
    Converts an RGB image to HSV color space.

    Args:
        image (np.ndarray): RGB image.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: HSV image.
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image.astype(save_type) / 255.

def compute_kmeans(image, n_clusters, save_type='float32'):
    """
    Applies K-means clustering for image color segmentation.

    Args:
        image (np.ndarray): RGB image.
        n_clusters (int): Number of color clusters.
        save_type (str, optional): Output data type. Defaults to 'float32'.

    Returns:
        np.ndarray: Segmented image.
    """

    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.85)
    _, labels, centers = cv2.kmeans(pixel_vals, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))
    return segmented_image.astype(save_type) / 255.