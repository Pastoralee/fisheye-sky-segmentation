import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter, generic_filter
import utils as ut
from termcolor import colored
from data import process_image
import lightgbm as lgb

def apply_crf(image, mask, num_classes=2, sxy_gauss=1, compat_gauss=1, sxy_bil=1, srgb_bil=1, compat_bil=50, steps=3):
    """
    Apply DenseCRF post-processing to refine a soft binary mask.

    Args:
        image (np.ndarray): Original RGB image of shape (H, W, 3).
        mask (np.ndarray): Soft or binary mask of shape (H, W), values in [0, 1] or [0, 255].
        num_classes (int): Number of segmentation classes.
        sxy_gauss (float): Gaussian kernel spatial standard deviation.
        compat_gauss (float): Compatibility for Gaussian kernel.
        sxy_bil (float): Bilateral kernel spatial standard deviation.
        srgb_bil (float): Bilateral kernel color standard deviation.
        compat_bil (float): Compatibility for bilateral kernel.
        steps (int): Number of CRF inference steps.

    Returns:
        np.ndarray: Refined binary mask of shape (H, W).
    """
    h, w = mask.shape
    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    model_op = np.zeros((2, h, w), dtype=np.float32)
    model_op[1, :, :] = mask
    model_op[0, :, :] = 1.0 - mask

    unary = unary_from_softmax(model_op)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(w, h, num_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=sxy_bil, srgb=srgb_bil, rgbim=image, compat=compat_bil, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(int(steps))
    return np.argmax(Q, axis=0).reshape((h, w))

def resize_image_to_multiple(image, multiple=32):
    """
    Resize image so that height and width are multiples of a given number.

    Args:
        image (np.ndarray): Input image of shape (H, W, C).
        multiple (int): Desired multiple.

    Returns:
        np.ndarray: Resized image.
    """
    h, w = image.shape[:2]
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def gray_contrast_energy(image):
    """
    Compute gray contrast energy of an RGB image using Laplacian filtering.

    Args:
        image (np.ndarray): RGB image as a NumPy array.

    Returns:
        np.ndarray: Normalized contrast energy map.
    """
    R, G, B = cv2.split(image)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    k = 0.1
    gh = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

    I_gh = cv2.filter2D(gray, -1, gh)
    Z_c = np.sqrt(I_gh**2)
    alpha = np.max(Z_c)
    CE_c = (alpha * Z_c) / (Z_c + alpha * k) - 0.2353
    return ut.min_max_norm(CE_c)

def preprocess(img):
    """
    Preprocess an image into float RGB, grayscale, HSV, and contrast maps.

    Args:
        img (np.ndarray): Input RGB image in uint8 format.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - img_float: Float32 RGB image.
            - gray_img: Grayscale image.
            - hsv_img: HSV image.
            - gray_contrast: Gray contrast energy map.
    """
    img_float = img.astype(np.float32)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    gray_contrast = gray_contrast_energy(img_float)
    return img_float, gray_img, hsv_img, gray_contrast

def extract_features_optimized(img_float, gray_img, hsv_img, gray_contrast, x, y, x0, x1, y0, y1):
    """
    Extract local features from a pixel and its patch for segmentation purposes.

    Args:
        img_float (np.ndarray): Float32 RGB image.
        gray_img (np.ndarray): Grayscale image.
        hsv_img (np.ndarray): HSV image.
        gray_contrast (np.ndarray): Gray contrast energy map.
        x, y (int): Target pixel coordinates.
        x0, x1, y0, y1 (int): Patch bounds around pixel.

    Returns:
        dict: Dictionary of computed pixel-level and patch-level features.
    """
    patch_gray = gray_img[y0:y1, x0:x1]
    r, g, b = img_float[y, x]
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    h, s, v = hsv_img[y, x]
    hist, _ = np.histogram(patch_gray, bins=10, range=(0, 1), density=True)
    center_val = gray_img[y, x]

    return {
        "r": r_norm, "g": g_norm, "b": b_norm,
        "h": h, "s": s, "v": v,

        "patch_mean": np.mean(patch_gray),
        "patch_std": np.std(patch_gray),
        "uniformity": np.sum(hist ** 2),
        "entropy": -np.sum(hist * np.log2(hist + 1e-9)),
        "texture": np.mean(np.abs(patch_gray - center_val)),

        "FD_RGB": -3.77*r_norm - 1.25*g_norm + 12.40*b_norm - 4.62,
        "FD_HSV": 3.35*h + 2.55*s + 8.58*v - 7.51,

        "excess_blue": 1.4 * b_norm - g_norm,
        "gray_contrast": gray_contrast[y, x],
    }

def mock_predict(image_resized, model, device):
    """
    Perform mock inference on a resized image using a given model.

    Args:
        image_resized (np.ndarray): Resized RGB image.
        model (torch.nn.Module): Segmentation model.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Softmax prediction for each class.
    """
    img_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        pred = model(img_tensor).squeeze().cpu().numpy()
    return pred

def load_image(path, target_size):
    """
    Load and resize an RGB image.

    Args:
        path (str): Path to the image file.
        target_size (int): Size to resize both height and width to.

    Returns:
        np.ndarray: Resized RGB image.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img

def load_mask(path, target_size=1024):
    """
    Load and binarize a mask image.

    Args:
        path (str): Path to the mask image.
        target_size (int): Target size to resize mask to.

    Returns:
        np.ndarray: Binary mask of shape (target_size, target_size).
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

def sample_valid_pixels(H, W, n_samples):
    """
    Sample pixels within a normalized circular region at the center of the image.

    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        n_samples (int): Number of samples to extract.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates.
    """
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    dist2ori = np.sqrt(xx**2 + yy**2)

    circle_mask = dist2ori <= 1.0  # rayon = 1
    ys_valid, xs_valid = np.where(circle_mask)

    if len(xs_valid) < n_samples:
        raise ValueError("Not enough valid pixels in the circle to sample")

    indices = np.random.choice(len(xs_valid), size=n_samples, replace=False)
    return xs_valid[indices], ys_valid[indices]

def extract_dataframe_from_split(data_dir, target_res, scales, num_samples, split, model, device, patch_size):
    """
    Extract pixel-level features and predictions from a dataset split and build a DataFrame.

    Args:
        data_dir (str): Path to the dataset root directory.
        target_res (int): Target resolution for images.
        scales (List[int]): List of scales to use for predictions.
        num_samples (int): Number of pixels to sample per image.
        split (str): Dataset split ('train', 'val', or 'test').
        model (torch.nn.Module): Segmentation model.
        device (torch.device): Computation device.
        patch_size (int): Size of the local patch around each pixel.

    Returns:
        pd.DataFrame: DataFrame containing features and labels.
    """
    img_dir = os.path.join(data_dir, "img", split)
    mask_dir = os.path.join(data_dir, "mask", split)
    img_files = sorted(os.listdir(img_dir))

    rows = []

    for fname in tqdm(img_files, desc=f"Processing {split}"):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img_orig = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_1024 = cv2.resize(img_rgb, (target_res, target_res))

        predictions = []
        for scale in scales:
            resized = cv2.resize(img_rgb, (scale, scale))
            pred = mock_predict(resized, model, device)
            pred_resized = cv2.resize(pred[..., 1] if pred.ndim == 3 else pred, (target_res, target_res), interpolation=cv2.INTER_LINEAR)
            predictions.append(pred_resized)

        mask = load_mask(mask_path, target_res)
        xs, ys = sample_valid_pixels(target_res, target_res, num_samples)
        img_float, gray_img, hsv_img, gray_contrast = preprocess(img_1024)

        for x, y in zip(xs, ys):
            row = {}

            half = patch_size // 2
            x0, x1 = max(0, x - half), min(target_res, x + half + 1)
            y0, y1 = max(0, y - half), min(target_res, y + half + 1)

            for i, s in enumerate(scales):
                pred_val = predictions[i][y, x]
                row[f"p_sky_{s}"] = pred_val

            features = extract_features_optimized(img_float, gray_img, hsv_img, gray_contrast, x, y, x0, x1, y0, y1)
            row.update(features)
            row["label"] = int(mask[y, x])
            rows.append(row)

    return pd.DataFrame(rows)

def texture_patch(gray_img, patch_size):
    """
    Compute local texture of a grayscale image using pixel neighborhood differences.

    Args:
        gray_img (np.ndarray): Grayscale image.
        patch_size (int): Size of the patch.

    Returns:
        np.ndarray: Texture map of the same shape as input.
    """
    def local_texture(patch):
        center = patch[len(patch) // 2]
        diffs = np.abs(patch - center)
        return np.sum(diffs) / (len(patch) - 1)
    
    return generic_filter(gray_img, local_texture, size=patch_size, mode='reflect')

def extract_rich_features(img, pred_dict):
    """
    Extract a rich set of features from an image and prediction maps.

    Args:
        img (np.ndarray): RGB input image.
        pred_dict (dict): Dictionary of prediction maps for different scales.

    Returns:
        np.ndarray: Feature matrix of shape (H*W, F), where F is number of features.
    """
    kernel_size = 7
    H, W, _ = img.shape
    img_float = img.astype(np.float32)
    img_norm = img_float / 255.0

    gray_contrast = gray_contrast_energy(img_float)

    r = img_norm[:, :, 0].flatten()
    g = img_norm[:, :, 1].flatten()
    b = img_norm[:, :, 2].flatten()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    h = hsv_img[:, :, 0].flatten()
    s = hsv_img[:, :, 1].flatten()
    v = hsv_img[:, :, 2].flatten()

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    patch_mean = uniform_filter(gray_img, size=kernel_size, mode='reflect')
    patch_sqmean = uniform_filter(gray_img ** 2, size=kernel_size, mode='reflect')
    patch_std = np.sqrt(patch_sqmean - patch_mean ** 2)

    texture = texture_patch(gray_img, patch_size=kernel_size)
    gray_img_quant = np.clip((gray_img * 10).astype(int), 0, 9)  # 10 bins
    one_hot = np.eye(10)[gray_img_quant.reshape(-1)].reshape(H, W, 10).astype(np.float32)
    uniformity = np.zeros((H, W), dtype=np.float32)
    entropy = np.zeros((H, W), dtype=np.float32)
    for bin_idx in range(10):
        bin_map = one_hot[:, :, bin_idx]
        bin_mean = uniform_filter(bin_map, size=kernel_size, mode='reflect')
        uniformity += bin_mean ** 2
        entropy -= bin_mean * np.log2(bin_mean + 1e-9)

    return np.stack([
        pred_dict["p_sky_512"],
        pred_dict["p_sky_1024"],
        pred_dict["p_sky_2048"],
        r, g, b,
        h, v,
        patch_mean.flatten(), 
        patch_std.flatten(),
        uniformity.flatten(), entropy.flatten(), texture.flatten(),
        -3.77*r - 1.25*g + 12.40*b - 4.62,
        3.35*h + 2.55*s + 8.58*v - 7.51,
        1.4*b-g,
        gray_contrast.flatten()
    ], axis=1)

def inference(pathImg, pathModel, model_name='efficientnet-b7', use_lgbm = False, resize_target=(1024, 1024)):
    """
    Perform inference on an image using a trained Unet++ model, optionally with a LightGBM meta-model.

    Loads a segmentation model checkpoint and applies preprocessing, resizing, and prediction to the input image.
    If `use_lgbm` is True, also loads a LightGBM meta-model and refines the prediction using extracted features.

    Args:
        pathImg (str): Path to the input image file.
        pathModel (str): Path to the directory containing the model checkpoint(s).
        model_name (str, optional): Name of the model/encoder (e.g., 'efficientnet-b7'). Default is 'efficientnet-b7'.
        use_lgbm (bool, optional): Whether to use a LightGBM meta-model for post-processing. Default is False.
        resize_target (tuple, optional): Target (width, height) for resizing the image and mask. Default is (1024, 1024).

    Returns:
        np.ndarray or None: The predicted binary mask as a NumPy array, or None if an error occurred.
    """

    threshold_dict = {
        'efficientnet-b5': 176 / 255,
        'efficientnet-b7': 157 / 255
    }
    threshold = threshold_dict.get(model_name, 0.5)

    if not os.path.exists(pathModel):
        print(colored(f"Error: Model folder not found at {pathModel}", "red"))
        return None

    checkpoint_path = os.path.join(pathModel, f'{model_name}.pt')
    if not os.path.isfile(checkpoint_path):
        print(colored(f"Error: Checkpoint file '{model_name}.pt' not found in {pathModel}", "red"))
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(colored(f"Error: Failed to load model from {checkpoint_path}. {e}", "red"))
        return None

    pre_processing = ut.decode_preprocessing(checkpoint['pre_processing'])
    input_channels = ut.compute_input_channels(pre_processing)
    model = smp.UnetPlusPlus(
        encoder_name=f'{model_name}', encoder_weights='advprop', 
        in_channels=input_channels, classes=1, activation='sigmoid'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    if not os.path.exists(pathImg):
        print(colored(f"Error: Image file not found at {pathImg}", "red"))
        return None
    image = cv2.imread(pathImg)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, resize_target, interpolation=cv2.INTER_LINEAR)

    if use_lgbm:
        lgbm_name = f"meta_model_{model_name.split('-')[-1]}.txt"
        lgbm_model_path = os.path.join(pathModel, lgbm_name)
        if not os.path.isfile(lgbm_model_path):
            print(colored(f"Error: LGBM model file '{lgbm_name}' not found in {pathModel}", "red"))
            return None
        try:
            model_lgbm = lgb.Booster(model_file=lgbm_model_path)
        except Exception as e:
            print(colored(f"Error: Failed to load LightGBM model from {lgbm_model_path}. {e}", "red"))
            return None

        scales = [512, 1024, 2048]
        pred_dict = {}
        for scale in scales:
            img_scaled = cv2.resize(image_rgb, (scale, scale), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.tensor(img_scaled, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = process_image(img_tensor.squeeze(), pre_processing).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(img_tensor).squeeze().cpu().numpy()
            pred_resized = cv2.resize(pred, resize_target, interpolation=cv2.INTER_LINEAR)
            pred_dict[f"p_sky_{scale}"] = pred_resized.flatten()

        X_lgbm = extract_rich_features(image_resized, pred_dict)
        pred_lgbm = model_lgbm.predict(X_lgbm)
        pred_mask = pred_lgbm.reshape(*resize_target)
    else:
        img_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = process_image(img_tensor.squeeze(), pre_processing).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()
        pred_mask = cv2.resize(pred, resize_target, interpolation=cv2.INTER_LINEAR)

    pred_mask = ut.fix_circle(pred_mask, size=resize_target[0])
    pred_mask = (pred_mask > threshold).astype(np.uint8)
    return pred_mask

def full_inference(pathImg, pathModel, model_name='efficientnet-b7', show_plots=True, sizes=(512, 1024, 2048)):
    """
    Run inference at multiple resolutions and with LGBM post-processing, and optionally display results.

    Args:
        pathImg (str): Path to the input image.
        pathModel (str): Path to the model directory.
        model_name (str): Model/encoder name.
        show_plots (bool): Whether to display the results.
        sizes (tuple): Resolutions to use for inference (default: (512, 1024, 2048)).

    Returns:
        tuple: Predictions at each resolution and with LGBM, or (None, ...) if error.
    """
    if not os.path.exists(pathImg):
        print(colored(f"Error: Image file not found at {pathImg}", "red"))
        return (None,) * (len(sizes) + 1)

    # Run inference at each size
    predictions = []
    for sz in sizes:
        pred = inference(pathImg, pathModel, model_name, use_lgbm=False, resize_target=(sz, sz))
        if pred is None:
            print(colored(f"Inference failed at size {sz}", "red"))
            return (None,) * (len(sizes) + 1)
        predictions.append(pred)

    # LGBM post-processing at 1024x1024
    pred_lgbm = inference(pathImg, pathModel, model_name, use_lgbm=True, resize_target=(1024, 1024))
    if pred_lgbm is None:
        print(colored("LGBM inference failed.", "red"))
        return tuple(predictions) + (None,)

    # Load image for plotting
    image = cv2.imread(pathImg)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if show_plots:
        n = len(sizes) + 2  # input + each size + LGBM
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Input RGB Image", fontsize=16, fontweight="bold")
        axes[0].axis("off")
        for i, (sz, pred) in enumerate(zip(sizes, predictions), start=1):
            axes[i].imshow(pred, cmap="gray")
            axes[i].set_title(f"Sky Probability\n({sz}×{sz} input)", fontsize=16, fontweight="bold")
            axes[i].axis("off")
        axes[-1].imshow(pred_lgbm, cmap="gray")
        axes[-1].set_title("Final LGBM Mask", fontsize=16, fontweight="bold")
        axes[-1].axis("off")
        plt.tight_layout()
        plt.show()

    return (*predictions, pred_lgbm)