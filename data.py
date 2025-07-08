import utils as ut
import os
import cv2
import concurrent.futures
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from termcolor import colored
from torchvision.transforms import v2

class Dataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for images and masks.

    Args:
        img (list): List of images.
        mask (list): List of corresponding masks.
        transform (callable): Transformation function applied to both images and masks.
        pre_processing (object): Pre-processing configuration.
        floatPrecision (torch.dtype, optional): Precision type ('torch.float32', 'torch.float16', etc.).
    """

    def __init__(self, img, mask, transform, pre_processing, floatPrecision=torch.float32):
        self.img = img
        self.mask = mask
        self.transform = transform
        self.pre_processing = pre_processing
        self.floatPrecision = floatPrecision
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        X = self.img[index]
        y = self.mask[index]
        if self.transform is not None:
            X, y = self.transform(X, y)
        if self.pre_processing is not None:
            X = process_image(X, self.pre_processing, self.floatPrecision)
        return X, y

def is_image_file(filename):
    """
    Check if a file is an image based on its extension.

    Args:
        filename (str): Filename to check.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return os.path.splitext(filename)[1].lower() in valid_extensions

def read_image(image_path, train_size):
    """
    Read an image using OpenCV and resize it.

    Args:
        image_path (str): Path to the image.
        train_size (int): Size to resize the image to (square).

    Returns:
        np.ndarray: The loaded and resized image.
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        print(f"Impossible de lire l'image : {image_path}")
    image = cv2.resize(image, (train_size, train_size), interpolation = cv2.INTER_CUBIC)
    return image

def load_images_from_folder(folder_path, train_size):
    """
    Load and resize all images from a folder.

    Args:
        folder_path (str): Path to the folder containing images.
        train_size (int): Size to resize the images to (square).

    Returns:
        list: List of loaded and resized images.
    """

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(colored(f"Folder not found : {folder_path}.", "red"))

    image_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if is_image_file(f)],
        key=str.lower
    )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(lambda img: read_image(img, train_size), image_files))
        
    return [img for img in images if img is not None]

def process_image(image_tensor, pre_processing, floatPrecision=torch.float32):
    """
    Apply selected pre-processing transformations to an image tensor.

    Args:
        image_tensor (torch.Tensor): Image tensor (C, H, W).
        pre_processing (object): Pre-processing configuration.
        floatPrecision (torch.dtype, optional): Precision type.

    Returns:
        torch.Tensor: Processed image tensor.
    """

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    image_np = (image_np * 255).astype(np.uint8)
    
    functions = {
        'dark_channel': ut.compute_dark_channel,
        'depth_estimation': ut.compute_depth_estimation,
        'hue_disparity': ut.compute_hue_disparity,
        'color_saturation': ut.compute_color_saturation,
        'contrast_energy': ut.compute_contrast_energy,
        'exponential_transform': ut.compute_exponential_transform,
        'color_gradient': ut.compute_color_gradient,
        'hsv': ut.compute_hsv,
        'kmeans': lambda img, save_type: ut.compute_kmeans(img, n_clusters=10, save_type=save_type)
    }
    transformed_images = []

    if pre_processing.base_img:
        base_tensor = image_tensor.to(dtype=floatPrecision)
        transformed_images.append(base_tensor)

    for key, func in functions.items():
        if getattr(pre_processing, key):
            img_processed = func(image_np, save_type=np.float32)
            if img_processed.ndim == 2:
                img_processed = np.expand_dims(img_processed, axis=2)
            tensor = torch.from_numpy(img_processed).permute(2, 0, 1)  # (C, H, W)
            transformed_images.append(tensor.to(dtype=floatPrecision))
    
    return torch.cat(transformed_images, dim=0) if len(transformed_images) >= 1 else image_tensor

def process_images(image_batch, pre_processing, floatPrecision=torch.float32):
    """
    Apply pre-processing to a batch of images.

    Args:
        image_batch (torch.Tensor): Batch of images (B, C, H, W).
        pre_processing (object): Pre-processing configuration.
        floatPrecision (torch.dtype, optional): Precision type.

    Returns:
        torch.Tensor: Batch of processed images.
    """

    return torch.stack([
        process_image(img, pre_processing, floatPrecision)
        for img in image_batch
    ])

def generate_split_data(list_images, list_masks, args, transform):
    """
    Generate training, testing, and validation datasets.

    Args:
        list_images (list): List of images.
        list_masks (list): List of masks.
        args (object): Arguments with split parameters.
        transform (callable): Transformation function.

    Returns:
        tuple: DataLoaders for train, test, and validation datasets.
    """

    dataloader_test, dataloader_val = None, None
    print(colored(f"Loading train data...", "green"))
    if args.test:
        print(colored(f"Splitting {args.test * 100}% for test data", "green"))
        test_sample_size = int(args.test * len(list_images))
        if test_sample_size <= 0:
            raise ValueError(colored(f"Not enough data: found {len(x_train)} images, but test set is supposed to be {args.test * 100}%", "red"))
        x_train, x_test, y_train, y_test = train_test_split(list_images, list_masks, test_size=test_sample_size, random_state=args.seed)
        dataloader_test = torch.utils.data.DataLoader(Dataset(x_test, y_test, None, args.processingArgs, args.floatPrecision), batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
    else:
        print(colored("No test split provided, skipping test phase", "yellow"))
        x_train, y_train = list_images, list_masks
    
    if args.validation:
        print(colored(f"Splitting {args.validation * 100}% for validation data", "green"))
        val_sample_size = int(args.validation * len(list_images))
        if val_sample_size <= 0:
            raise ValueError(colored(f"Not enough data: found {len(x_train)} images, but validation set is supposed to be {args.validation * 100}%", "red"))
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_sample_size, random_state=args.seed)
        dataloader_val = torch.utils.data.DataLoader(Dataset(x_val, y_val, None, args.processingArgs, args.floatPrecision), batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
    else:
        print(colored("No validation split provided, skipping validation phase", "yellow"))

    dataloader_train = torch.utils.data.DataLoader(Dataset(x_train, y_train, transform, args.processingArgs, args.floatPrecision), batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
    return dataloader_train, dataloader_test, dataloader_val

def find_folder_by_prefix(path, prefix):
    """
    Find the first folder starting with the given prefix (case-insensitive).

    Args:
        path (str): Path to search in.
        prefix (str): Prefix to search for.

    Returns:
        str or None: Path to the found folder, or None if not found.
    """

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if folder.lower().startswith(prefix.lower()) and os.path.isdir(folder_path):
            return folder_path
    return None

def load_dataset(image_paths, mask_paths, train_size, transform_image, transform_mask):
    """
    Load images and masks, apply transformations, and return them as lists.

    Args:
        image_paths (str): Path to images.
        mask_paths (str): Path to masks.
        train_size (int): Image resizing size.
        transform_image (callable): Image transformation function.
        transform_mask (callable): Mask transformation function.

    Returns:
        tuple: Lists of images and masks.
    """

    images = [transform_image(img) for img in load_images_from_folder(image_paths, train_size)]
    print(colored(f"Loading associated masks: {mask_paths}...", "green"))
    masks = [transform_mask(mask) for mask in load_images_from_folder(mask_paths, train_size)]
    
    if not(len(images)):
        raise ValueError(colored(f"No images found: {image_paths}", "red"))
    if not(len(masks)):
        raise ValueError(colored(f"No masks found: {mask_paths}", "red"))
    if len(images) != len(masks):
        raise ValueError(colored(f"Mismatch: {len(images)} images vs {len(masks)} masks", "red"))
    
    return images, masks

def get_image_files_with_subdirs(root_folder):
    """
    Get sorted image files organized by subfolders.

    Args:
        root_folder (str): Root folder to walk through.

    Returns:
        dict: Dictionary {relative_subdir: list_of_images}.
    """

    all_files = {}
    for subdir, _, files in os.walk(root_folder):
        relative_subdir = os.path.relpath(subdir, root_folder)  # Sous-dossier relatif
        image_files = sorted(
            [os.path.join(subdir, f) for f in files if is_image_file(f)],
            key=str.lower
        )
        if image_files:
            all_files[relative_subdir] = image_files
    return all_files

def validate_structure(path_data, path_masks):
    """
    Validate that data and mask folders have the same structure and files.

    Args:
        path_data (str): Path to data folder.
        path_masks (str): Path to mask folder.

    Raises:
        ValueError: If structure or files mismatch.
    """
    
    print(colored(f"Verifying data structure...", "green"))
    data_files = get_image_files_with_subdirs(path_data)
    mask_files = get_image_files_with_subdirs(path_masks)
    
    if set(data_files.keys()) != set(mask_files.keys()):
        raise ValueError(colored(f"The following subdirectories do not match between data and masks: {set(data_files.keys()) - set(mask_files.keys())}, {set(mask_files.keys()) - set(data_files.keys())}.", "red"))

    for subdir in data_files:
        data_names = {os.path.basename(f) for f in data_files[subdir]}
        mask_names = {os.path.basename(f) for f in mask_files[subdir]}
        
        if data_names != mask_names:
            raise ValueError(colored(f"The following images names do not match between data and masks: {data_names - mask_names}, {mask_names - data_names}", "red"))


def split_data_with_folders(args):
    """
    Split dataset into train, test, and validation sets based on folders or arguments.

    Args:
        args (object): Arguments containing paths and split percentages.

    Returns:
        tuple: DataLoaders for train, test, and validation datasets.

    Raises:
        ValueError: If folder structure is invalid or splits cannot be performed.
    """

    validate_structure(args.pathData, args.pathMasks)

    dataloader_train, dataloader_test, dataloader_val = None, None, None
    transform = ut.CustomAugmentation(args.dataAugmentation.prob_flip, args.dataAugmentation.prob_augment, args.dataAugmentation.prob_distorsion)
    transform_image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    transform_mask = v2.Compose([v2.ToImage(), v2.Grayscale(num_output_channels=1), v2.ToDtype(torch.float32, scale=True)])

    path_train = find_folder_by_prefix(args.pathData, "tr")
    path_test = find_folder_by_prefix(args.pathData, "te")
    path_val = find_folder_by_prefix(args.pathData, "va")

    if path_train:
        remaining_images = [os.path.join(args.pathData, f) for f in os.listdir(args.pathData) if os.path.isfile(os.path.join(args.pathData, f)) and is_image_file(f)]
        remaining_masks = [os.path.join(args.pathMasks, f) for f in os.listdir(args.pathMasks) if os.path.isfile(os.path.join(args.pathMasks, f)) and is_image_file(f)]
        if remaining_images:
            print(colored(f"Warning: {len(remaining_images)} images still present in {args.pathData} (outside subfolders).", "yellow"))
        if remaining_masks:
            print(colored(f"Warning: {len(remaining_masks)} masks still present in {args.pathMasks} (outside subfolders).", "yellow"))

        print(colored(f"Train folder found : {path_train}. Loading images...", "green"))
        path_train_mask = os.path.join(args.pathMasks, os.path.basename(path_train))
        x_train, y_train = load_dataset(path_train, path_train_mask, args.trainSize, transform_image, transform_mask)

        if path_test:
            print(colored(f"Test folder found: {path_test}. Loading images...", "green"))
            path_test_mask = os.path.join(args.pathMasks, os.path.basename(path_test))
            x_test, y_test = load_dataset(path_test, path_test_mask, args.trainSize, transform_image, transform_mask)
            dataset_test = Dataset(x_test, y_test, None, args.processingArgs, args.floatPrecision)
            dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
        else:
            print(colored("Test folder not found, trying with test argument...", "yellow"))
            if args.test:
                print(colored(f"Splitting {args.test * 100}% for test data", "green"))
                test_sample_size = int(args.test * len(x_train))
                if test_sample_size <= 0:
                    raise ValueError(colored(f"Not enough data: found {len(x_train)} images, but test set is supposed to be {args.test * 100}%", "red"))
                x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_sample_size, shuffle=True, random_state=args.seed)
                dataset_test = Dataset(x_test, y_test, None, args.processingArgs, args.floatPrecision)
                dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
            else:
                print(colored("Test argument not set, skipping test phase...", "yellow"))

        if path_val:
            print(colored(f"Validation folder found: {path_val}", "green"))
            path_val_mask = os.path.join(args.pathMasks, os.path.basename(path_val))
            x_val, y_val = load_dataset(path_val, path_val_mask, args.trainSize, transform_image, transform_mask)
            dataloader_val = torch.utils.data.DataLoader(Dataset(x_val, y_val, None, args.processingArgs, args.floatPrecision), batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
        else:
            print(colored("Validation folder not found, trying with validation argument...", "yellow"))
            if args.validation:
                print(colored(f"Splitting {args.validation * 100}% for validation data", "green"))
                validation_sample_size = int(args.validation * len(x_train))
                if validation_sample_size <= 0:
                    raise ValueError(colored(f"Not enough data: found {len(x_train)} images, but validation set is supposed to be {args.validation * 100}%", "red"))
                x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_sample_size, shuffle=True, random_state=args.seed)
                dataloader_val = torch.utils.data.DataLoader(Dataset(x_val, y_val, None, args.processingArgs, args.floatPrecision), batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
            else:
                print(colored("Validation argument not set, skipping validation phase...", "yellow"))
        if dataloader_train is None:
            dataloader_train = torch.utils.data.DataLoader(Dataset(x_train, y_train, transform, args.processingArgs, args.floatPrecision), batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers, pin_memory=True)
    elif path_test:
        raise ValueError(colored("Test folder found, but no train folder detected. Please create a train folder.", "red"))
    elif path_val:
        raise ValueError(colored("Validation folder found, but no train folder detected. Please create a train folder.", "red"))
    else:
        list_images = [transform_image(img) for img in load_images_from_folder(args.pathData, args.trainSize)]
        list_masks = [transform_mask(mask) for mask in load_images_from_folder(args.pathMasks, args.trainSize)]
        if len(list_images) != len(list_masks):
            raise ValueError(colored(f"Mismatch: {len(list_images)} images vs {len(list_masks)} masks", "red"))
        return generate_split_data(list_images, list_masks, args, transform)
    return dataloader_train, dataloader_test, dataloader_val

def load_data(args):
    """
    Load and prepare the training, testing, and validation datasets based on the provided arguments.

    This function handles:
    - Detecting and validating the dataset structure (folders, filenames).
    - Applying transformations and augmentations.
    - Building DataLoaders for train, test, and validation datasets.
    - Logging dataset sizes while handling possible `None` cases safely (prints 0 if missing).

    Args:
        args (Args): An instance of the Args dataclass containing all necessary parameters such as paths, split ratios, augmentations, batch sizes, etc.

    Returns:
        tuple:
            - dataloader_train (torch.utils.data.DataLoader or None): DataLoader for the training set, or None if unavailable.
            - dataloader_test (torch.utils.data.DataLoader or None): DataLoader for the test set, or None if unavailable.
            - dataloader_val (torch.utils.data.DataLoader or None): DataLoader for the validation set, or None if unavailable.

    Raises:
        ValueError: If the folder structure is invalid, splits cannot be performed properly, or if there is a mismatch between images and masks.
    """
    dataloader_train, dataloader_test, dataloader_val = split_data_with_folders(args)
    print(colored(f"---------- DATA: ----------", "green"))
    print(colored(f" ˪ TRAIN: {len(dataloader_train.dataset) if dataloader_train else 0} images & masks", "green"))
    print(colored(f" ˪ TEST: {len(dataloader_test.dataset) if dataloader_test else 0} images & masks", "green"))
    print(colored(f" ˪ VALIDATION: {len(dataloader_val.dataset) if dataloader_val else 0} images & masks", "green"))
    return dataloader_train, dataloader_test, dataloader_val
    