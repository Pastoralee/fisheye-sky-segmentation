from dataclasses import dataclass, field
import torch
from data import load_data
from train import SAMLoss, train_model, CustomLoss
import utils as ut
from termcolor import colored
import segmentation_models_pytorch as smp
from Networks.efficientnet import EfficientNetBackbone
import argparse

@dataclass
class PreProcessing():
    """
    Dataclass for configuring preprocessing steps applied to input images.
    Each attribute enables or disables a specific preprocessing operation.
    """
    
    base_img: bool = True
    dark_channel: bool = False
    depth_estimation: bool = False
    hue_disparity: bool = False
    color_saturation: bool = False
    contrast_energy: bool = False
    exponential_transform: bool = False
    color_gradient: bool = False
    hsv: bool = False
    kmeans: bool = False
    
@dataclass
class CutMix():
    beta1: int = 1
    beta2: int = 1
    cutmix_prob: float = 0.5

@dataclass
class DataAugmentation():
    cut_mix: CutMix = field(default_factory=lambda: CutMix())
    prob_flip: float = 0.5
    prob_augment: float = 0.5
    prob_distorsion: float = 0.5
    
@dataclass
class Args():
    """
    Dataclass that holds all arguments required for the training pipeline.

    Attributes:
        pathData (str): Path to the dataset images.
        pathMasks (str): Path to the dataset masks.
        pathSave (str): Directory to save model checkpoints and logs.
        batchSize (int): Batch size for data loading.
        epochs (int): Number of training epochs.
        encoder (str): Encoder backbone to use (e.g., efficientnet-b7, efficientnet-b5, etc.).
        decoder (str): Decoder architecture to use (e.g., UnetPlusPlus, Unet, etc.).
        processingArgs (PreProcessing or None): PreProcessing instance for image preprocessing.
        dataAugmentation (DataAugmentation): Data augmentation parameters.
        pathPreTrainedModel (str or None): Path to the pretrained model checkpoint.
        lr (float): Learning rate used for optimizer.
        weightDecay (float): Weight decay (L2 regularization) used in optimizer.
        trainSize (int): Target resolution (height and width) to which training images are resized.
        test (float or None): Percentage of the dataset allocated to testing, if no test folder is found.
        validation (float or None): Percentage of the dataset allocated to validation, if no validation folder is found.
        device (torch.device): Device used for computation (e.g., CPU or CUDA-enabled GPU).
        numWorkers (int): Number of worker threads used by the data loaders.
        seed (int): Random seed for reproducibility of training.
        criterion (torch.nn.Module): Loss function used during training.
        floatPrecision (str): Desired numerical precision (e.g., 'float32', 'bfloat16').
    """

    pathData: str
    pathMasks: str
    pathSave: str
    batchSize: int
    epochs: int
    encoder : str = 'efficientnet-b7'
    decoder : str = 'UnetPlusPlus'
    processingArgs: PreProcessing = None
    dataAugmentation: DataAugmentation = field(default_factory=lambda: DataAugmentation())
    pathPreTrainedModel: str = None
    lr: float = 4e-4
    weightDecay: float = 1e-3
    trainSize: int = 1024
    test: float = None #will check for test folder
    validation: float = None #will check for validation folder
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    numWorkers: int = 1
    seed: int = 42
    criterion: torch.nn.Module = field(default_factory=lambda: CustomLoss(alpha=0.05))
    floatPrecision: str = 'bfloat16'

def parse_args():
    """
    Parse command-line arguments for configuring the segmentation training pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a segmentation model with configurable hyperparameters.")

    # Required arguments
    parser.add_argument('--pathData', type=str, required=True, help='Path to the dataset images.')
    parser.add_argument('--pathMasks', type=str, required=True, help='Path to the dataset masks.')
    parser.add_argument('--pathSave', type=str, required=True, help='Directory to save model checkpoints and logs.')

    # Model architecture arguments
    parser.add_argument('--encoder', type=str, default='efficientnet-b7', help='Encoder backbone to use (e.g., efficientnet-b7, efficientnet-b5, etc.).')
    parser.add_argument('--decoder', type=str, default='UnetPlusPlus', help='Decoder architecture to use (e.g., UnetPlusPlus, Unet, etc.).')

    # Args
    parser.add_argument('--batchSize', type=int, default=4, help='Batch size for data loading.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=4e-4, help='Learning rate used for optimizer.')
    parser.add_argument('--weightDecay', type=float, default=1e-3, help='Weight decay (L2 regularization) used in optimizer.')
    parser.add_argument('--trainSize', type=int, default=1024, help='Target resolution (height and width) to which training images are resized.')
    parser.add_argument('--test', type=float, default=None, help='Percentage of the dataset allocated to testing, if no test folder is found.')
    parser.add_argument('--validation', type=float, default=None, help='Percentage of the dataset allocated to validation, if no validation folder is found.')
    parser.add_argument('--numWorkers', type=int, default=1, help='Number of worker threads used by the data loaders.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility of training.')
    parser.add_argument('--floatPrecision', type=str, default='bfloat16', help="Desired numerical precision (e.g., 'float32', 'bfloat16').")
    parser.add_argument('--pathPreTrainedModel', type=str, default=None, help='Path to the pretrained model checkpoint.')
    parser.add_argument('--alpha', type=float, default=0.05, help='Alpha parameter for CustomLoss.')

    # PreProcessing
    parser.add_argument('--base_img', action='store_true', help='Use base image in preprocessing.')
    parser.add_argument('--dark_channel', action='store_true', help='Use dark channel in preprocessing.')
    parser.add_argument('--depth_estimation', action='store_true', help='Use depth estimation in preprocessing.')
    parser.add_argument('--hue_disparity', action='store_true', help='Use hue disparity in preprocessing.')
    parser.add_argument('--color_saturation', action='store_true', help='Use color saturation in preprocessing.')
    parser.add_argument('--contrast_energy', action='store_true', help='Use contrast energy in preprocessing.')
    parser.add_argument('--exponential_transform', action='store_true', help='Use exponential transform in preprocessing.')
    parser.add_argument('--color_gradient', action='store_true', help='Use color gradient in preprocessing.')
    parser.add_argument('--hsv', action='store_true', help='Use HSV in preprocessing.')
    parser.add_argument('--kmeans', action='store_true', help='Use kmeans in preprocessing.')

    # DataAugmentation
    parser.add_argument('--prob_flip', type=float, default=0.5, help='Probability of flipping during augmentation.')
    parser.add_argument('--prob_augment', type=float, default=0.5, help='Probability of augmentation.')
    parser.add_argument('--prob_distorsion', type=float, default=0.5, help='Probability of distortion during augmentation.')

    # CutMix
    parser.add_argument('--no_cutmix', action='store_true', help='Disable CutMix augmentation.')
    parser.add_argument('--cutmix_beta1', type=int, default=1, help='Beta1 parameter for CutMix.')
    parser.add_argument('--cutmix_beta2', type=int, default=1, help='Beta2 parameter for CutMix.')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability of CutMix.')

    return parser.parse_args()

def build_args_from_parsed(parsed):
    """
    Build the Args dataclass (and nested dataclasses) from parsed command-line arguments.

    Args:
        parsed (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Args: Fully constructed Args dataclass instance.
    """

    cutmix = None if getattr(parsed, "no_cutmix", False) else CutMix(
        beta1=parsed.cutmix_beta1,
        beta2=parsed.cutmix_beta2,
        cutmix_prob=parsed.cutmix_prob
    )
    data_aug = DataAugmentation(
        cut_mix=cutmix,
        prob_flip=parsed.prob_flip,
        prob_augment=parsed.prob_augment,
        prob_distorsion=parsed.prob_distorsion
    )
    preprocessing_flags = [
        parsed.base_img, parsed.dark_channel, parsed.depth_estimation, parsed.hue_disparity,
        parsed.color_saturation, parsed.contrast_energy, parsed.exponential_transform,
        parsed.color_gradient, parsed.hsv, parsed.kmeans
    ]
    preprocessing = None
    if any(preprocessing_flags):
        preprocessing = PreProcessing(
            base_img=parsed.base_img,
            dark_channel=parsed.dark_channel,
            depth_estimation=parsed.depth_estimation,
            hue_disparity=parsed.hue_disparity,
            color_saturation=parsed.color_saturation,
            contrast_energy=parsed.contrast_energy,
            exponential_transform=parsed.exponential_transform,
            color_gradient=parsed.color_gradient,
            hsv=parsed.hsv,
            kmeans=parsed.kmeans
        )
    args = Args(
        pathData=parsed.pathData,
        pathMasks=parsed.pathMasks,
        pathSave=parsed.pathSave,
        batchSize=parsed.batchSize,
        epochs=parsed.epochs,
        encoder=parsed.encoder,
        decoder=parsed.decoder,
        processingArgs=preprocessing,
        dataAugmentation=data_aug,
        pathPreTrainedModel=parsed.pathPreTrainedModel,
        lr=parsed.lr,
        weightDecay=parsed.weightDecay,
        trainSize=parsed.trainSize,
        test=parsed.test,
        validation=parsed.validation,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        numWorkers=parsed.numWorkers,
        seed=parsed.seed,
        criterion=CustomLoss(alpha=parsed.alpha),
        floatPrecision=parsed.floatPrecision
    )
    return args

def main(args: Args):
    """
    Main function to initialize arguments, prepare data loaders,
    build the model, and start the training loop.

    Args:
        args (Args): Configuration dataclass containing all training parameters.
    """

    ut.test_args_parameters(args)
    args.floatPrecision = getattr(torch, args.floatPrecision)
    ut.fix_seed(seed=args.seed)
    dataloader_train, dataloader_test, dataloader_val = load_data(args)

    input_channels = 3
    if args.processingArgs:
        input_channels = ut.compute_input_channels(args.processingArgs)
    print(colored(f"Input channels: {input_channels}", "green"))

    model_class = getattr(smp, args.decoder)
    model = model_class(
        encoder_name=args.encoder,
        encoder_weights="advprop" if "efficientnet" in args.encoder else "imagenet",
        in_channels=input_channels,
        classes=1,
        activation='sigmoid'
    )

    if args.pathPreTrainedModel:
        is_swav, state = ut.load_correct_state(args.pathPreTrainedModel, args.device, input_channels)
        if is_swav:
            model_proto = EfficientNetBackbone(model_name="efficientnet_b7", normalize=True, hidden_mlp=2048, output_dim=128, nmb_prototypes=200)
            ut.load_swav_pretrained_encoder_weights(model.encoder, model_proto, state)
        else:
            model.load_state_dict(state)
    
    ut.count_parameters(model)
    model = model.to(args.device)
    iou, avg_epoch_time = train_model(model, args, ut.encode_preprocessing(args.processingArgs), dataloader_train, dataloader_test, dataloader_val)

if __name__ == "__main__":
    parsed = parse_args()
    args = build_args_from_parsed(parsed)
    main(args)