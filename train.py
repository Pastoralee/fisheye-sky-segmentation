import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import KFold
from metrics import compute_metrics
import torch.optim.lr_scheduler as lr_scheduler
from utils import CutMix
from termcolor import colored
import time
from datetime import timedelta

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.

    This loss is designed to address class imbalance by down-weighting
    easy examples and focusing training on hard negatives.

    Args:
        gamma (float, optional): Focusing parameter that adjusts the rate
            at which easy examples are down-weighted. Default is 2.0.
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.gamma = gamma

    def focal_loss(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce)  # Probability of the correct class
        focal = (1 - pt) ** self.gamma * bce
        return focal

    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)

class DiceLoss(nn.Module):
    """
    Dice Loss implementation for segmentation tasks.

    Dice Loss measures the overlap between the predicted and ground truth masks.

    Note:
        Inputs should ideally be probabilities (after sigmoid), but this implementation
        expects raw logits and does not apply sigmoid internally.
    """

    def __init__(self):
        super().__init__()

    def dice_loss(self, inputs, targets):
        smooth = 1e-5
        #inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets)

class SAMLoss(nn.Module):
    """
    Segment Anything Model Loss (SAMLoss) combining Focal Loss and Dice Loss.

    This loss function dynamically balances Focal Loss and Dice Loss,
    aiming to improve segmentation performance on imbalanced datasets.

    Args:
        alpha (float, optional): Weight factor for balancing Focal Loss and Dice Loss.
            Default is 0.05.
        gamma (float, optional): Focusing parameter for Focal Loss. Default is 2.0.
    """

    def __init__(self, alpha=0.05, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.alpha * focal + (1 - self.alpha) * dice

class CustomLoss(nn.Module):
    """
    Custom combined loss of BCEWithLogitsLoss and Dice Loss.

    Args:
        alpha (float, optional): Weight factor for balancing BCE and Dice Loss.
            Default is 0.05.
    """

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.alpha * bce + (1 - self.alpha) * dice

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.

    Args:
        patience (int, optional): Number of epochs to wait after last improvement.
            Default is 4.
        delta (float, optional): Minimum change in validation loss to qualify as improvement.
            Default is 0.
        path (str, optional): Directory to save model checkpoints. Default is 'checkpoint'.
    """

    def __init__(self, patience=4, delta=0, path='checkpoint'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch, preprocessing_code):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, preprocessing_code)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, preprocessing_code)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, preprocessing_code):
        print(colored(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...', 'green'))
        torch.save({'model_state_dict': model.state_dict(), 'pre_processing': preprocessing_code}, f'{self.path}/epoch_{epoch+1}.pt')
        self.val_loss_min = val_loss

def weight_histograms_conv2d(writer, step, weights, name):
    """
    Logs Conv2D layer weight histograms to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard summary writer.
        step (int): Current training step or epoch.
        weights (torch.Tensor): Weight tensor of the Conv2D layer.
        name (str): Name of the layer for logging.
    """

    weights = weights.view(weights.shape[0], -1)
    writer.add_histogram(name, weights, global_step=step, bins='tensorflow')

def weight_histograms_linear(writer, step, weights, name):
    """
    Logs Linear layer weight histograms to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard summary writer.
        step (int): Current training step or epoch.
        weights (torch.Tensor): Weight tensor of the Linear layer.
        name (str): Name of the layer for logging.
    """

    flattened_weights = weights.detach().cpu().numpy().flatten()
    writer.add_histogram(name, flattened_weights, global_step=step, bins='tensorflow')

def weight_histograms(writer, step, model):
    """
    Logs all Conv2D and Linear layer weight histograms from the model to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard summary writer.
        step (int): Current training step or epoch.
        model (nn.Module): PyTorch model.
    """

    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                weight_histograms_conv2d(writer, step, layer.weight, name)
            elif isinstance(layer, nn.Linear):
                weight_histograms_linear(writer, step, layer.weight, name)

def train_model(model, args, preprocessing_code, train_loader, test_loader, val_loader):
    """
    Orchestrates the training process by initializing resources and running the training loop.

    Args:
        model (nn.Module): The PyTorch model to train.
        args (Namespace): Configuration and hyperparameters.
        preprocessing_code (int): Describes the Pre Processing configuration
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Testing data loader (optional).
        val_loader (DataLoader): Validation data loader (optional).
    """

    writer = SummaryWriter(log_dir=f'{args.pathSave}/logs')
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=4, path=f'{args.pathSave}')
    iou, avg_epoch_time = run_training(model, args, preprocessing_code, train_loader, test_loader, val_loader, writer, early_stopping, scaler)
    writer.close()
    return iou, avg_epoch_time

def run_training(model, args, preprocessing_code, train_loader, test_loader, val_loader, writer, early_stopping, scaler):
    """
    Executes the main training loop with optional CutMix augmentation, early stopping, 
    and model checkpointing. Also logs metrics and losses to TensorBoard.

    Args:
        model (nn.Module): The PyTorch model to train.
        args (Namespace): Configuration and hyperparameters.
        preprocessing_code (int): Describes the Pre Processing configuration
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Testing data loader.
        val_loader (DataLoader): Validation data loader.
        writer (SummaryWriter): TensorBoard summary writer.
        early_stopping (EarlyStopping): Early stopping handler.
        scaler (GradScaler): AMP gradient scaler for mixed-precision training.
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cutmix = None
    if args.dataAugmentation.cut_mix:
        cutmix = CutMix(args.dataAugmentation.cut_mix.beta1, args.dataAugmentation.cut_mix.beta2)
    best_iou = 0
    total_time = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        print(colored(f"---------- Epoch {epoch+1}/{args.epochs} ----------", "green"))
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader):
            if cutmix and torch.rand(1).item() < args.dataAugmentation.cut_mix.cutmix_prob:
                images, masks = cutmix(images, masks)
            images, masks = images.to(device=args.device, dtype=args.floatPrecision), masks.to(device=args.device, dtype=args.floatPrecision)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=args.device.type, dtype=args.floatPrecision):
                logits = model(images)
                loss = args.criterion(logits, masks)
            scaler.scale(loss).backward()
            #scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #weight_histograms(writer, epoch, model)
            train_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(colored(f"Training Loss: {avg_train_loss}", "green"))
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)

        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        elapsed_time = timedelta(seconds=int(end_time - start_time))
        print(colored(f"Epoch {epoch+1} training time: {elapsed_time}", "green"))
        total_time += end_time - start_time
        
        model.eval()

        #Test phase
        if test_loader is not None:
            accuracy, precision, recall, f1, iou = compute_metrics(test_loader, model, args.device, args.floatPrecision, args.trainSize, writer, epoch)
            print(colored(f"   ˪ Accuracy: {accuracy * 100:.2f}%", "green"))
            print(colored(f"   ˪ Precision: {precision * 100:.2f}%", "green"))
            print(colored(f"   ˪ Recall: {recall * 100:.2f}%", "green"))
            print(colored(f"   ˪ F1 Score: {f1 * 100:.2f}%", "green"))
            print(colored(f"   ˪ IoU: {iou * 100:.2f}%", "green"))
            if iou > best_iou:
                if val_loader is None:
                    print(colored(f"IoU improved: {best_iou * 100:.2f}% --> {iou * 100:.2f}%. Saving model ...", "green"))
                    torch.save({'model_state_dict': model.state_dict(), 'pre_processing': preprocessing_code}, f'{args.pathSave}/epoch_{epoch+1}.pt')
                best_iou = iou

        # Validation phase
        if val_loader is not None:
            val_loss = 0
            with torch.no_grad():
                for images, masks in tqdm(val_loader):
                    images, masks = images.to(device=args.device, dtype=torch.float), masks.to(device=args.device, dtype=torch.float)
                    with torch.amp.autocast(device_type=args.device.type, dtype=args.floatPrecision):
                        logits = model(images)
                        loss = args.criterion(logits, masks)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(colored(f"Validation Loss: {avg_val_loss}", "green"))

            # Check early stopping
            if test_loader is not None:
                early_stopping(avg_val_loss, model, epoch, preprocessing_code)
            else:
                early_stopping(avg_val_loss, model, epoch, preprocessing_code)
            if early_stopping.early_stop:
                print(colored(f"Early stopping counter reached: {early_stopping.patience}. Stopping training...", "yellow"))
                break

        #Save model
        else:
            print(colored(f"Saving model ...", "green"))
            torch.save({'model_state_dict': model.state_dict(), 'pre_processing': preprocessing_code}, f'{args.pathSave}/epoch_{epoch+1}.pt')

    avg_epoch_time = timedelta(seconds=int(total_time / (epoch + 1)))
    return best_iou, avg_epoch_time