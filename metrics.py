import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm

def create_circle(size):
    """
    Creates a binary circular mask of the given size.

    Args:
        size (int): Diameter of the circle (also image size).

    Returns:
        torch.Tensor: Binary mask of shape (size, size) with values 0 or 1.
    """

    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    x, y = torch.meshgrid(x, y, indexing='ij')
    dist2ori = torch.sqrt(x**2 + y**2)
    mask = torch.zeros((size, size))
    mask[dist2ori <= 1] = 1
    return mask

def compute_metrics(dataloader, model, device, floatPrecision, trainSize, writer, epoch):
    """
    Computes evaluation metrics on a given dataloader.

    Applies a circular mask to limit evaluation area, thresholds predictions,
    and computes Accuracy, Precision, Recall, F1 Score, and IoU.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        model (torch.nn.Module): Model to evaluate.
        device (torch.device): Computation device.
        floatPrecision (torch.dtype): Float precision to use during inference.
        trainSize (int): Input image size.
        writer (torch.utils.tensorboard.SummaryWriter): Tensorboard writer for logging.
        epoch (int): Current epoch number.

    Returns:
        tuple: (accuracy, precision, recall, f1, iou) scores.
    """
    
    all_preds = []
    all_labels = []
    mask = create_circle(size=trainSize).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W) ==> (B, C, H, W)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            batch_size, _, height, width = images.shape
            images, labels = images.to(device=device, dtype=floatPrecision), labels.to(device=device, dtype=floatPrecision)
            with torch.amp.autocast(device_type=device.type, dtype=floatPrecision):
                outputs = model(images)
            batch_mask = mask.expand(batch_size, 1, height, width) # (B, 1, H, W) ==> (B, C, H, W)
            masked_preds = (outputs > 0.5).float() * batch_mask
            masked_labels = (labels > 0.5).float() * batch_mask
            valid_indices = batch_mask.flatten(start_dim=1) == 1
            preds_flat = torch.masked_select(masked_preds.flatten(start_dim=1), valid_indices)
            labels_flat = torch.masked_select(masked_labels.flatten(start_dim=1), valid_indices)
            all_preds.append(preds_flat)
            all_labels.append(labels_flat)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    iou = jaccard_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    writer.add_scalar('Metrics/Accuracy', accuracy, epoch)
    writer.add_scalar('Metrics/Precision', precision, epoch)
    writer.add_scalar('Metrics/Recall', recall, epoch)
    writer.add_scalar('Metrics/F1 Score', f1, epoch)
    writer.add_scalar('Metrics/IoU', iou, epoch)

    return accuracy, precision, recall, f1, iou