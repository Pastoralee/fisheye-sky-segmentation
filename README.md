# Pixel-Wise Sky-Obstacle Segmentation in Fisheye Imagery Using Deep Learning and Gradient Boosting

This repository contains the training and inference scripts for the paper:  
**“Pixel-Wise Sky-Obstacle Segmentation in Fisheye Imagery Using Deep Learning and Gradient Boosting”**.  

It enables segmentation of sky and obstacles in fisheye (ultra-wide angle) images, producing binary masks for further analysis.

---

## Repository Contents

- **`data.py`**  
  Utilities for loading, preprocessing, and augmenting image/mask data for training and validation.

- **`GSV_dataset_extract.ipynb`**  
  Jupyter notebook for downloading and generating fisheye images from Google Street View panoramas.

- **`GSV_utils.py`**  
  Helper functions for downloading, assembling, and projecting Google Street View panoramas.

- **`inference.py`**  
  Functions for model inference, including multi-scale processing and LightGBM meta-model post-processing.

- **`LGBM.ipynb`**  
  Jupyter notebook for training and interpreting a LightGBM meta-model using features from segmentation predictions.

- **`main.ipynb`**  
  Interactive notebook for running the main training pipeline and performing inference.

- **`main.py`**  
  Main script for configuring, training, and evaluating segmentation models via CLI.

- **`metrics.py`**  
  Implements evaluation metrics (e.g., Accuracy, IoU) for segmentation results, including circular masking.

- **`requirements.txt`**  
  Lists all Python dependencies required to run the project.

- **`train.py`**  
  Contains the training loop, custom loss functions, and model checkpointing.

- **`utils.py`**  
  General utility functions for preprocessing, augmentation, and reproducibility.

- **`networks/efficientnet.py`**  
  Implementation of the EfficientNet architecture used for SwAV pretraining.

---

## Getting Started

### 1. Install Dependencies

All dependencies are listed in `requirements.txt`.  
You can either:  
- Install them manually using `pip install -r requirements.txt`, **or**  
- Automatically set up the environment by running:  

```bash
python setup_env.py
```
This script creates a virtual environment and installs all required packages, including Jupyter Notebook.
**Note**: The project was developed with Python 3.12. Other versions may work, but ensure compatibility with your setup.

### 2. Activate the Virtual Environment

- On Windows:
```bash
./venv/Scripts/activate
```

- On Linux:
```bash
source venv/bin/activate
```

### 3. Run the program

- To use jupyter notebook:
```bash
jupyter-notebook
```

- To run the CLI:
```bash
python main.py
```

- View all available arguments:
```bash
python main.py -h
```

## Dataset Structure for Training

You have two options for organizing your dataset:

### Option 1: Simple Folder Split

- Place all images in one folder, and all masks in another.
- Specify a split ratio (e.g., `0.8` for 80% training data). The program will randomly divide the dataset.

### Option 2: Pre-structured Folders

Organize images and masks as follows:
```
Images/
├── train/
├── test/
└── validation/

Masks/
├── train/
├── test/
└── validation/
```
Make sure folder names are correct for the program to recognize them.

---

## Dataset and Trained Models

This repository is accompanied by a dataset and trained model weights, available for download:  

### Download
[Download Dataset & Trained Models (Google Drive)](https://drive.google.com/drive/folders/1PnKakX55PCW72MTsl-TXBb6TM5EOUejA?usp=drive_link)

---

### Contents
The Google Drive folder includes:  

- **Segmentation masks** for both base models and meta-models.  
- **Validation and test images** for evaluation.  
- **Trained model weights** for base models and LightGBM meta-models.  

**Training images are NOT included** because they are sourced from Google Street View and cannot be redistributed directly due to licensing restrictions.  

Each image filename includes the **latitude and longitude** coordinates so you can fetch the corresponding panoramas yourself.

---

### Reconstruct Training Data
To obtain the training images:  
- Use the provided **`GSV_extract_dataset.ipynb`** notebook to fetch and process images directly from Google Street View.  
- Alternatively, call the utility functions in **`GSV_utils.py`** to programmatically download and assemble the images using the included panorama IDs.  

> **Note:** Downloading images requires a Google Street View API key and must comply with Google’s Terms of Service.

---

## Citation

If you use this code or dataset, please cite:
