# Dataset
https://drive.google.com/file/d/1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE/view

It is Task 9 of Medical Segmentation Decathlon

# spleen-segmentation
# Spleen Segmentation using U-Net

This project implements a U-Net deep learning model for automated spleen segmentation in medical CT scans. The model is designed to work with the Medical Segmentation Decathlon Task 9 (Spleen) dataset.

## Features

- **U-Net Architecture**: Complete implementation of the U-Net model with encoder-decoder structure
- **3-Channel Input**: Uses adjacent CT slices (previous, current, next) for better context
- **Combined Loss Function**: Combines Binary Cross-Entropy and Dice Loss for improved training
- **Data Augmentation**: Includes rotation, translation, and horizontal flip augmentations
- **Comprehensive Evaluation**: Provides Dice score, precision, recall, F1-score, and confusion matrix
- **Test Set Predictions**: Generates 3D segmentation predictions for test volumes

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
nibabel>=3.2.0
numpy>=1.21.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## Dataset Structure

The code expects the following directory structure (Medical Segmentation Decathlon format):

```
Task09_Spleen/
├── imagesTr/          # Training images (.nii or .nii.gz)
├── labelsTr/          # Training labels (.nii or .nii.gz)
├── imagesTs/          # Test images (.nii or .nii.gz)
└── labelsTs/          # Test labels (optional, .nii or .nii.gz)
```

## Key Components

### Model Architecture
- **Encoder**: 4 levels of double convolution blocks with max pooling
- **Decoder**: 3 levels of upsampling with skip connections
- **Input**: 3-channel (adjacent slices) of size 512x512
- **Output**: Single channel binary segmentation mask

### Data Processing
- **CT Windowing**: Clips values to [-100, 400] HU for optimal spleen visualization
- **Normalization**: Normalizes pixel values to [0, 1] range
- **Slice Selection**: Only includes slices containing spleen tissue for training
- **Multi-channel Input**: Creates 3-channel input using adjacent slices

### Loss Function
- **Combined Loss**: Weighted combination of BCE and Dice Loss
- **BCE Weight**: 0.5 (configurable)
- **Dice Loss**: Includes smoothing factor to handle empty predictions

## Usage

### Training

```python
# Modify the base_dir path in the train_model() function
base_dir = "/path/to/your/Task09_Spleen"

# Run training
python spleen_segmentation.py
```

### Key Parameters

- **Batch Size**: 8 (adjust based on GPU memory)
- **Learning Rate**: 1e-4 with ReduceLROnPlateau scheduler
- **Epochs**: 25 (early stopping based on validation Dice score)
- **Train/Val Split**: 80/20

### Output Files

The training process generates:
- `best_model.pth`: Best model weights based on validation Dice score
- `val_epoch_X/`: Validation visualizations for each epoch
- `test_results/`: Test set evaluation results and visualizations
- `test_predictions/`: 3D NIfTI predictions for test volumes

## Model Performance

The model is evaluated using multiple metrics:
- **Dice Coefficient**: Primary metric for segmentation overlap
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance

## Data Augmentation

Training includes the following augmentations:
- Random rotation (±10 degrees)
- Random translation (±10% of image size)
- Random horizontal flip

## Technical Details

### Memory Optimization
- Uses `pin_memory=True` for faster GPU transfer
- Implements efficient data loading with multiple workers
- Only loads slices containing spleen tissue for training

### Error Handling
- Robust file loading with corruption detection
- Automatic fallback for corrupted files
- Comprehensive logging of dataset statistics

### Visualization
- Sample predictions during validation
- Confusion matrix generation
- Training progress tracking


### File Matching
The code automatically matches image and label files based on filename (excluding extensions). Files with `.nii` or `.nii.gz` extensions are supported.

## Customization

### Modifying the Model
- Change `in_channels` and `out_channels` in UNet constructor
- Adjust network depth by modifying encoder/decoder layers
- Experiment with different activation functions

### Loss Function Tuning
- Adjust `bce_weight` parameter in CombinedLoss
- Modify Dice loss smoothing factor
- Add focal loss for class imbalance

### Data Preprocessing
- Modify CT windowing range for different anatomical structures
- Adjust normalization strategy
- Change slice selection criteria

## License

This project is available for educational and research purposes. Please check the original dataset license terms before commercial use.
