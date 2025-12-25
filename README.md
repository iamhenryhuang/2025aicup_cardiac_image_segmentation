# Cardiac Image Segmentation

3D Residual U-Net implementation for AICUP cardiac muscle segmentation. The project includes two training variants and an ensemble inference pipeline.

## Setup

- **Dependencies**: `torch`, `torchvision`, `nibabel`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`, `scipy`
- **Environment**: Google Colab with GPU recommended
- **Data format**: 3D NIfTI files (`.nii` / `.nii.gz`), typically `512 × 512 × 300+`

## Data Structure

```
MyDrive/aicup_data/
├── training_image/     # Training images (patient0001.nii.gz)
├── training_label/     # Ground truth labels (patient0001_gt.nii.gz)
├── testing_image/      # Test images
└── aicup_results/      # Trained models and submission files
```

## Training

### train_v1.ipynb: Fine-tuning with Pre-trained Weights

- **Model**: 3D Residual U-Net, `base_channels=32` (~23M parameters)
- **Input resolution**: `320 × 320 × 160` (downsampled from original)
- **Training strategy**: Patch-based sampling
  - Patch size: `(112, 112, 112)`
  - 4 patches per volume
  - 70% foreground-biased sampling
- **Augmentation**: Rotation (±15°), flips, Gaussian noise, brightness/contrast adjustment
- **Loss**: `0.4 × CE + 0.3 × Dice + 0.3 × Boundary` with class weights `[0.5, 1.5, 1.5, 1.5]`
- **Training**: Adam (lr=1e-4, weight_decay=1e-4), ReduceLROnPlateau, 150 epochs
- **Best model selection**: Combined Score = (Dice + IoU) / 2

### train_v2.ipynb: Larger Model from Scratch

- **Model**: 3D Residual U-Net, `base_channels=48` (~52M parameters)
- **Input resolution**: `384 × 384 × 192`
- **Training strategy**: Same patch-based approach with gradient accumulation
  - Effective batch size: 16 (batch_size=4, grad_accum_steps=4)
- **Augmentation**: Enhanced with 3D elastic deformation (Gaussian-smoothed displacement fields)
- **Loss**: `0.2 × CE + 0.45 × Dice + 0.35 × Boundary`
- **Training**: Same optimizer/scheduler settings, 150 epochs

## Inference

### inference.ipynb: Ensemble Prediction

Combines two trained models using weighted soft voting:

- **Model 1**: `base_channels=32`, input `320×320×160`
- **Model 2**: `base_channels=48`, input `384×384×192`
- **Ensemble**: `0.6 × probs_model1 + 0.4 × probs_model2`

**Inference pipeline**:
1. Normalize images using 1-99 percentile clipping
2. Resample to each model's target resolution
3. Sliding window inference (patch size `160×160×160`, overlap=0.8)
4. Optional TTA: original + horizontal flip (enabled for Model 1 by default)
5. Resample probabilities back to original resolution
6. Weighted ensemble and argmax
7. Post-processing: 3D Largest Connected Component (LCC) for classes 1-3 (min_size=300)
8. Save predictions and create submission zip file

**Optimizations**: Reduced memory copies, optimized resampling, periodic GPU cache clearing.
