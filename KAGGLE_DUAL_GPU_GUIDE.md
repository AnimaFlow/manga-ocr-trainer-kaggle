# Kaggle Dual GPU Training Guide

## Prerequisites
1. **Kaggle Account**: You need a Kaggle account with GPU quota
2. **WANDB Account**: Set up Weights & Biases for experiment tracking
3. **Dataset**: Manga109 dataset (or your custom manga dataset)

## Step 1: Create Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com)
2. Click "New Notebook"
3. Select "GPU" accelerator (choose P100 or T4 x2 for dual GPU)
4. Set the language to Python

## Step 2: Upload Your Code

Upload the following files to your Kaggle notebook:
- `training/train.py` (modified for dual GPU)
- `training/dataset.py`
- `training/get_model.py`
- `training/metrics.py`
- `envpath/env.py`
- `setup_kaggle_dual_gpu.py`
- `keep_alive.py`
- `requirements.txt`

## Step 3: Set Up Data

### Option A: Use Manga109 Dataset
```python
# In Kaggle notebook
!pip install kaggle
!kaggle datasets download -d your-dataset-name
!unzip your-dataset-name.zip -d /kaggle/input/manga109-dataset/
```

### Option B: Upload Custom Dataset
1. Create a Kaggle dataset with your manga images
2. Mount it in the notebook

## Step 4: Configure Environment

1. **Set WANDB API Key**:
   - Go to Kaggle notebook settings
   - Add your WANDB API key as a secret named "WANDB_API_KEY"

2. **Update Paths** in `envpath/env.py`:
```python
# For Kaggle
MANGA109_ROOT = Path("/kaggle/input/manga109-dataset")
DATA_SYNTHETIC_ROOT = Path("/kaggle/input/synthetic-data")  # If using synthetic data
TRAIN_ROOT = Path("/kaggle/working/training_output")
FONTS_ROOT = Path("/kaggle/input/fonts")  # If using custom fonts
```

## Step 5: Install Dependencies

```python
# Run this in your Kaggle notebook
!pip install -r requirements.txt
```

## Step 6: Run Setup Script

```python
# Run the setup script
!python setup_kaggle_dual_gpu.py
```

## Step 7: Start Training

### Option A: Use the Setup Script
```python
!python /kaggle/working/train_dual_gpu.py
```

### Option B: Manual Training
```python
from training.train import run

# Train with dual GPU
run(
    run_name="kaggle_dual_gpu_experiment",
    encoder_name="facebook/deit-tiny-patch16-224",
    decoder_name="cl-tohoku/bert-base-japanese-char-v2",
    max_len=300,
    num_decoder_layers=2,
    batch_size=32,  # Per GPU batch size
    num_epochs=8,
    fp16=True,
    use_dual_gpu=True
)
```

## Step 8: Prevent Timeout (Important!)

Run this in a separate Kaggle cell to prevent timeout:

```python
# Keep the session alive
!python keep_alive.py
```

## Key Features of Dual GPU Setup

### Automatic GPU Detection
- Detects available GPUs automatically
- Adjusts batch size based on GPU count
- Falls back to single GPU if only one is available

### Memory Optimization
- Uses FP16 mixed precision
- Optimized data loading with pinned memory
- Gradient accumulation for better GPU utilization

### Distributed Training
- Uses NCCL backend for efficient GPU communication
- Automatic process group management
- Proper cleanup after training

## Monitoring Training

### Weights & Biases
- All metrics are logged to WANDB
- Monitor loss, accuracy, and GPU usage in real-time
- Compare experiments across different configurations

### GPU Usage
```python
# Check GPU usage during training
!nvidia-smi
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size
   - Enable gradient accumulation
   - Use FP16 more aggressively

2. **Slow Training**:
   - Increase dataloader workers
   - Use pinned memory
   - Optimize data preprocessing

3. **Kaggle Timeout**:
   - Use the keep_alive.py script
   - Run shorter experiments
   - Save checkpoints frequently

### Performance Tips

1. **Batch Size**: Start with 32 per GPU, adjust based on memory
2. **Workers**: Use 2-4 dataloader workers
3. **Mixed Precision**: Always use FP16 for faster training
4. **Data Format**: Use efficient data formats (parquet, etc.)

## Expected Performance

- **Dual T4 GPUs**: ~2x speedup vs single GPU
- **Dual P100 GPUs**: ~1.8x speedup vs single GPU
- **Memory Usage**: ~8-12GB per GPU depending on model size

## Saving and Loading Models

```python
# The training script automatically saves checkpoints
# Models are saved to /kaggle/working/training_output/

# To download models after training:
from IPython.display import FileLink
FileLink('/kaggle/working/training_output/checkpoint-best/pytorch_model.bin')
```

## Next Steps

1. Experiment with different model architectures
2. Try different batch sizes and learning rates
3. Use synthetic data generation for better results
4. Fine-tune on your specific manga dataset

Happy training! 🎯
