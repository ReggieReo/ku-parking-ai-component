# Google Colab Training Guide 🚀

This guide will help you train your YOLO parking spot detection model on Google Colab with free GPU access.

## 📋 Prerequisites

1. **Google Account** - You need a Google account to use Colab
2. **Dataset Prepared** - Your parking spot dataset should be ready with images and labels
3. **Google Drive** (Recommended) - Upload your dataset to Google Drive for easy access

---

## 🎯 Quick Start (5 Steps)

### Step 1: Prepare Your Dataset

Your dataset should follow this structure:

```
parking_spot_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

### Step 2: Upload Dataset to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder (e.g., `parking_spot_dataset`)
3. Upload your entire dataset folder
4. Note the path (e.g., `/MyDrive/parking_spot_dataset`)

### Step 3: Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click `File → Upload notebook`
3. Upload the `yolo_parking_colab_training.ipynb` file
4. Or use `File → Open notebook → GitHub` and enter your repo URL

### Step 4: Enable GPU

1. In Colab, go to `Runtime → Change runtime type`
2. Select `Hardware accelerator → GPU`
3. Choose `GPU type → T4` (free tier)
4. Click `Save`

### Step 5: Run the Notebook

1. Update `DATASET_PATH` in **Cell 6** to match your Google Drive path
2. Click `Runtime → Run all` to execute all cells
3. When prompted, authorize Google Drive access
4. Wait for training to complete (may take 1-3 hours depending on dataset size)

---

## 🔧 Configuration Options

### Change Model Size

In **Cell 14**, modify `PRETRAINED_MODEL_NAME`:

```python
PRETRAINED_MODEL_NAME = "yolo11n.pt"  # Nano (fastest, least accurate)
# PRETRAINED_MODEL_NAME = "yolo11s.pt"  # Small
# PRETRAINED_MODEL_NAME = "yolo11m.pt"  # Medium
# PRETRAINED_MODEL_NAME = "yolo11l.pt"  # Large (slowest, most accurate)
```

### Adjust Training Parameters

In **Cell 14**, modify the hyperparameter grid:

```python
param_grid = {
    "optimizer": ["auto"],
    "batch": [8],           # Increase to 16 if you have enough GPU memory
    "weight_decay": [0.0005],
    "epochs": [100],        # Increase for better results
}
```

**Memory Tips:**
- If you get "Out of Memory" errors, reduce `batch` to `4`
- Free Colab T4 GPU has ~15GB memory
- Smaller batches = slower training but less memory

### Training Single Model (Faster)

To train just one model instead of grid search, use:

```python
param_grid = {
    "optimizer": ["auto"],
    "batch": [8],
    "weight_decay": [0.0005],
    "epochs": [100],
}
```

This will train only 1 model instead of 4.

---

## 📊 After Training

### View Results

- **Cell 9**: Summary of all training runs
- **Cell 10**: Training curves and validation predictions
- **Cell 11**: Test the best model on a sample image

### Download Your Model

**Option 1: Direct Download**

In **Cell 12**, uncomment the download line:

```python
files.download(best_model_path)
```

**Option 2: Save to Google Drive** (Recommended)

Run **Cell 13** to copy all results to Google Drive:
- Models: `/MyDrive/yolo_training_results`
- MLflow logs: `/MyDrive/yolo_mlflow_results`

---

## 🐛 Troubleshooting

### Problem: "No GPU detected"

**Solution:**
- Go to `Runtime → Change runtime type → GPU`
- If GPU quota is exhausted, wait a few hours or use Kaggle

### Problem: "Dataset not found"

**Solution:**
- Check the path in Cell 6 matches your Google Drive folder
- Make sure you mounted Google Drive (Cell 6)
- Verify folder structure: `images/train`, `images/val`, `labels/train`, `labels/val`

### Problem: "Out of Memory"

**Solution:**
- Reduce batch size to `4` or `2` in Cell 14
- Use smaller model: `yolo11n.pt` instead of `yolo11m.pt`
- Close other browser tabs to free system memory

### Problem: "Disconnected from runtime"

**Solution:**
- Colab free tier has runtime limits (~12 hours)
- Save results to Google Drive frequently (Cell 13)
- For longer training, consider Colab Pro or use cloud VMs

### Problem: "Training taking too long"

**Solution:**
- Reduce epochs to 50 instead of 100
- Use smaller dataset (fewer images)
- Train single model instead of grid search
- Colab free tier may throttle after prolonged use

---

## ⚡ Performance Tips

### 1. Dataset Optimization
- **Resize large images**: Keep images at 640x640 or 1280x1280
- **Clean annotations**: Remove duplicate or low-quality images
- **Balanced dataset**: Ensure train/val split is ~80/20

### 2. Speed Up Training
- **Use smaller model**: Start with `yolo11n.pt` for testing
- **Increase batch size**: Try `16` if GPU allows
- **Cache dataset**: YOLO automatically caches after first epoch

### 3. Better Results
- **More epochs**: Try 150-200 epochs for production models
- **Data augmentation**: YOLO automatically applies augmentation
- **Larger model**: Use `yolo11m.pt` or `yolo11l.pt` for final training

---

## 📚 Additional Resources

### Free GPU Alternatives to Colab

1. **Kaggle Notebooks**
   - 30 hours/week GPU quota
   - Similar to Colab
   - [kaggle.com/code](https://www.kaggle.com/code)

2. **Paperspace Gradient**
   - Free tier available
   - Persistent storage
   - [gradient.paperspace.com](https://gradient.paperspace.com)

3. **Lightning AI**
   - Free GPU hours
   - Easy to use
   - [lightning.ai](https://lightning.ai)

### Upgrade Options

**Google Colab Pro** ($10/month)
- More GPU access
- Longer runtimes (24 hours)
- Background execution
- Better GPUs (V100, A100)

**Google Colab Pro+** ($50/month)
- Highest priority GPU access
- Longest runtimes
- Most memory

### YOLO Documentation

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [Training Custom Models](https://docs.ultralytics.com/modes/train/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)

---

## 🎓 Example Training Timeline

**Small Dataset (1000 images, 50 epochs):**
- Setup: 2-3 minutes
- Training: 20-30 minutes
- Total: ~35 minutes

**Medium Dataset (5000 images, 100 epochs):**
- Setup: 2-3 minutes
- Training: 2-3 hours
- Total: ~3 hours

**Large Dataset (10000+ images, 100 epochs):**
- Setup: 2-3 minutes
- Training: 5-8 hours
- Total: ~8 hours

---

## 💡 Pro Tips

1. **Start Small**: Test with 10 epochs first to verify everything works
2. **Monitor GPU Usage**: Use `!nvidia-smi` in a code cell to check GPU usage
3. **Save Frequently**: Run Cell 13 periodically to backup to Google Drive
4. **Use Checkpoints**: YOLO saves checkpoints automatically, you can resume if disconnected
5. **Compare Models**: Train multiple configurations and compare results in Cell 9

---

## 🆘 Need Help?

- **GitHub Issues**: [Your repo issues page]
- **Ultralytics Discord**: [ultralytics.com/discord](https://ultralytics.com/discord)
- **Stack Overflow**: Tag questions with `yolov8` and `google-colab`

---

## ✅ Checklist

Before starting training:
- [ ] Dataset uploaded to Google Drive
- [ ] GPU enabled in Colab
- [ ] Dataset path updated in Cell 6
- [ ] Training parameters configured in Cell 14
- [ ] Google Drive mounted and authorized

After training:
- [ ] Results reviewed in Cell 9-11
- [ ] Best model downloaded or saved to Drive
- [ ] MLflow artifacts backed up (Cell 13)
- [ ] Model tested on validation images

---

Happy Training! 🎉
