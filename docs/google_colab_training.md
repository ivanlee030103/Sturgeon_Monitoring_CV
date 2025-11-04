# YOLOv11 Training on Google Colab (500 Images)

Follow this guide to train a custom YOLOv11 model for sturgeon detection using 500 labelled images on Google Colab. The instructions assume you have image annotations in YOLO format (one `.txt` file per image) and that the dataset is stored on Google Drive.

## 1. Prepare the Dataset

1. Organise your dataset on Google Drive with the following structure:
   ```
   sturgeon_dataset/
       images/
           train/
           val/
       labels/
           train/
           val/
   ```
   - Reserve ~80% (400 images) for `train` and 20% (100 images) for `val`.
   - Ensure each image in `images/...` has a matching label file in `labels/...`.
2. Create a `data.yaml` describing the dataset:
   ```yaml
   path: /content/sturgeon_dataset
   train: images/train
   val: images/val
   names:
     0: sturgeon
   ```
   Upload this file to the root of the dataset folder in Drive.

## 2. Start a Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Switch the runtime to GPU: **Runtime > Change runtime type > Hardware accelerator = GPU**.
3. Mount Google Drive so the notebook can access your dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## 3. Install Dependencies

```python
%pip install ultralytics==8.3.0 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> Tip: Adjust the CUDA version in the PyTorch index URL if Colab updates the default GPU drivers.

## 4. Link the Dataset

```python
import shutil
import pathlib

source = pathlib.Path('/content/drive/MyDrive/sturgeon_dataset')
destination = pathlib.Path('/content/sturgeon_dataset')

if destination.exists():
    shutil.rmtree(destination)
shutil.copytree(source, destination)
```

Verify that images and labels copied correctly before training (optional):

```python
from pathlib import Path

image_count = len(list(Path('/content/sturgeon_dataset/images/train').glob('*.jpg')))
print(f"Training images: {image_count}")
```

## 5. Train YOLOv11

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # start from the nano checkpoint for faster training
model.train(
    data='/content/sturgeon_dataset/data.yaml',
    epochs=100,
    imgsz=1280,
    project='sturgeon-training',
    name='yolo11-sturgeon',
    batch=16,
    patience=20,
    optimizer='AdamW',
)
```

Guidance:
- `epochs=100` is typically sufficient for 500 images; adjust based on validation metrics.
- Increase `imgsz` or use a larger base model (`yolo11s.pt`, etc.) if quality needs improvement and runtime allows.
- Monitor the generated training plots in the Colab file explorer under `sturgeon-training/yolo11-sturgeon`.

## 6. Evaluate and Export

After training, run validation and export the weights:

```python
metrics = model.val()
print(metrics)

export_path = model.export(format='onnx')  # optional extra formats
print(f"Exported model: {export_path}")
```

The best-performing weights will be saved at:
```
/content/sturgeon-training/yolo11-sturgeon/weights/best.pt
```
Copy them back to Google Drive for safekeeping:

```python
import shutil

best_weights = '/content/sturgeon-training/yolo11-sturgeon/weights/best.pt'
shutil.copy(best_weights, '/content/drive/MyDrive/sturgeon_dataset/best_yolo11.pt')
```

## 7. Use the Model Locally

Download `best_yolo11.pt` from Google Drive and place it in your project directory. When running the monitoring pipeline, pass the weights file with:

```bash
python -m sturgeon_monitoring.pipeline path/to/video.mp4 --model best_yolo11.pt --display --display-heatmap
```

This will perform detection, tracking, live heatmap visualisation, and periodic analytics using your trained model.
