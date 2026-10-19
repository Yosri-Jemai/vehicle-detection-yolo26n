
# Vehicle Detection using YOLO26n

A vehicle detection project built with **YOLO26n**, trained using the Ultralytics framework on a vehicle dataset. The project focuses on detecting and classifying vehicles in images, with a primary use case of **single-vehicle detection per image**.

## Overview

The model is based on YOLO26n, a lightweight object detection architecture designed for efficient inference.

It was fine-tuned from pretrained weights using a vehicle dataset containing six classes. Although the primary objective is to detect one vehicle in an image, the model can also detect multiple vehicles when they are present.


## Dataset

This project uses the [Vehicle Dataset for YOLO](https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo), published by Nadin Pethiyagoda on Kaggle.

The dataset contains vehicle images annotated with bounding boxes in YOLO format for object detection. It includes six vehicle categories.

### Dataset Splits Used

| Split | Images |
|---|---:|
| Training | 2,100 |
| Validation | 450 |
| Testing | 450 |
| **Total used** | **3,000** |

The training split was used to fine-tune YOLO26n, the validation split to monitor model performance, and the test split to evaluate the trained model on separate images.

### Classes

- Car
- Tuk-tuk (three-wheel vehicle)
- Bus
- Truck
- Motorcycle
- Minivan

### Dataset Source

- **Creator:** Nadin Pethiyagoda
- **Platform:** Kaggle
- **Link:** [Vehicle Dataset for YOLO](https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo)

The dataset is used for educational and object detection experimentation. Please refer to the original dataset page for its licensing terms and usage conditions.


## Vehicle Classes

The model detects the following categories:

- Car
- Tuk-tuk
- Bus
- Truck
- Motorcycle
- Minivan

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLO26n |
| Task | Object Detection |
| Framework | Ultralytics |
| Training epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Training images | 2,100 |
| Validation images | 450 |
| Test images | 450 |
| Number of classes | 6 |
| Training hardware | NVIDIA Tesla T4 GPU |
| Model parameters (fused) | 2.39 million |
| Model size | Approximately 5.4 MB |

## Performance

The model was evaluated on separate validation and test splits.

### Validation Results

| Metric | Score |
|---|---:|
| Precision | 96.06% |
| Recall | 96.16% |
| mAP@50 | 97.84% |
| mAP@50–95 | 91.82% |

### Test Results

| Metric | Score |
|---|---:|
| Precision | 97.60% |
| Recall | 95.31% |
| mAP@50 | 98.78% |
| mAP@50–95 | 92.46% |

**Note:** These results are based on the provided evaluation logs. Performance on real-world images may vary depending on image quality, lighting, camera angle, and vehicle appearance.

## Trained Model

The trained weights are hosted on Hugging Face.

**Model repository:** [Yosriii/vehicle-detection-yolo26n](https://huggingface.co/Yosriii/vehicle-detection-yolo26n)

The weights are not stored directly in this GitHub repository.

## Technologies

- Python
- Ultralytics YOLO26
- PyTorch
- Google Colab
- Hugging Face Hub

## Installation

Install the required libraries:

```bash
pip install ultralytics huggingface_hub
```

## Usage

### 1. Download the trained model

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Yosriii/vehicle-detection-yolo26n",
    filename="best.pt"
)
```

Make sure the Hugging Face repository contains `best.pt`. If the repository is private, authenticate with Hugging Face first.

### 2. Run detection on an image

```python
from ultralytics import YOLO

model = YOLO(model_path)

results = model.predict(
    source="path/to/your/image.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)
```

Replace `path/to/your/image.jpg` with the path to your image.

The model returns the detected vehicle classes, bounding boxes, and confidence scores. Results are saved by Ultralytics when `save=True`.

### 3. Detect one primary vehicle

For images expected to contain one vehicle, you can limit the output to the highest-confidence detection:

```python
results = model.predict(
    source="path/to/your/image.jpg",
    imgsz=640,
    conf=0.25,
    max_det=1,
    save=True
)
```

**Important:** `max_det=1` limits the output to one detection. It does not guarantee that the detected vehicle is correct, nor does it improve the model's accuracy. The model itself can detect multiple vehicles when this limit is not applied.

## Project Notebook

The Jupyter notebook contains the training, evaluation, and inference workflow.

- `vehicle detection.ipynb`

## Future Improvements

- Evaluate performance on more diverse real-world images.
- Improve robustness to occlusion and challenging lighting.
- Explore vehicle counting and video-based detection.
- Compare single-detection and multi-detection use cases.

## Author

**Yosri Jemai**

- GitHub: [Yosri-Jemai](https://github.com/Yosri-Jemai/vehicle-detection-yolo26)
- Hugging Face: [Yosriii](https://huggingface.co/Yosriii/vehicle-detection-yolo26n)
