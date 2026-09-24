# 🔍 Zero-Shot AI Object Detector & Tagging Engine using CLIP

[![Day](https://img.shields.io/badge/Day-28--30-blue?style=for-the-badge&logo=python)](https://github.com/manasha1232/30-Day-Computer-Vision-Challenge)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

An Open-Vocabulary Vision & Deep Learning system for **Zero-Shot Object Detection and Tagging**. Detects arbitrary user-specified natural language text prompts (*"vintage red mug"*, *"yellow sports ball"*, *"metallic wrench tool"*, *"green plant pot"*) in images without requiring custom dataset retraining.

---

## 🌟 Key Features

- 🧠 **Open-Vocabulary Prompt Querying**: Detects objects specified dynamically in free-form English text.
- 📐 **Multi-Scale Region Proposals**: Adaptive contour bounding box extraction and multi-scale sliding windowing.
- ⚡ **Vision-Language Cosine Similarity**: PyTorch vector dot-product similarity $\mathbf{S}_{i, j} = \frac{\mathbf{v}_{\text{img}} \cdot \mathbf{v}_{\text{text}}}{\|\mathbf{v}_{\text{img}}\| \|\mathbf{v}_{\text{text}}\|}$ with Softmax temperature scaling.
- 🎯 **Non-Maximum Suppression (NMS)**: Overlap filtering to keep highest-confidence non-overlapping bounding boxes.
- 📊 **Telemetry Audit Exporter**: Exports JSON telemetry detailing proposal counts, bounding boxes, and zero-shot confidence distributions.

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/manasha1232/zero_shot_clip_object_detector.git
cd zero_shot_clip_object_detector

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1️⃣ Run with Synthetic Test Generator
```bash
python generate_demo_clip_input.py
python clip_object_detector.py
```

---

## 📊 Sample Output Telemetry JSON

```json
{
    "project": "Zero-Shot AI Object Detector & Tagging Engine using CLIP",
    "day": 28,
    "status": "SUCCESS",
    "resolution": {
        "width": 640,
        "height": 480
    },
    "query_text_prompts": [
        "vintage red mug",
        "yellow sports ball",
        "metallic wrench tool",
        "green plant pot"
    ],
    "performance": {
        "execution_duration_sec": 0.285,
        "region_proposals_evaluated": 8,
        "final_detections_count": 4
    },
    "detections": [
        {
            "label": "vintage red mug",
            "confidence_score": 0.94,
            "bounding_box": {"x": 50, "y": 50, "width": 180, "height": 180}
        },
        {
            "label": "yellow sports ball",
            "confidence_score": 0.92,
            "bounding_box": {"x": 420, "y": 50, "width": 180, "height": 180}
        },
        {
            "label": "metallic wrench tool",
            "confidence_score": 0.89,
            "bounding_box": {"x": 50, "y": 270, "width": 200, "height": 160}
        },
        {
            "label": "green plant pot",
            "confidence_score": 0.91,
            "bounding_box": {"x": 420, "y": 260, "width": 180, "height": 180}
        }
    ]
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
