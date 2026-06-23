#  Bird or Forest — Deep Learning Image Classifier

A binary image classifier that distinguishes **birds** from **forests** using transfer learning with [fastai](https://docs.fast.ai/) and a ResNet-18 backbone. Images are sourced via the Pexels API and DuckDuckGo Search.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adi-Arora1/Bird_or_forest_DeepLearning/blob/main/BirdOrNot_DeepLearning.ipynb)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)

---

## Overview

This project demonstrates an end-to-end deep learning workflow:

1. **Data collection** — fetch labeled images from the Pexels API
2. **Preprocessing** — resize and organize images into class folders
3. **Model training** — fine-tune a pretrained ResNet-18 with fastai
4. **Inference** — classify a new image as bird or forest

The classifier uses transfer learning, so it converges quickly even with a small dataset.

---

## Project Structure

```
Bird_or_forest_DeepLearning/
├── BirdOrNot_DeepLearning.ipynb   # Main notebook
├── bird_or_not/
│   ├── bird/                      # Downloaded bird images
│   └── forest/                    # Downloaded forest images
└── bird.jpg                       # Sample image for inference
```

---

## Requirements

| Package | Purpose |
|---|---|
| `fastai` | Model training, data loading, transfer learning |
| `duckduckgo-search` | Supplementary image search |
| `Pillow` | Image processing |
| `fastdownload` | Downloading images from URLs |
| `requests` | Pexels API calls |

Install all dependencies:

```bash
pip install -Uqq fastai duckduckgo-search
```

> **Note:** This notebook is designed to run on [Google Colab](https://colab.research.google.com/), which has GPU support enabled by default.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Adi-Arora1/Bird_or_forest_DeepLearning.git
cd Bird_or_forest_DeepLearning
```

### 2. Get a Pexels API key

Sign up for a free API key at [pexels.com/api](https://www.pexels.com/api/) and replace the placeholder in the notebook:

```python
PEXELS_KEY = 'your_api_key_here'
```

### 3. Open the notebook

Either open in Google Colab using the badge above, or locally:

```bash
jupyter notebook BirdOrNot_DeepLearning.ipynb
```

---

## Usage

### Training

Run all cells in order. The notebook will:

- Fetch images for the `bird` and `forest` categories from Pexels
- Resize all images to a maximum of 400px
- Build a `DataBlock` with an 80/20 train/validation split
- Fine-tune ResNet-18 for 3 epochs

### Inference

After training, classify any image:

```python
is_bird, _, probs = learn.predict(PILImage.create('your_image.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird (vs forest): {probs[0]:.4f}")
```

---

## How It Works

### Data Collection

Images are fetched using the Pexels API with a simple helper:

```python
def search_images(term, max_images=100):
    url = "https://api.pexels.com/v1/search"
    headers = {'Authorization': PEXELS_KEY}
    params = {'query': term, 'per_page': max_images}
    res = requests.get(url, headers=headers, params=params).json()
    return L([r['src']['large'] for r in res['photos']])
```

### Model Architecture

The model uses **ResNet-18** pretrained on ImageNet, fine-tuned on the custom dataset via fastai's `vision_learner`:

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

Images are resized to `192×192` using squish resizing before being passed to the model.

### DataBlock Configuration

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)
```

Labels are derived from the parent folder name (`bird` or `forest`).

---

## Results

The model is evaluated on a 20% validation split using **error rate** as the primary metric. With 3 fine-tuning epochs on ResNet-18, the classifier achieves strong performance on the binary classification task.

---

## Acknowledgements

- [fast.ai](https://www.fast.ai/) for the fantastic deep learning library and course
- [Pexels](https://www.pexels.com/) for the free image API
