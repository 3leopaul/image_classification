# CLIP Image Classification & Semantic Search

A high-performance implementation of **Zero-Shot Image Classification** and **Semantic Image Search** using OpenAI's **CLIP (Contrastive Language-Image Pre-training)** model. This project demonstrates how to bridge the gap between visual concepts and natural language for versatile computer vision tasks without the need for task-specific training.

---

## Overview

This repository features a Jupyter Notebook that leverages state-of-the-art transformer architectures to:
1.  **Classify images** based on arbitrary text prompts (Zero-Shot).
2.  **Search through image collections** using natural language queries.

By using the `clip-vit-base-patch16` model, we can map images and text into a shared embedding space, allowing for direct similarity comparisons.

---

## Key Features

- **Zero-Shot Classification**: Predict image categories without any fine-tuning. Simply provide a list of labels like `"a photo of a cat"` or `"a professional portrait"`.
- **Semantic Image Search**: Input a descriptive query (e.g., `"a cute dog"`) and retrieve the most relevant image from a local dataset.
- **Deep Learning Backbone**: Powered by PyTorch and Hugging Face Transformers.
- **Interactive Visualization**: Uses `Pillow` for seamless image handling and display within the notebook environment.

---

## Technology Stack

| Component | Library/Model |
| :--- | :--- |
| **Model Architecture** | [OpenAI CLIP (ViT-B/16)](https://huggingface.co/openai/clip-vit-base-patch16) |
| **Framework** | [PyTorch](https://pytorch.org/) |
| **Model Hub** | [Hugging Face Transformers](https://github.com/huggingface/transformers) |
| **Image Processing** | [Pillow (PIL)](https://python-pillow.org/) |
| **Environment** | Jupyter Notebook |

---

## Project Structure

```text
.
├── image_classification.ipynb   # Main exploration and implementation notebook
├── cat_cute.png                 # Sample test image
├── cat2.jpg                     # Sample test image
├── dog.png                      # Sample test image
├── goat.png                     # Sample test image
└── README.md                    # Project documentation
```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install torch torchvision transformers pillow
```

### Running the Project

1.  Clone the repository or navigate to the project folder.
2.  Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open `image_classification.ipynb` and run the cells sequentially.

---

## Usage Example

### Zero-Shot Classification
```python
# Define your own classes on the fly!
text_classes = ["a photo of a cat", "a photo of a dog", "a photo of a goat"]

# The model calculates probabilities for each class
# Output: Image 1: a photo of a cat | Probability = 0.9979
```

### Semantic Search
```python
query = "a cat on grass"
# Returns the image with the highest cosine similarity to the text embedding.
```

---

## Results & Performance

The CLIP model demonstrates remarkable robustness in generalizing to various visual styles. In this project, we achieve near-perfect accuracy on basic classification tasks and high relevance in semantic search across the included sample image set.

---

## License

This project is intended for educational and research purposes. CLIP is subject to [OpenAI's Model License](https://github.com/openai/CLIP/blob/main/model-card.md).
