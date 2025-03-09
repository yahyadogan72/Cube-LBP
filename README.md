# Cube-LBP

## Introduction

Cube-LBP is an advanced extension of Local Binary Pattern (LBP) that introduces a volumetric approach to feature extraction. Unlike conventional LBP, which relies on grayscale conversion or independent channel processing, Cube-LBP treats images as three-dimensional structures, preserving spatial and chromatic relationships. This repository provides an implementation of Cube-LBP-based feature extraction and classification, with applications in CIFAR-10, CIFAR-100, and face recognition datasets. The framework integrates with deep learning models to enhance classification performance.

## Repository Structure

This repository includes the following Python scripts:

- `LBPKup.py` : Core LBP feature extraction functions.
- `LBPKupCifar10.py` : Cube-LBP-based classification on the CIFAR-10 dataset.
- `LBPKupCifar100.py` : Cube-LBP-based classification on the CIFAR-100 dataset.
- `LbpKupFace.py` : Face recognition using Cube-LBP features.
- `LBPKupTest.py` : A test script to validate Cube-LBP-based processing.

## Dependencies

The required dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Running Cube-LBP Feature Extraction

```python
from LBPKup import lbp_from_tensor
import tensorflow as tf

img_path = "path/to/image.jpg"
img = tf.io.read_file(img_path)
img = tf.io.decode_jpeg(img, channels=3)
img = tf.keras.layers.Resizing(256, 256)(img)

result = lbp_from_tensor(img, mode=1)
```

### 2. Running Classification on CIFAR-10 and CIFAR-100

```bash
python LBPKupCifar10.py
python LBPKupCifar100.py
```

### 3. Running Face Recognition Model

```bash
python LbpKupFace.py
```

## Citation

If you use Cube-LBP in your research, please cite it as follows:

```
@article{xx,
  author    = {Your Name},
  title     = {xx},
  journal   = {xx},
  year      = {2024},
  volume    = {xx},
  number    = {xx},
  pages     = {xx--xx},
  note      = {Under consideration},
}
```

## License

This project is released under the MIT License.
