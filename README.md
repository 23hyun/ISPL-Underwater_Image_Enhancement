# ISPL-Underwater_Image_Enhancement
Pusan National University, Department of Electronics Engineering, Image Signal Processing Laboratory 2026 Fall Semester Graduation Project
<p align="center">
  <img src="/architecture.png" width="600">
</p>

## Dataset

Download the training and test datasets from [Underwater_Dataset](https://drive.google.com/drive/folders/19O3WtVpYTwp4jry7lTlQfv4ol1KXQ-dp?usp=sharing).

Download the training, validation, and test datasets and arrange the files as follows:

```text
ISPL-Underwater_Image_Enhancement/
├── data/
│   ├── trainA/
│   └── trainB/
├── validation/
└── test/
```

The `data/trainA` directory contains degraded underwater images for training, while the `data/trainB` directory contains the corresponding clean ground-truth images.

The images in `trainA` and `trainB` should have matching filenames.
