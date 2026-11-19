# Predicting ASD from Eye-Tracking Data

Classifying children as **Autism Spectrum Disorder (ASD)** or **Typically Developing (TD)** from eye-tracking data, comparing two approaches:

1. **Image-based deep learning:** pretrained ResNet50, DenseNet121, ViT-B/16 and Swin-T fine-tuned on scanpath visualisation images.
2. **Tabular machine learning:** per-participant features (fixation/saccade proportions and the mean, median and std of pupil size and point of regard) taken from the raw recordings, then classified with a neural network, SVM, Decision Tree, Naive Bayes, KNN and Random Forest.

## Publication

This repository contains the code for the following paper:

> Syukron, M., & Faresta, R. A. (2026). Deep learning algorithms for autism spectrum disorder detection using eye-tracking patterns. *Jurnal Ilmiah Kursor*, 13(3). https://doi.org/10.21107/kursor.v13i3.468

If you use this code, please cite:

```bibtex
@article{syukron2026asd,
  title   = {Deep learning algorithms for autism spectrum disorder detection using eye-tracking patterns},
  author  = {Syukron, Muhamad and Faresta, Rangga Alif},
  journal = {Jurnal Ilmiah Kursor},
  volume  = {13},
  number  = {3},
  year    = {2026},
  doi     = {10.21107/kursor.v13i3.468},
  url     = {https://www.kursorjournal.org/index.php/kursor/article/view/468}
}
```

## Repository structure

```
.
├── ASD_Classification.ipynb   # main notebook (image + tabular experiments)
├── archive/                   # earlier exploratory notebooks (2024)
│   ├── 2024-08-24_image_preprocessing_experiments.ipynb   # image enhancement, greyscale vs RGB, autoencoder
│   └── 2024-09-13_asd_level_experiments.ipynb             # greyscale models, CNN, 3-class ASD-level attempt
├── requirements.txt
└── README.md
```

The dataset is **not** included in this repository (see below).

## Dataset

This project uses the public dataset by Cilia et al. (2022): 59 children (29 ASD, 30 TD), with scanpath images and raw eye-tracking recordings.

> Cilia, F., Carette, R., Elbattah, M., Guérin, J., & Dequen, G. (2022). *Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder.* In Proceedings of the IJCAI–ECAI Workshop on Scarce Data in Artificial Intelligence for Healthcare (SDAIH).

Download it from the authors' release and place it in the repository root like this (these folders are git-ignored):

```
Eye-Tracking Dataset/
└── Eye-tracking Output/       # raw recordings, one CSV per recording session
Images/
├── TSImages/                  # scanpath images, ASD participants  (label 1)
└── TCImages/                  # scanpath images, TD participants   (label 0)
Metadata/
└── Metadata_Participants.csv
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ASD_Classification.ipynb
```

The notebook also runs on **Google Colab** (a GPU runtime is recommended). There it mounts Google Drive and reads the data from `MyDrive/ASD Level/`. Locally, it reads the dataset folders from the repository root.

## Results

Best results reported in the [paper](https://doi.org/10.21107/kursor.v13i3.468):

| Approach | Best model | Accuracy | Recall |
|---|---|---|---|
| Tabular eye-tracking features (gaze position, pupil diameter) | Multilayer Perceptron (MLP) | 91.7% | 83.3% |
| Scanpath images | Vision Transformer (ViT) | 78.2% | 88.6% |

See the paper for the full comparison of all models. Outputs saved in the notebook come from a later re-run and may differ slightly from the published numbers.
