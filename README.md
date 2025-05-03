# Deepfake Music Detection (V1)

A machine learning project for detecting AI-generated music using mel spectrograms and a simple CNN classifier.
This experiment was conducted as part of a research paper for a graduate course in machine learning.

---

## 🎯 Project Goal

To investigate whether a convolutional neural network can distinguish between real human-performed music and AI-generated "deepfake" music clips using spectrogram-based features.

---

## 📂 Contents

* `notebooks/` — Jupyter notebooks for dataset creation, model training, and experimental runs
* `README.md` — Project overview and instructions
* `CREDITS.md` — References to tutorials and research inspiration (see below)

> ⚠️ This repo does **not** include the full audio dataset or trained model files due to size. Notebooks can regenerate the data if needed.

---

## 🧠 How It Works

### 1. Data Preparation

Music samples were sourced and grouped into two classes:

* **Human**: Professionally performed or recorded music
* **Deepfake**: AI-generated clips from public tools and repositories

Several variants of the dataset were created using `librosa` for pitch shifting, tempo changes, and key modifications.
Spectrograms were extracted using mel scale transforms.

### 2. Model Architecture

A simple CNN (ResNet18 pretrained on ImageNet) was used to classify spectrograms. Final experiments included variations with iterative training, pitch and tempo distortion testing.

### 3. Evaluation

Accuracy and confusion matrix metrics were used to compare performance across augmentations:

* **Base case**: Unaltered music
* **Pitch-shifted**
* **Tempo-shifted**
* **Combined pitch + tempo distortion**

See the included research paper (available upon request) for detailed results.

---

## 🚀 Getting Started

### Requirements

* Python 3.8+
* PyTorch
* librosa
* matplotlib
* numpy
* scikit-learn

Install with pip:

```bash
pip install torch torchvision librosa matplotlib numpy scikit-learn
```

### Running the Notebooks

1. Generate dataset:

   * Run any of the `create_dataset_*.ipynb` notebooks
2. Train and evaluate:

   * Use `research_human_v_deepfake.ipynb` or `research_iterative_learning.ipynb`

---

## 📊 Results Summary

The classifier achieved strong separation on original and moderately altered datasets, but struggled more with heavily pitch- or tempo-shifted examples — raising open questions about model robustness and overfitting to stylistic features.

For comparative analysis and further benchmarking, consider exploring the [FakeMusicCaps Dataset on Hugging Face](https://huggingface.co/datasets/google/MusicCaps), which provides a collection of AI-generated music samples for detection and attribution studies.

---

## 🧾 Credits & References

See [`CREDITS.md`](./CREDITS.md) for full attribution.

### 📚 Academic Influence

* [Detecting Music Deepfakes is Easy but Actually Hard (Afchar et al., 2024)](https://arxiv.org/abs/2405.04181)
* [FakeMusicCaps](https://arxiv.org/abs/2409.10684) — Evaluation and benchmarking methodology
* [SONICS: Synthetic Or Not - Identifying Counterfeit Songs](https://arxiv.org/abs/2408.14080) — Contextual background and use cases

### 🎧 Dataset Sources

* [FakeMusicCaps Dataset on Hugging Face](https://huggingface.co/datasets/PoliMi-DeepFakes/FakeMusicCaps)
* [Google MusicCaps (via AudioSet)](https://github.com/google-research/audioset/tree/master/music_caps)

### 🔨 Tools and Libraries

* [Librosa](https://librosa.org/)
* [PyTorch](https://pytorch.org/)
* [ResNet18 - torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

### 🧪 Tutorial Inspiration

* [Kaggle: GTZAN Mel Spectrogram + ResNet18](https://www.kaggle.com/code/nippani/gtzan-mel-spectrogram-resnet18)
* [Using Librosa to Change Pitch](https://medium.com/strategio/using-librosa-to-change-the-pitch-of-an-audio-file-49efdb2dd6c)
* [How to Understand Mel Spectrograms](https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056)
* [Ketan Doshi’s Audio-Mel](https://ketanhdoshi.github.io/Audio-Mel/)
* [StackOverflow Librosa Answer](https://stackoverflow.com/a/52683474)

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE.md` for more details.

If you use this project or build upon it, please consider citing the references above and crediting the original authors. 🎶
