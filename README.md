# Handwritten Digit Recognition using MNIST Dataset

A beginner-friendly deep learning project using TensorFlow and the MNIST handwritten digit dataset.

This project trains a neural network to recognize handwritten digits (0–9) using the MNIST dataset.

---

# Features

- Loads raw MNIST `.ubyte` files
- Visualizes handwritten digits
- Performs preprocessing and normalization
- Builds a neural network using TensorFlow/Keras
- Trains and evaluates the model
- Predicts handwritten digits

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- idx2numpy

---

# Dataset

MNIST Dataset from Kaggle:

Link for the dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Download the dataset and place the files inside a `dataset/` folder.

---

# Project Structure

```bash
Handwritten-Digit-Recognition/
│
├── dataset/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone <your-repository-link>
cd Handwritten-Digit-Recognition
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the Python script:

```bash
python main.py
```

---

# Model Architecture

The neural network contains:

- Flatten Layer
- Dense Layer (128 neurons, ReLU)
- Dense Layer (64 neurons, ReLU)
- Output Layer (10 neurons, Softmax)

---

# Preprocessing Steps

- Dataset loading
- Shape inspection
- Pixel normalization
- Flattening images
- Training/testing split

---

# Output

The model:
- Displays sample handwritten digits
- Trains on MNIST data
- Evaluates accuracy
- Predicts digits from test images

Expected accuracy:
```bash
97% - 98%
```

---

# Concepts Learned

- Neural Networks
- TensorFlow/Keras
- Image preprocessing
- Normalization
- Activation functions
- Softmax classification
- Gradient descent
- Deep learning workflow

---

# Future Improvements

- Convolutional Neural Networks (CNNs)
- Real-time digit recognition
- Drawing canvas GUI
- Streamlit web app deployment
- Custom OCR dataset support

---

# License

This project is for learning and educational purposes.
