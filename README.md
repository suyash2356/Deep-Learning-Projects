# Deep Learning — Learning Journey & Projects

This repository is a working record of my Deep Learning progress — practice code, experiments, and projects, organized by architecture, in the order I learned them.

Instead of notes, I document my learning through **code**: what I studied, how I practiced it, and what I built to prove I understood it.

```
Learn → Understand → Implement → Experiment → Build
```

---

## Learning Path

```text
Deep Learning
│
├── Artificial Neural Networks (ANN)          ✅ Done
├── Convolutional Neural Networks (CNN)        ✅ Done
├── Recurrent Neural Networks (RNN)            🔄 In Progress
└── LSTM / GRU / Attention / Transformers      🔜 Next
```

Each stage below lists the concepts covered and the project(s) that put them into practice.

---

## Artificial Neural Networks (ANN)

**Concepts covered:** neurons, layers, weights & biases, forward propagation, activation functions (Sigmoid, ReLU, Tanh), loss functions, gradient descent, backpropagation, epochs/batches, learning rate, overfitting/underfitting, binary and multi-class classification, model evaluation.

**Projects:**
- **Binary Classification** — sigmoid activation, probability-based predictions, decision boundaries.
- **Handwritten Digit Recognition** — multi-class classification on image data.
- **Activation Function Experiments** — comparative study of Sigmoid, ReLU, and Tanh, and their effect on training.

**What this demonstrates:** I can build a feedforward network from the ground up, explain what each layer is doing to the data, and reason about why a given activation or loss function was chosen rather than just calling an API.

---

## Convolutional Neural Networks (CNN)

**Concepts covered:** images as numerical data, image preprocessing, convolution, kernels/filters, feature maps, padding, stride, pooling, CNN architecture, flattening, fully connected layers, image classification, overfitting in image models.

**Projects:**
- **Skin Cancer Detection** — image classification on a medical imaging dataset, covering preprocessing, CNN architecture design, training, validation, and evaluation of generalization.
- **Facial Recognition & verification** — Detect and recognize face from image, video and live video, using YOLO and transfer learning.
- **Object detection & segmentation** - Detect object in an image and crop it, enhanced it. Along with image segmentation. 

**What this demonstrates:** I can take raw image data through a full pipeline — preprocessing, a CNN built and trained from scratch, and an honest evaluation of how well it generalizes rather than just how well it fits training data.

---

## Recurrent Neural Networks (RNN) — In Progress

**Concepts covered:** sequential data, RNN architecture, hidden states, recurrent connections, sequence length, text preprocessing, tokenization, vocabulary, word indices, sequence padding, word embeddings, `Embedding` layers, `SimpleRNN`, binary text classification.

**Current project:**
- **Text Classification (IMDB dataset)** — sentiment classification pipeline:

```text
Text → Tokenized sequences → Padding → Embedding → SimpleRNN → Dense + Sigmoid → Positive / Negative
```

**What this demonstrates:** I can move from spatial data (images) to sequential data (text), and apply the same core deep-learning workflow — preprocessing, model design, training, evaluation — to a different data type and problem class.

---

## Tools & Technologies

| Category | Tools |
|---|---|
| Language | Python |
| Data & Numerical Computing | NumPy, Pandas |
| Visualization | Matplotlib |
| Classical ML | Scikit-learn |
| Deep Learning | TensorFlow, Keras, Pytorch |
| Environment | Jupyter Notebook, VS Code |

---

## Model Evaluation

Across all projects, evaluation has consistently covered: accuracy, precision, recall, F1-score, and comparing training vs. validation performance to check for overfitting/underfitting rather than optimizing for training accuracy alone.

---

## Repository Structure

```text
Deep-Learning/
│
├── ANN/
│   ├── Practice/
│   └── Projects/
│
├── CNN/
│   ├── Practice/
│   └── Projects/
│
├── RNN/
│   ├── Practice/
│   └── Projects/
│
└── README.md
```

---

## What's Next

- LSTM
- GRU
- Bidirectional RNNs
- Attention Mechanism
- Transformers
- Transfer Learning
- Larger, real-world projects
- Model deployment

---

## How to Use This Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
jupyter notebook
```

Each folder is self-contained — open the notebook for the concept or project you're interested in and run it directly.

---

## Why This Repository Exists

This is a public log of how I'm learning Deep Learning, not a claim that I know everything in the field. Each folder is evidence of a concept I've implemented myself, not just read about — the goal is that the code speaks for the understanding behind it.
