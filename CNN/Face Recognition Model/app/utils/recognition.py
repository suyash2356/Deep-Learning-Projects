# ============================================================
# FACE RECOGNITION ENGINE
# PyTorch ResNet-50 + SVM
# ============================================================

from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# 1. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Recognition device:", DEVICE)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SVM_PATH = PROJECT_ROOT / "face_recognition_svm_11class.pkl"

CLASSES_PATH = PROJECT_ROOT / "face_classes_11class.pkl"


# ============================================================
# 3. LOAD CLASSIFIER
# ============================================================

if not SVM_PATH.exists():

    raise FileNotFoundError(
        f"SVM model not found:\n{SVM_PATH}"
    )


if not CLASSES_PATH.exists():

    raise FileNotFoundError(
        f"Class file not found:\n{CLASSES_PATH}"
    )


classifier = joblib.load(SVM_PATH)

selected_people = joblib.load(CLASSES_PATH)


print(
    "Recognition classes:",
    len(selected_people)
)


# ============================================================
# 4. LOAD RESNET-50
# ============================================================

weights = ResNet50_Weights.DEFAULT

model = resnet50(
    weights=weights
)

# Remove ImageNet classifier
model.fc = nn.Identity()

model = model.to(DEVICE)

model.eval()


# ============================================================
# 5. SAME PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 6. RECOGNITION THRESHOLD
# ============================================================

RECOGNITION_THRESHOLD = 0.70


# ============================================================
# 7. EXTRACT EMBEDDING
# ============================================================

def extract_embedding(face_bgr):

    """
    Convert detected face into
    normalized 2048-dimensional embedding.
    """

    # BGR → RGB
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    # NumPy → PIL
    face_pil = Image.fromarray(
        face_rgb
    )

    # Preprocessing
    tensor = transform(
        face_pil
    )

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # GPU / CPU
    tensor = tensor.to(DEVICE)


    # Extract embedding
    with torch.no_grad():

        embedding = model(
            tensor
        )


    # Tensor → NumPy
    embedding = (
        embedding
        .cpu()
        .numpy()
        .flatten()
    )


    # L2 normalization
    norm = np.linalg.norm(
        embedding
    )

    embedding = embedding / (
        norm + 1e-10
    )


    return embedding


# ============================================================
# 8. RECOGNIZE FACE
# ============================================================

def recognize_face(face_bgr):

    """
    Recognize one detected face.

    Returns:

        name
        confidence
        class_id
        is_known
    """

    embedding = extract_embedding(
        face_bgr
    )


    embedding = embedding.reshape(
        1,
        -1
    )


    # Prediction
    prediction = classifier.predict(
        embedding
    )[0]


    # Probabilities
    probabilities = classifier.predict_proba(
        embedding
    )[0]


    confidence = float(
        probabilities[prediction]
    )


    person = selected_people[
        prediction
    ]


    # Unknown logic
    if confidence < RECOGNITION_THRESHOLD:

        person = "Unknown"

        is_known = False

    else:

        is_known = True


    return (
        person,
        confidence,
        int(prediction),
        is_known
    )