
# ============================================================
# FACE RECOGNITION ENGINE
# YuNet + OpenCV Enhancement + ResNet-50
# ============================================================

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_recognition_resnet50.pth"
)

YUNET_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

# Recognition threshold.
#
# We deliberately do NOT use a very low threshold.
# If the model is not confident, it should say Unknown.

RECOGNITION_THRESHOLD = 0.50

DETECTION_THRESHOLD = 0.50

TOP_K = 5


# ============================================================
# MODEL ARCHITECTURE
# EXACT ARCHITECTURE USED DURING TRAINING
# ============================================================

class FaceRecognitionModel(nn.Module):

    def __init__(
        self,
        num_classes,
        embedding_dim=512
    ):

        super().__init__()

        # ----------------------------------------------------
        # ResNet-50 backbone
        # ----------------------------------------------------

        backbone = resnet50(
            weights=None
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone


        # ----------------------------------------------------
        # 2048 -> 512 embedding
        # ----------------------------------------------------

        self.embedding = nn.Sequential(

            nn.Linear(
                2048,
                embedding_dim
            ),

            nn.BatchNorm1d(
                embedding_dim
            ),

            nn.ReLU()
        )


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )


    def forward(self, x):

        features = self.backbone(x)

        embedding = self.embedding(
            features
        )

        # L2 normalization

        embedding = embedding / (
            torch.norm(
                embedding,
                p=2,
                dim=1,
                keepdim=True
            ) + 1e-10
        )

        logits = self.classifier(
            embedding
        )

        return logits, embedding


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    selected_people = checkpoint[
        "selected_people"
    ]

    embedding_dim = checkpoint[
        "embedding_dim"
    ]

    num_classes = len(
        selected_people
    )

    model = FaceRecognitionModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(DEVICE)

    model.eval()

    return (
        model,
        selected_people,
        checkpoint
    )


# ============================================================
# IMAGE TRANSFORMATION
# MUST MATCH TRAINING
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
# YUNET FACE DETECTOR
# ============================================================

def create_detector():

    if not YUNET_PATH.exists():

        raise FileNotFoundError(
            f"YuNet model not found:\n{YUNET_PATH}"
        )

    detector = cv2.FaceDetectorYN.create(

        str(YUNET_PATH),

        "",

        (320, 320),

        DETECTION_THRESHOLD,

        0.3,

        5000
    )

    return detector


# ============================================================
# ENHANCEMENT
# LEVEL 1 OPENCV ENHANCEMENT
# ============================================================

def enhance_face(face):

    if face is None:
        return None

    # --------------------------------------------------------
    # Resize before enhancement
    # --------------------------------------------------------

    height, width = face.shape[:2]

    if width < 300 or height < 300:

        scale = max(
            300 / width,
            300 / height
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        face = cv2.resize(
            face,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )


    # --------------------------------------------------------
    # CLAHE enhancement
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = (
        cv2.split(lab)
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel
        )
    )

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )


    # --------------------------------------------------------
    # Mild sharpening
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.25,
        blurred,
        -0.25,
        0
    )

    return enhanced


# ============================================================
# DETECT FACES
# ============================================================

def detect_faces(
    image,
    detector
):

    if image is None:
        return []


    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    _, detections = detector.detect(
        image
    )


    if detections is None:
        return []


    results = []


    for detection in detections:

        x, y, w, h = (
            detection[:4]
            .astype(int)
        )

        confidence = float(
            detection[14]
        )


        # Clamp coordinates

        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(
            width,
            x + w
        )

        y2 = min(
            height,
            y + h
        )


        if x2 <= x1 or y2 <= y1:
            continue


        face = image[
            y1:y2,
            x1:x2
        ].copy()


        results.append({

            "bbox": (
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ),

            "confidence": confidence,

            "face": face
        })


    return results


# ============================================================
# PREPARE FACE
# ============================================================

def prepare_face(face):

    enhanced = enhance_face(
        face
    )

    rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    tensor = transform(
        pil_image
    )

    tensor = tensor.unsqueeze(
        0
    )

    return (
        tensor.to(DEVICE),
        enhanced
    )


# ============================================================
# RECOGNIZE ONE FACE
# ============================================================

def recognize_face(
    face,
    model,
    selected_people
):

    input_tensor, enhanced = (
        prepare_face(face)
    )


    with torch.no_grad():

        logits, embedding = model(
            input_tensor
        )


    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]


    probabilities_np = (
        probabilities
        .cpu()
        .numpy()
    )


    sorted_indices = np.argsort(
        probabilities_np
    )[::-1]


    top_predictions = []


    for index in sorted_indices[:TOP_K]:

        top_predictions.append({

            "name":
                selected_people[index],

            "confidence":
                float(
                    probabilities_np[index]
                )
        })


    best_index = sorted_indices[0]

    best_confidence = float(
        probabilities_np[
            best_index
        ]
    )

    best_name = selected_people[
        best_index
    ]


    # --------------------------------------------------------
    # UNKNOWN DECISION
    # --------------------------------------------------------

    if (
        best_confidence
        < RECOGNITION_THRESHOLD
    ):

        identity = "Unknown"

    else:

        identity = best_name


    return {

        "identity": identity,

        "confidence":
            best_confidence,

        "embedding":
            embedding[0]
            .cpu()
            .numpy(),

        "enhanced_face":
            enhanced,

        "top_predictions":
            top_predictions
    }


# ============================================================
# PROCESS COMPLETE IMAGE
# ============================================================

def process_image(
    image,
    model,
    detector,
    selected_people
):

    faces = detect_faces(
        image,
        detector
    )


    results = []


    for face_data in faces:

        recognition = recognize_face(

            face_data["face"],

            model,

            selected_people
        )


        result = {

            "bbox":
                face_data["bbox"],

            "detection_confidence":
                face_data["confidence"],

            **recognition
        }


        results.append(
            result
        )


    return results
