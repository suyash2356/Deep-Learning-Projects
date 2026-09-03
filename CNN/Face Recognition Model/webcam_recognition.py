# ============================================================
# REAL-TIME FACE RECOGNITION
# PyTorch ResNet-50 + SVM + OpenCV
#
# Diagnostic Version
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
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SVM_PATH = PROJECT_ROOT / "face_recognition_svm_11class.pkl"
CLASSES_PATH = PROJECT_ROOT / "face_classes_11class.pkl"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("REAL-TIME FACE RECOGNITION")
print("PyTorch ResNet-50 + SVM + OpenCV")
print("=" * 70)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. LOAD SVM CLASSIFIER
# ============================================================

if not SVM_PATH.exists():

    raise FileNotFoundError(
        f"\nSVM model not found:\n{SVM_PATH}\n"
    )


if not CLASSES_PATH.exists():

    raise FileNotFoundError(
        f"\nClass file not found:\n{CLASSES_PATH}\n"
    )


print("\nLoading SVM classifier...")

classifier = joblib.load(SVM_PATH)

selected_people = joblib.load(CLASSES_PATH)


print("SVM classifier loaded!")

print(
    "Number of classes:",
    len(selected_people)
)


print("\nClasses:")

for idx, person in enumerate(selected_people):

    print(
        f"{idx}: {person}"
    )


# ============================================================
# 4. LOAD RESNET-50
# ============================================================

print("\nLoading ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(
    weights=weights
)

# Remove ImageNet classifier
model.fc = nn.Identity()

model = model.to(device)

model.eval()

print("ResNet-50 loaded!")

print("Embedding size: 2048")


# ============================================================
# 5. IMAGE TRANSFORMATION
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
# 6. FACE DETECTOR
# ============================================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    cascade_path
)


if face_cascade.empty():

    raise RuntimeError(
        "Could not load OpenCV Haar Cascade."
    )


print("\nFace detector loaded!")


# ============================================================
# 7. SETTINGS
# ============================================================

CAMERA_INDEX = 0

# Start with 50% for debugging.
# We can increase this later.
CONFIDENCE_THRESHOLD = 0.50

# Padding around detected face.
# This makes the crop slightly larger.
FACE_PADDING = 0.25

# Print detailed predictions every N frames.
DEBUG_INTERVAL = 30

print(
    f"\nConfidence threshold: "
    f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
)

print(
    f"Face crop padding: "
    f"{FACE_PADDING * 100:.0f}%"
)


# ============================================================
# 8. FUNCTION: EXTRACT EMBEDDING
# ============================================================

def extract_embedding(face_bgr):

    """
    Convert a detected face into a
    2048-dimensional ResNet-50 embedding.
    """

    # --------------------------------------------------------
    # OpenCV BGR -> RGB
    # --------------------------------------------------------

    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # NumPy -> PIL
    # --------------------------------------------------------

    face_pil = Image.fromarray(
        face_rgb
    )


    # --------------------------------------------------------
    # Apply ResNet preprocessing
    # --------------------------------------------------------

    face_tensor = transform(
        face_pil
    )


    # Add batch dimension
    face_tensor = face_tensor.unsqueeze(
        0
    )


    # Move to GPU / CPU
    face_tensor = face_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Extract embedding
    # --------------------------------------------------------

    with torch.no_grad():

        embedding = model(
            face_tensor
        )


    # --------------------------------------------------------
    # GPU -> CPU -> NumPy
    # --------------------------------------------------------

    embedding = (
        embedding
        .cpu()
        .numpy()
        .flatten()
    )


    # --------------------------------------------------------
    # L2 normalization
    # --------------------------------------------------------

    embedding = embedding / (
        np.linalg.norm(embedding)
        + 1e-10
    )


    return embedding


# ============================================================
# 9. FUNCTION: RECOGNIZE FACE
# ============================================================

def recognize_face(face_bgr):

    """
    Recognize one detected face.

    Returns:
        predicted_person
        confidence
        top_predictions
    """

    # --------------------------------------------------------
    # Extract embedding
    # --------------------------------------------------------

    embedding = extract_embedding(
        face_bgr
    )


    embedding = embedding.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # SVM prediction
    # --------------------------------------------------------

    prediction = classifier.predict(
        embedding
    )[0]


    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    probabilities = classifier.predict_proba(
        embedding
    )[0]


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = probabilities[
        prediction
    ]


    # --------------------------------------------------------
    # Predicted person
    # --------------------------------------------------------

    predicted_person = selected_people[
        prediction
    ]


    # --------------------------------------------------------
    # Top 5 predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]


    top_predictions = []

    for idx in top_indices:

        name = selected_people[idx]

        probability = probabilities[idx]

        top_predictions.append(
            (
                name,
                probability
            )
        )


    return (
        predicted_person,
        confidence,
        top_predictions
    )


# ============================================================
# 10. OPEN WEBCAM
# ============================================================

print("\nStarting webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    raise RuntimeError(
        "\nCould not open webcam.\n"
        "Try changing CAMERA_INDEX from 0 to 1."
    )


# Camera resolution
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("\nWebcam started!")

print(
    "Press 'q' to quit."
)

print(
    "Stand directly in front of the camera."
)

print(
    "Watch the terminal for TOP PREDICTIONS."
)


# ============================================================
# 11. VARIABLES
# ============================================================

frame_count = 0

last_prediction = None

last_confidence = None


# ============================================================
# 12. REAL-TIME LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Read frame
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print(
            "Could not read frame."
        )

        break


    # --------------------------------------------------------
    # Mirror image
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(80, 80)
    )


    # --------------------------------------------------------
    # Process each face
    # --------------------------------------------------------

    for (x, y, w, h) in faces:


        # ====================================================
        # CREATE PADDED FACE CROP
        # ====================================================

        pad_x = int(
            w * FACE_PADDING
        )

        pad_y = int(
            h * FACE_PADDING
        )


        x1 = max(
            0,
            x - pad_x
        )

        y1 = max(
            0,
            y - pad_y
        )

        x2 = min(
            frame.shape[1],
            x + w + pad_x
        )

        y2 = min(
            frame.shape[0],
            y + h + pad_y
        )


        face = frame[
            y1:y2,
            x1:x2
        ]


        # ----------------------------------------------------
        # Validate crop
        # ----------------------------------------------------

        if face.size == 0:

            continue


        # ====================================================
        # RECOGNITION
        # ====================================================

        try:

            (
                predicted_person,
                confidence,
                top_predictions
            ) = recognize_face(
                face
            )


        except Exception as e:

            print(
                "\nRecognition error:",
                e
            )

            predicted_person = "Error"

            confidence = 0.0

            top_predictions = []


        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        frame_count += 1


        if frame_count % DEBUG_INTERVAL == 0:

            print("\n" + "=" * 60)

            print(
                "TOP PREDICTIONS"
            )

            print("=" * 60)


            for name, probability in top_predictions:

                print(
                    f"{name:<30}"
                    f"{probability * 100:>7.2f}%"
                )


            print("=" * 60)


        # ====================================================
        # APPLY UNKNOWN THRESHOLD
        # ====================================================

        if confidence >= CONFIDENCE_THRESHOLD:

            display_name = predicted_person

        else:

            display_name = "Unknown"


        # ====================================================
        # LABEL
        # ====================================================

        label = (
            f"{display_name} "
            f"({confidence * 100:.1f}%)"
        )


        # ====================================================
        # BOUNDING BOX
        # ====================================================

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2
        )


        # ====================================================
        # LABEL BACKGROUND
        # ====================================================

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            2
        )


        label_y1 = max(
            0,
            y - text_height - baseline - 10
        )


        cv2.rectangle(

            frame,

            (
                x,
                label_y1
            ),

            (
                x + text_width + 10,
                y
            ),

            (0, 255, 0),

            -1
        )


        # ====================================================
        # LABEL TEXT
        # ====================================================

        cv2.putText(

            frame,

            label,

            (
                x + 5,
                y - 7
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 0, 0),

            2
        )


        # ====================================================
        # DEBUG: DRAW PREDICTION BELOW FACE
        # ====================================================

        debug_text = (
            f"Raw: {predicted_person}"
        )


        cv2.putText(

            frame,

            debug_text,

            (
                x,
                y + h + 25
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        "Real-Time Face Recognition",

        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(
        1
    ) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# 13. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print(
    "\nWebcam stopped."
)

print(
    "Face recognition application closed."
)