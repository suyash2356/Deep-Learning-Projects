import cv2

from utils.recognition import recognize_face


IMAGE_PATH = r"E:\Pyhton code\Deep Learning\CNN\Face Recognition Model\data\raw\test.jpg"


image = cv2.imread(
    IMAGE_PATH
)


if image is None:

    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


name, confidence, class_id, known = recognize_face(
    image
)


print()
print("=" * 50)
print("RECOGNITION TEST")
print("=" * 50)

print("Name:", name)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)

print(
    "Class ID:",
    class_id
)

print(
    "Known:",
    known
)