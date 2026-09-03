import cv2

from utils.face_quality import check_face_quality


IMAGE_PATH =  r"E:\Pyhton code\Deep Learning\CNN\Face Recognition Model\data\raw\test.jpg"


image = cv2.imread(
    IMAGE_PATH
)


if image is None:

    raise FileNotFoundError(
        IMAGE_PATH
    )


result = check_face_quality(
    image
)


print(
    "Face quality:",
    result
)