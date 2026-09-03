# ============================================================
# FACE QUALITY FILTER
# ============================================================

import cv2


def check_face_quality(
    face_bgr,
    min_size=60,
    blur_threshold=35
):

    """
    Basic quality checks before recognition.

    Returns:
        True  -> suitable face
        False -> reject
    """

    if face_bgr is None:

        return False


    height, width = face_bgr.shape[:2]


    # --------------------------------------------------------
    # Size check
    # --------------------------------------------------------

    if width < min_size:

        return False

    if height < min_size:

        return False


    # --------------------------------------------------------
    # Blur check
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2GRAY
    )


    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )


    blur_score = laplacian.var()


    if blur_score < blur_threshold:

        return False


    return True