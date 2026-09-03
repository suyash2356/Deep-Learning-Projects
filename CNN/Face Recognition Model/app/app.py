
# ============================================================
# AI FACE RECOGNITION SYSTEM
# Stage 4
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


# ============================================================
# PATH
# ============================================================

APP_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = APP_DIR.parent

sys.path.insert(
    0,
    str(APP_DIR)
)


from face_engine import (
    load_model,
    create_detector,
    process_image
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="AI Face Recognition",

    page_icon="👁️",

    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "👁️ AI Face Recognition System"
)

st.caption(
    "YuNet + OpenCV Enhancement + "
    "Custom ResNet-50 Face Recognition"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def initialize_system():

    model, selected_people, checkpoint = (
        load_model()
    )

    detector = create_detector()

    return (
        model,
        detector,
        selected_people,
        checkpoint
    )


try:

    (
        model,
        detector,
        selected_people,
        checkpoint
    ) = initialize_system()

except Exception as e:

    st.error(
        f"Could not initialize AI system:\n\n{e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ System Information"
)

st.sidebar.write(
    f"Classes: {len(selected_people)}"
)

st.sidebar.write(
    "Embedding: 512-D"
)

st.sidebar.write(
    "Detector: YuNet"
)

st.sidebar.write(
    "Backbone: ResNet-50"
)


# ============================================================
# INPUT MODE
# ============================================================

mode = st.radio(

    "Select input source",

    [
        "📷 Image",
        "🎥 Video",
        "🔴 Live Camera"
    ],

    horizontal=True
)


# ============================================================
# DRAW RESULT
# ============================================================

def draw_results(
    image,
    results
):

    output = image.copy()


    for result in results:

        x, y, w, h = (
            result["bbox"]
        )

        identity = result[
            "identity"
        ]

        confidence = result[
            "confidence"
        ]


        if identity == "Unknown":

            label = (
                f"Unknown "
                f"({confidence * 100:.1f}%)"
            )

        else:

            label = (
                f"{identity} "
                f"({confidence * 100:.1f}%)"
            )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(

            output,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        cv2.putText(

            output,

            label,

            (x, max(25, y - 10)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2
        )


    return output


# ============================================================
# IMAGE MODE
# ============================================================

if mode == "📷 Image":

    st.subheader(
        "Upload an image"
    )


    uploaded_file = st.file_uploader(

        "Choose an image",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )


        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "Could not read image."
            )

            st.stop()


        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        with st.spinner(
            "Detecting and recognizing faces..."
        ):

            results = process_image(

                image,

                model,

                detector,

                selected_people
            )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        result_image = draw_results(
            image,
            results
        )


        result_rgb = cv2.cvtColor(

            result_image,

            cv2.COLOR_BGR2RGB
        )


        st.image(

            result_rgb,

            caption="Recognition Result",

            use_container_width=True
        )


        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.subheader(
            "Recognition Results"
        )


        if len(results) == 0:

            st.warning(
                "No reliable face detected."
            )

        else:

            for i, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"### Face {i}"
                )

                st.write(
                    f"Identity: "
                    f"**{result['identity']}**"
                )

                st.write(
                    f"Recognition confidence: "
                    f"**{result['confidence'] * 100:.2f}%**"
                )

                st.write(
                    f"Detection confidence: "
                    f"**{result['detection_confidence'] * 100:.2f}%**"
                )


                with st.expander(
                    "Top predictions"
                ):

                    for prediction in (
                        result[
                            "top_predictions"
                        ]
                    ):

                        st.write(

                            f"{prediction['name']} "
                            f"— "
                            f"{prediction['confidence'] * 100:.2f}%"
                        )


# ============================================================
# VIDEO MODE
# ============================================================

elif mode == "🎥 Video":

    st.subheader(
        "Upload a video"
    )


    uploaded_video = st.file_uploader(

        "Choose a video",

        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )


    if uploaded_video is not None:

        video_path = (
            PROJECT_ROOT
            / "app"
            / "_uploaded_video.mp4"
        )


        with open(
            video_path,
            "wb"
        ) as f:

            f.write(
                uploaded_video.read()
            )


        st.video(
            str(video_path)
        )


        st.info(
            "Video upload is working. "
            "Frame-by-frame recognition "
            "and tracking will be added "
            "in Stage 5."
        )


# ============================================================
# LIVE CAMERA MODE
# ============================================================

elif mode == "🔴 Live Camera":

    st.subheader(
        "Live camera"
    )


    st.info(
        "Use the camera button below "
        "to provide live frames."
    )


    camera_image = st.camera_input(
        "Take a picture from your camera"
    )


    if camera_image is not None:

        file_bytes = np.asarray(

            bytearray(
                camera_image.read()
            ),

            dtype=np.uint8
        )


        image = cv2.imdecode(

            file_bytes,

            cv2.IMREAD_COLOR
        )


        results = process_image(

            image,

            model,

            detector,

            selected_people
        )


        result_image = draw_results(

            image,

            results
        )


        result_rgb = cv2.cvtColor(

            result_image,

            cv2.COLOR_BGR2RGB
        )


        st.image(

            result_rgb,

            caption="Camera Recognition",

            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Stage 4 — AI Face Recognition System"
)
