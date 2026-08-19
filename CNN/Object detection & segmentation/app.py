import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import torch

from PIL import Image
from ultralytics import YOLO
import io

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ObjectVision",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0b1020;
        color: #f5f7ff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #11182b;
        border-right: 1px solid #26304a;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #9ca8c7;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Cards */
    .info-card {
        background: #121a2e;
        border: 1px solid #26304a;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .info-card h3 {
        margin-top: 0;
    }

    /* Small badges */
    .badge {
        display: inline-block;
        background: #1d2945;
        color: #b9c7e8;
        padding: 6px 12px;
        border-radius: 20px;
        margin-right: 6px;
        font-size: 13px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        padding: 10px;
    }

    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "stage4_yolo11n_seg.pt"


@st.cache_resource
def load_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found:\n{os.path.abspath(model_path)}"
        )

    model = YOLO(model_path)

    return model


# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎯 ObjectVision</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered object detection, segmentation and object-aware media editing.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.markdown(
        '<span class="badge">YOLO11 Segmentation</span>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<span class="badge">80 COCO Classes</span>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<span class="badge">GPU Accelerated</span>'
        if torch.cuda.is_available()
        else '<span class="badge">CPU Mode</span>',
        unsafe_allow_html=True
    )

    st.divider()

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    image_size = st.select_slider(
        "Inference size",
        options=[320, 416, 512, 640],
        value=320
    )

    st.divider()

    st.markdown("### Model")

    st.caption(
        os.path.abspath(MODEL_PATH)
    )

    if torch.cuda.is_available():

        st.success(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        st.warning(
            "CUDA GPU not detected. CPU mode."
        )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model(MODEL_PATH)

except Exception as e:

    st.error(str(e))

    st.info(
        "Place stage4_yolo11n_seg.pt in the same folder as app.py."
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

image_tab, video_tab = st.tabs(
    ["🖼️ Image", "🎬 Video"]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_mask(result, index, width, height):

    if result.masks is None:
        return None

    mask = (
        result.masks.data[index]
        .cpu()
        .numpy()
    )

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    return mask > 0.5


def get_objects(result):

    objects = []

    if result.boxes is None:
        return objects

    for i, box in enumerate(result.boxes):

        class_id = int(box.cls.item())

        confidence = float(
            box.conf.item()
        )

        objects.append(
            {
                "index": i,
                "name": result.names[class_id],
                "confidence": confidence
            }
        )

    return objects


def blur_masked_object(image, mask):

    blurred = cv2.GaussianBlur(
        image,
        (41, 41),
        0
    )

    output = image.copy()

    output[mask] = blurred[mask]

    return output


def highlight_masked_object(image, mask):

    output = (
        image.astype(np.float32) * 0.25
    ).astype(np.uint8)

    output[mask] = image[mask]

    return output


def extract_masked_object(image, mask):

    rgba = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGBA
    )

    rgba[:, :, 3] = (
        mask.astype(np.uint8) * 255
    )

    return rgba


def crop_masked_object(image, mask):

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    crop = image[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    crop_mask = mask[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGBA
    )

    rgba[:, :, 3] = (
        crop_mask.astype(np.uint8) * 255
    )

    return rgba


# ============================================================
# IMAGE TAB
# ============================================================

with image_tab:

    st.markdown(
        "### 🖼️ Object-aware image editing"
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_upload"
    )

    if uploaded_image:

        image_pil = Image.open(
            uploaded_image
        ).convert("RGB")

        image_rgb = np.array(
            image_pil
        )

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.markdown(
                "#### Original"
            )

            st.image(
                image_pil,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Run detection
        # ----------------------------------------------------

        with st.spinner(
            "Detecting objects..."
        ):

            result = model.predict(
                image_bgr,
                conf=confidence,
                imgsz=image_size,
                device=DEVICE,
                verbose=False
            )[0]

        objects = get_objects(
            result
        )

        # ----------------------------------------------------
        # Detection summary
        # ----------------------------------------------------

        st.divider()

        if not objects:

            st.warning(
                "No objects detected. "
                "Try lowering the confidence threshold."
            )

        else:

            st.success(
                f"{len(objects)} object(s) detected."
            )

            object_labels = []

            for obj in objects:

                object_labels.append(
                    f"{obj['name']} "
                    f"({obj['confidence']:.2f})"
                )

            selected_object = st.selectbox(
                "Select object",
                range(len(objects)),
                format_func=lambda i:
                    object_labels[i]
            )

            operation = st.selectbox(
                "Choose operation",
                [
                    "Detect",
                    "Blur",
                    "Highlight",
                    "Extract",
                    "Crop"
                ]
            )

            # ------------------------------------------------
            # Process selected object
            # ------------------------------------------------

            obj = objects[
                selected_object
            ]

            mask = get_mask(
                result,
                obj["index"],
                image_bgr.shape[1],
                image_bgr.shape[0]
            )

            if mask is None:

                st.error(
                    "Segmentation mask unavailable."
                )

            else:

                if operation == "Detect":

                    output_bgr = result.plot()

                    output_rgb = cv2.cvtColor(
                        output_bgr,
                        cv2.COLOR_BGR2RGB
                    )

                    output_image = Image.fromarray(
                        output_rgb
                    )

                    filename = "detected.jpg"

                elif operation == "Blur":

                    output_bgr = (
                        blur_masked_object(
                            image_bgr,
                            mask
                        )
                    )

                    output_rgb = cv2.cvtColor(
                        output_bgr,
                        cv2.COLOR_BGR2RGB
                    )

                    output_image = Image.fromarray(
                        output_rgb
                    )

                    filename = "blurred.jpg"

                elif operation == "Highlight":

                    output_bgr = (
                        highlight_masked_object(
                            image_bgr,
                            mask
                        )
                    )

                    output_rgb = cv2.cvtColor(
                        output_bgr,
                        cv2.COLOR_BGR2RGB
                    )

                    output_image = Image.fromarray(
                        output_rgb
                    )

                    filename = "highlighted.jpg"

                elif operation == "Extract":

                    output_rgba = (
                        extract_masked_object(
                            image_bgr,
                            mask
                        )
                    )

                    output_image = Image.fromarray(
                        output_rgba
                    )

                    filename = "extracted.png"

                else:

                    output_rgba = (
                        crop_masked_object(
                            image_bgr,
                            mask
                        )
                    )

                    if output_rgba is None:

                        st.error(
                            "Could not create crop."
                        )

                        st.stop()

                    output_image = Image.fromarray(
                        output_rgba
                    )

                    filename = "cropped.png"

                # ------------------------------------------------
                # Display result
                # ------------------------------------------------

                with col2:

                    st.markdown(
                        "#### Result"
                    )

                    st.image(
                        output_image,
                        use_container_width=True
                    )

                # ------------------------------------------------
                # Download
                # ------------------------------------------------

                

                output_bytes = io.BytesIO()

                if filename.endswith(".png"):

                    output_image.save(
                        output_bytes,
                        format="PNG"
                    )

                    mime_type = "image/png"

                else:

                    output_image.convert(
                        "RGB"
                    ).save(
                        output_bytes,
                        format="JPEG",
                        quality=95
                    )

                    mime_type = "image/jpeg"

                output_bytes.seek(0)

                st.download_button(
                    label="⬇️ Download result",
                    data=output_bytes.getvalue(),
                    file_name=filename,
                    mime=mime_type
                )
                    


# ============================================================
# VIDEO TAB
# ============================================================

with video_tab:

    st.markdown(
        "### 🎬 Object-aware video editing"
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_upload"
    )

    if uploaded_video:

        # ----------------------------------------------------
        # Save temporary input video
        # ----------------------------------------------------

        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_temp.write(
            uploaded_video.read()
        )

        input_temp.close()

        # ----------------------------------------------------
        # Video settings
        # ----------------------------------------------------

        video_operation = st.selectbox(
            "Choose video operation",
            [
                "Track",
                "Blur",
                "Highlight"
            ]
        )

        target_class = st.text_input(
            "Object to process",
            value="person",
            help=(
                "Examples: person, car, dog, cat, "
                "bicycle, laptop, bottle"
            )
        )

        process_video_button = st.button(
            "▶️ Process video",
            type="primary"
        )

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        if process_video_button:

            output_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_temp.name

            output_temp.close()

            cap = cv2.VideoCapture(
                input_temp.name
            )

            if not cap.isOpened():

                st.error(
                    "Could not open uploaded video."
                )

                st.stop()

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            cap.release()

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress = st.progress(
                0
            )

            status = st.empty()

            start_time = time.time()

            # ------------------------------------------------
            # YOLO tracking stream
            # ------------------------------------------------

            results_stream = model.track(
                source=input_temp.name,
                tracker="bytetrack.yaml",
                conf=confidence,
                imgsz=image_size,
                device=DEVICE,
                stream=True,
                persist=True,
                verbose=False
            )

            processed = 0

            for result in results_stream:

                processed += 1

                frame = (
                    result.orig_img.copy()
                )

                # ============================================
                # TRACK
                # ============================================

                if video_operation == "Track":

                    output = result.plot()

                # ============================================
                # BLUR / HIGHLIGHT
                # ============================================

                else:

                    output = frame.copy()

                    if (
                        result.boxes is not None
                        and result.masks is not None
                    ):

                        for i, box in enumerate(
                            result.boxes
                        ):

                            class_id = int(
                                box.cls.item()
                            )

                            class_name = (
                                result.names[
                                    class_id
                                ]
                            )

                            object_conf = float(
                                box.conf.item()
                            )

                            if (
                                class_name.lower()
                                != target_class.lower()
                            ):
                                continue

                            if (
                                object_conf
                                < confidence
                            ):
                                continue

                            mask = get_mask(
                                result,
                                i,
                                width,
                                height
                            )

                            if mask is None:
                                continue

                            # ------------------------------
                            # BLUR
                            # ------------------------------

                            if (
                                video_operation
                                == "Blur"
                            ):

                                blurred = (
                                    cv2.GaussianBlur(
                                        frame,
                                        (41, 41),
                                        0
                                    )
                                )

                                output[mask] = (
                                    blurred[mask]
                                )

                            # ------------------------------
                            # HIGHLIGHT
                            # ------------------------------

                            elif (
                                video_operation
                                == "Highlight"
                            ):

                                darkened = (
                                    frame.astype(
                                        np.float32
                                    ) * 0.25
                                ).astype(
                                    np.uint8
                                )

                                output = darkened

                                output[mask] = (
                                    frame[mask]
                                )

                writer.write(output)

                # Progress
                if total_frames > 0:

                    progress_value = min(
                        processed / total_frames,
                        1.0
                    )

                    progress.progress(
                        progress_value
                    )

                elapsed = (
                    time.time() - start_time
                )

                if elapsed > 0:

                    processing_fps = (
                        processed / elapsed
                    )

                    remaining = (
                        total_frames
                        - processed
                    )

                    eta = (
                        remaining
                        / processing_fps
                        if processing_fps > 0
                        else 0
                    )

                    status.write(
                        f"Processing: "
                        f"{processed}/{total_frames} "
                        f"frames • "
                        f"{processing_fps:.1f} FPS • "
                        f"ETA: {eta:.0f}s"
                    )

            writer.release()

            elapsed = (
                time.time() - start_time
            )

            progress.progress(1.0)

            status.success(
                f"Completed {processed} frames "
                f"in {elapsed:.1f} seconds."
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.markdown(
                "#### Processed video"
            )

            try:

                video_file = open(
                    output_path,
                    "rb"
                ).read()

                st.video(
                    video_file
                )

                st.download_button(
                    label="⬇️ Download processed video",
                    data=video_file,
                    file_name=(
                        f"{video_operation.lower()}_"
                        f"{target_class}.mp4"
                    ),
                    mime="video/mp4"
                )

            except Exception as e:

                st.warning(
                    "Video was created, but "
                    f"preview could not be loaded: {e}"
                )

            # Cleanup
            try:
                os.unlink(input_temp.name)
            except:
                pass