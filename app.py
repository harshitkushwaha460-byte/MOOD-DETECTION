import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

from sarvemkey import generate_response

# ------------------------------------
# Page Configuration
# ------------------------------------

st.set_page_config(
    page_title="AI Mood Detection",
    page_icon="😊",
    layout="wide"
)

st.title("😊 AI Mood Detection & Wellness Assistant")

st.subheader(
    """Capture your face using the camera,detect your emotion,and receive personalized AI wellness advice.
"""
)


# Load CNN Model


@st.cache_resource
def load_cnn():

    return load_model(
        "models/best_model.keras"
    )

model = load_cnn()


# Emotion Labels


emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


# Face Detector


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Session State


if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

if "emotion" not in st.session_state:
    st.session_state.emotion = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "reply" not in st.session_state:
    st.session_state.reply = ""

# Select Image Source


st.header("📷 Select Image Source")

option = st.radio(
    "How would you like to provide your image?",
    (
        "📷 Take a Photo",
        "📁 Upload an Image"
    )
)

picture = None


# Camera


if option == "📷 Take a Photo":

    picture = st.camera_input(
        "Capture your face"
    )


# Upload


elif option == "📁 Upload an Image":

    picture = st.file_uploader(
        "Upload a clear face image",
        type=["jpg", "jpeg", "png"]
    )


# Save Image


if picture is not None:

    image = Image.open(picture)

    st.session_state.captured_image = image


# Preview Captured Image

continue_btn = False
retake_btn = False
if st.session_state.captured_image is not None:
   
    st.subheader("Captured Image")
    st.image(
    st.session_state.captured_image,
    width=400)
    continue_btn = st.button( "✅ Continue",
        use_container_width=True
    )

    # Retake Photo
    if retake_btn:

        st.session_state.captured_image = None
        st.session_state.emotion = None
        st.session_state.confidence = None
        st.session_state.reply = ""

        st.rerun()

# Predict Emotion


if continue_btn:

    # Convert PIL Image to OpenCV format
    image = np.array(st.session_state.captured_image)

    # Convert RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # No Face Found
    if len(faces) == 0:

        st.error("❌ No face detected. Please retake the photo.")

    else:

        # Select the Largest Face
        (x, y, w, h) = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        # Crop Face
        face = image[y:y+h, x:x+w]

        # Resize for CNN
        face = cv2.resize(
            face,
            (224, 224)
        )
        # EfficientNet preprocessing
        face = preprocess_input(face.astype(np.float32))
        face = np.expand_dims(face, axis=0)

        # Normalize
        # face = face.astype("float32") / 255.0

        # Reshape
        # face = np.expand_dims(face, axis=0)
        #face = np.expand_dims(face, axis=-1)

        # Predict Emotion
        prediction = model.predict(
            face,
            verbose=0
        )

        emotion_index = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        emotion = emotion_dict[emotion_index]

        # Save Results
        st.session_state.emotion = emotion

        st.session_state.confidence = confidence

        # Draw Rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Display Emotion
        cv2.putText(
            image,
            f"{emotion} ({confidence:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Convert Back to RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Prediction Result")

        st.image(
            image,
            caption="Detected Emotion",
            use_container_width=True
        )

        st.success(
            f"Emotion : {emotion}"
        )

        st.info(
            f"Confidence : {confidence:.2f}%"
        )

# AI Recommendation


if st.session_state.emotion is not None:

    st.divider()

    st.subheader("🧠 AI Wellness Coach")

    st.write(
        f"Detected Emotion : **{st.session_state.emotion}**"
    )

    st.write(
        f"Confidence : **{st.session_state.confidence:.2f}%**"
    )

    # Generate Recommendation Button
    if st.button("💡 Get AI Recommendation"):

        with st.spinner("Generating AI Recommendation..."):

            try:

                st.session_state.reply = generate_response(
                    st.session_state.emotion
                )

            except Exception as e:

                st.error(f"Error: {e}")

    # Display Recommendation
    if st.session_state.reply:

        st.success("AI Recommendation")

        st.info(st.session_state.reply)             

# Emotion Dashboard


if st.session_state.emotion is not None:

    st.divider()

    emoji = {
        "Happy": "😄",
        "Sad": "😢",
        "Angry": "😠",
        "Fear": "😨",
        "Surprise": "😲",
        "Neutral": "😐",
        "Disgust": "🤢"
    }

    description = {
        "Happy": "You appear to be in a positive mood.",
        "Sad": "You may be feeling low today.",
        "Angry": "You seem frustrated or angry.",
        "Fear": "You may be feeling anxious.",
        "Surprise": "You look surprised.",
        "Neutral": "You appear calm and neutral.",
        "Disgust": "You seem uncomfortable."
    }

    st.markdown("## 📊 Emotion Dashboard")

    col1, col2 = st.columns([1, 4])

    with col1:
        st.markdown(
            f"<h1 style='text-align:center'>{emoji[st.session_state.emotion]}</h1>",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"### {st.session_state.emotion}"
        )

        st.write(
            description[st.session_state.emotion]
        )

        st.progress(
            min(
                int(st.session_state.confidence),
                100
            )
        )

        st.write(
            f"Confidence : {st.session_state.confidence:.2f}%"
        )

# AI Wellness Report


if st.session_state.reply:

    st.divider()

    st.markdown("## 🤖 Personalized Wellness Report")

    with st.container(border=True):

        st.markdown("### 😊 Detected Emotion")

        st.success(st.session_state.emotion)

        st.markdown("### 📈 Confidence")

        st.progress(
            min(int(st.session_state.confidence),100)
        )

        st.write(
            f"{st.session_state.confidence:.2f}%"
        )

        st.markdown("---")

        st.markdown("### 💡 AI Recommendation")

        st.write(
            st.session_state.reply
        )


# Download Report


        report = f"""
AI Mood Detection Report

Emotion:
{st.session_state.emotion}

Confidence:
{st.session_state.confidence:.2f}%

Recommendation:

{st.session_state.reply}
"""

        st.download_button(

            label="📥 Download Report",

            data=report,

            file_name="Mood_Report.txt",

            mime="text/plain"

        )


# Reset App


st.divider()

if st.button("🔄 Analyze Another Photo"):

    st.session_state.captured_image = None

    st.session_state.emotion = None

    st.session_state.confidence = None

    st.session_state.reply = ""

    st.rerun()                   