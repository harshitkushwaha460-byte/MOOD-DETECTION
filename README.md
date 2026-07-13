# 😊 AI Mood Detection & Wellness Assistant

An AI-powered wellness assistant that detects a user's facial emotion from a captured or uploaded image and provides personalized mental wellness recommendations using **EfficientNet-B0**, **Retrieval-Augmented Generation (RAG)**, and the **Sarvam AI API**.

The project combines **Computer Vision**, **Deep Learning**, and **Generative AI** to create an intelligent assistant that not only recognizes emotions but also offers supportive, context-aware wellness guidance.<br>

---

# 📖 Project Overview

Mental health is an important aspect of overall well-being, yet many people struggle to recognize and manage their emotions.

This project aims to bridge that gap by building an AI-powered assistant capable of:

- Detecting facial emotions from images.
- Understanding the detected emotional state.
- Retrieving relevant wellness information from a curated knowledge base using RAG.
- Generating personalized recommendations with the Sarvam AI language model.

Instead of providing generic advice, the assistant uses relevant contextual information retrieved from PDF documents to produce more meaningful and personalized responses.
<br>
**The current model is a prototype trained on a limited dataset. My focus in this project was integrating computer vision with RAG and an LLM to build an end-to-end AI wellness assistant. I'm continuing to improve the emotion classifier through additional training and dataset refinement.**


---

# ✨ Features

- 📷 Capture image using webcam
- 📁 Upload an image from device
- 😊 Detect 7 different facial emotions
- ⚡ EfficientNet-B0 based emotion classification
- 📚 Retrieval-Augmented Generation (RAG)
- 🤖 Personalized AI wellness recommendations using Sarvam AI
- 📊 Emotion dashboard with confidence score
- 📥 Download emotion analysis report
- 💻 Interactive Streamlit web application

---

# 😊 Supported Emotions

The model can recognize the following emotions:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

---

# 🏗 Project Architecture

```
                User
                  │
                  ▼
     Capture Photo / Upload Image
                  │
                  ▼
        OpenCV Face Detection
                  │
                  ▼
      EfficientNet-B0 Emotion Model
                  │
                  ▼
         Predicted Emotion
                  │
                  ▼
      Retrieve Relevant Documents
           (FAISS + RAG)
                  │
                  ▼
        Sarvam AI Language Model
                  │
                  ▼
 Personalized Wellness Recommendation
                  │
                  ▼
        Emotion Dashboard & Report
```

---

# 🛠 Tech Stack

## Frontend

- Streamlit

## Computer Vision

- OpenCV

## Deep Learning

- TensorFlow
- EfficientNet-B0

## Retrieval-Augmented Generation

- LangChain
- FAISS
- Sentence Transformers

## Generative AI

- Sarvam AI API

## Other Libraries

- NumPy
- Pillow
- PyPDF
- Python Dotenv

---

# 📂 Project Structure

```
AI-Mood-Detection/
│
├── app.py
├── rag.py
├── sarvemkey.py
├── requirements.txt
├── packages.txt
│
├── models/
│   └── best_model.keras
│
├── vector_db/
│
├── pdfs/
│
├── .streamlit/
│   └── secrets.toml
│
└── README.md
```


---

# 📊 Output

The application provides:

- Detected facial emotion
- Prediction confidence
- Emotion dashboard
- AI-generated wellness recommendation
- Downloadable report

---

# 📚 Knowledge Base

The RAG pipeline retrieves information from curated PDF documents on topics such as:

- Mental wellness
- Stress management
- Meditation
- Mindfulness
- Positive thinking
- Healthy lifestyle
- Emotional well-being
- Relaxation techniques

---

# 🌟 Key Highlights

- Transfer Learning using EfficientNet-B0
- Real-time facial emotion recognition
- Retrieval-Augmented Generation (RAG)
- Context-aware AI recommendations
- Modern Streamlit user interface
- Downloadable emotion reports
- Modular and extensible architecture

---

# 🔮 Future Improvements


- 📈 Mood history tracking
- 📅 Daily and weekly mood analytics
- 🎤 Voice interaction
- 🌙 Dark mode

---

# 👨‍💻 Author

**Harshit Kushwaha**

B.Tech in Computer Science Engineering  
Punjab Engineering College (PEC), Chandigarh

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub. Your support helps improve and maintain the project.
