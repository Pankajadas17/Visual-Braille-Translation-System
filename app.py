import streamlit as st
from main import process_braille_image, enhance_with_llm, text_to_speech
import os

st.title("🔠 Visual Braille Translator")

uploaded_file = st.file_uploader("Upload Braille Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save uploaded image
    image_path = "uploaded_braille.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Step 1: OCR from Braille image
        decoded_text = process_braille_image(image_path)
        st.subheader("📜 Decoded Braille Text")
        st.text(decoded_text)

        # Step 2: LLM Enhancement
        enhanced = enhance_with_llm(decoded_text)
        st.subheader("✨ Enhanced Text with LLM")
        st.text(enhanced)

        # Step 3: TTS
        audio_path = text_to_speech(enhanced)
        st.subheader("🔊 Text-to-Speech Output")
        audio_file = open(audio_path, 'rb')
        st.audio(audio_file.read(), format='audio/mp3')

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
