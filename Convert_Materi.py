import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import os
import json

# Credentials (ensure these are set in your environment variables or replace with actual values)
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

# Function to perform OCR using OpenAI
def ocr_image(image_content):
    image_data = base64.b64encode(image_content).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    # Prompt in Indonesian for OCR
    prompt = "Bacakan dan jelaskan gambar ini untuk murid tuna netra. Gunakan bahasa Indonesia."
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\n\n![image](data:image/jpeg;base64,{image_data})"
            }
        ],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 800,
    }

    try:
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
            headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        print(f'API Response: {json.dumps(res, indent=2)}')  # Detailed API response for debugging

        if 'choices' in res and len(res['choices']) > 0:
            ocr_output = res['choices'][0]['message']['content']
            if detect_language(ocr_output) != 'id':
                translated_result = translate_text(ocr_output)
                return translated_result
            else:
                return ocr_output
        else:
            print('No valid response from API')
            return None

    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        return None

# Function to translate text to Bahasa Indonesia using OpenAI
def translate_text(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    prompt = f"Silakan terjemahkan teks berikut ke Bahasa Indonesia:\n\n{text}"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 800,
    }

    try:
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
            headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        print(f'Translation API Response: {json.dumps(res, indent=2)}')  # Detailed API response for debugging

        if 'choices' in res and len(res['choices']) > 0:
            translated_text = res['choices'][0]['message']['content']
            return translated_text
        else:
            print('No valid response from API')
            return None

    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        return None

# Function to detect language using OpenAI
def detect_language(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    prompt = f"Deteksi bahasa dari teks berikut:\n\n{text}"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 50,
    }

    try:
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
            headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        print(f'Language Detection API Response: {json.dumps(res, indent=2)}')  # Detailed API response for debugging

        if 'choices' in res and len(res['choices']) > 0:
            detected_lang = res['choices'][0]['message']['content']
            return detected_lang.strip().lower()  # Ensure lowercase for consistency
        else:
            print('No valid response from API')
            return None

    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        return None

# Streamlit app
def main():
    st.title("NETRA AI")

    st.header("Ubah Materi Teks dan Gambar Menjadi Audio")
    uploaded_file = st.file_uploader("Pilih file PDF atau Gambar", type=['pdf', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_content = uploaded_file.read()
        file_type = uploaded_file.type

        if file_type.startswith('image/'):
            ocr_result = ocr_image(file_content)
            if ocr_result:
                st.write(f'OCR Output: {ocr_result}')  # Debugging statement to check the OCR output
                # Assuming you have a function to convert text to speech
                # audio_file = text_to_speech(ocr_result)
                # st.audio(audio_file, format='audio/wav')

        elif file_type == 'application/pdf':
            images = pdf_to_images(file_content)
            full_ocr_result = ""
            for image in images:
                ocr_result = ocr_image(image)
                if ocr_result:
                    full_ocr_result += ocr_result + "\n"

            if full_ocr_result:
                st.write(f'Full OCR Output: {full_ocr_result}')  # Debugging statement to check the full OCR output
                # Assuming you have a function to convert text to speech
                # audio_file = text_to_speech(full_ocr_result)
                # st.audio(audio_file, format='audio/wav')

    st.markdown("<p style='text-align: center;'>Powered by OpenAI</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
