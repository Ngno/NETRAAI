import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import re
import json

# Load environment variables
load_dotenv()

# Credentials
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
SPEECH_KEY = os.getenv('SPEECH_KEY')
SERVICE_REGION = os.getenv('SERVICE_REGION')

print(f'SPEECH_KEY: {SPEECH_KEY}')
print(f'SERVICE_REGION: {SERVICE_REGION}')

# Function to perform OCR using OpenAI
def perform_transcription(image_content):
    image_data = base64.b64encode(image_content).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

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
            transcription_output = res['choices'][0]['message']['content']
            return transcription_output
        else:
            print('No valid response from API')
            return None
    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        return None

# Function to translate text to Bahasa Indonesia using OpenAI
def translate_to_indonesian(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    prompt = f"Terjemahkan teks berikut ke Bahasa Indonesia:\n\n{text}"
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

# Function to clean and format text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\\', '')
    text = re.sub(r'(?<!\.)\n(?!\.)', '. ', text)
    if text and text[-1] not in {'.', '!', '?'}:
        text += '.'
    return text

# Function to convert PDF to images
def pdf_to_images(pdf_content):
    pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        images.append(img_byte_arr.getvalue())
    return images

# Streamlit app
def main():
    st.title("NETRA AI")

    st.header("Ubah Materi Teks dan Gambar Menjadi Audio")
    uploaded_file = st.file_uploader("Pilih file PDF atau Gambar", type=['pdf', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_content = uploaded_file.read()
        file_type = uploaded_file.type

        if file_type.startswith('image/'):
            transcription_output = perform_transcription(file_content)
            if transcription_output:
                st.write(f'Transcription Output: {transcription_output}')  # Debugging statement to check the transcription output
                translated_result = translate_to_indonesian(transcription_output)
                if translated_result:
                    st.write(f'Translated Output: {translated_result}')
                else:
                    st.error("Failed to translate the transcription output.")
            else:
                st.error("Failed to perform transcription.")

        elif file_type == 'application/pdf':
            images = pdf_to_images(file_content)
            full_transcription_output = ""
            for image in images:
                transcription_output = perform_transcription(image)
                if transcription_output:
                    full_transcription_output += transcription_output + "\n"
            if full_transcription_output:
                st.write(f'Full Transcription Output: {full_transcription_output}')  # Debugging statement to check the full transcription output
                translated_result = translate_to_indonesian(full_transcription_output)
                if translated_result:
                    st.write(f'Translated Output: {translated_result}')
                else:
                    st.error("Failed to translate the transcription output from PDF.")
            else:
                st.error("Failed to perform transcription from PDF.")

        else:
            st.warning("Format file tidak didukung. Harap unggah PDF atau gambar.")

    st.markdown("<p style='text-align: center;'>Powered by OpenAI and Azure Speech SDK</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
