import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import os
import json
import re

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

        if 'choices' in res and len(res['choices']) > 0:
            ocr_output = res['choices'][0]['message']['content']
            if detect_language(ocr_output) != 'id':
                translated_result = translate_text(ocr_output)
                return translated_result
            else:
                return ocr_output
        else:
            return None

    except requests.RequestException as e:
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

        if 'choices' in res and len(res['choices']) > 0:
            translated_text = res['choices'][0]['message']['content']
            return translated_text
        else:
            return None

    except requests.RequestException as e:
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

        if 'choices' in res and len(res['choices']) > 0:
            detected_lang = res['choices'][0]['message']['content']
            return detected_lang.strip().lower()  # Ensure lowercase for consistency
        else:
            return None

    except requests.RequestException as e:
        return None

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
            ocr_result = ocr_image(file_content)
            if ocr_result:
                st.write(f'OCR Output: {ocr_result}')  # Display OCR output

        elif file_type == 'application/pdf':
            images = pdf_to_images(file_content)
            full_ocr_result = ""
            for image in images:
                ocr_result = ocr_image(image)
                if ocr_result:
                    full_ocr_result += ocr_result + "\n"

            if full_ocr_result:
                st.write(f'Full OCR Output: {full_ocr_result}')  # Display full OCR output

    st.markdown("<p style='text-align: center;'>Powered by OpenAI</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
