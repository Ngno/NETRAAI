import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import os
import re
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Credentials
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
SPEECH_KEY = os.getenv('SPEECH_KEY')
SERVICE_REGION = os.getenv('SERVICE_REGION')

print(f'SPEECH_KEY: {SPEECH_KEY}')
print(f'SERVICE_REGION: {SERVICE_REGION}')

# Function to perform OCR using OpenAI
def ocr_image(image_content):
    image_data = base64.b64encode(image_content).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    prompt = "Read and explain this image for visually impaired students. Use English."
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
            ocr_text = res['choices'][0]['message']['content']
            return ocr_text
        else:
            print('No valid response from API')
            return ''  # Return empty string if no valid response
    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        return ''  # Return empty string on request exception

# Function to clean and format text
def clean_text(text):
    if text:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\\', '')
        text = re.sub(r'(?<!\.)\n(?!\.)', '. ', text)
        if text and text[-1] not in {'.', '!', '?'}:
            text += '.'
        return text
    else:
        return ''

# Function to translate text to Bahasa Indonesia using OpenAI
def translate_text(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    # Check if the text is already in Bahasa Indonesia
    if detect_language(text) == 'id':
        return text

    prompt = f"Please translate the following text to Bahasa Indonesia:\n\n{text}"
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

    prompt = f"Detect the language of the following text:\n\n{text}"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 50,  # Adjust max tokens based on the expected length of text
    }

    try:
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview",
            headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        print(f'Detect Language API Response: {json.dumps(res, indent=2)}')  # Detailed API response for debugging

        if 'choices' in res and len(res['choices']) > 0:
            detected_lang = res['choices'][0]['message']['content']
            return detected_lang
        else:
            print('No valid response from API')
            return None
    except requests.RequestException as e:
        print(f"Failed to make the request. Error: {e}")
        return None

# Streamlit app
def main():
    st.title("NETRA AI")

    # Centering the main content
    st.header("Ubah Materi Teks dan Gambar Menjadi Audio")
    uploaded_file = st.file_uploader("Pilih file PDF atau Gambar", type=['pdf', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_content = uploaded_file.read()
        file_type = uploaded_file.type

        if file_type.startswith('image/'):
            # Perform OCR on image
            ocr_result = ocr_image(file_content)

            # Clean and format OCR result
            clean_ocr_result = clean_text(ocr_result)
            st.write(f'OCR Output: {clean_ocr_result}')  # Debugging statement to check OCR output

            # Translate OCR result to Bahasa Indonesia
            if clean_ocr_result:
                translated_result = translate_text(clean_ocr_result)
                if translated_result:
                    st.write(f'Translated Output: {translated_result}')  # Debugging statement to check translation output
                else:
                    st.error("Failed to translate OCR output.")
            else:
                st.error("Failed to perform OCR.")

        elif file_type == 'application/pdf':
            # Process PDF pages to images and perform OCR on each page
            images = pdf_to_images(file_content)
            full_ocr_result = ""
            for image in images:
                ocr_result = ocr_image(image)
                if ocr_result:
                    full_ocr_result += ocr_result + "\n"

            # Clean and format full OCR result
            clean_full_ocr_result = clean_text(full_ocr_result)
            st.write(f'Full OCR Output: {clean_full_ocr_result}')  # Debugging statement to check full OCR output

            # Translate full OCR result to Bahasa Indonesia
            if clean_full_ocr_result:
                translated_result = translate_text(clean_full_ocr_result)
                if translated_result:
                    st.write(f'Translated Output: {translated_result}')  # Debugging statement to check translation output
                else:
                    st.error("Failed to translate full OCR output.")
            else:
                st.error("Failed to perform OCR from PDF.")

        else:
            st.warning("Format file tidak didukung. Harap unggah PDF atau gambar.")

    st.markdown("<p style='text-align: center;'>Powered by OpenAI and Azure Speech SDK</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
