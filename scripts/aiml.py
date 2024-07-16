import os
import json
import time
import base64
import requests

from openai import AzureOpenAI
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk

from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
LANGUAGE_KEY = os.getenv('LANGUAGE_KEY')
LANGUAGE_ENDPOINT = os.getenv('LANGUAGE_ENDPOINT')
SPEECH_KEY = os.getenv('SPEECH_KEY')
SERVICE_REGION = os.getenv('SERVICE_REGION')


def explain_image(encoded_image):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Kamu adalah seorang guru di sekolah luar biasa (SLB). Saat ini kamu sedang mengadakan ujian. Peserta didik memiliki keterbatasan penglihatan (tuna netra). Oleh karena itu, kamu harus menjelaskan soal ujian berikut ini dalam bentuk narasi.\n\nIngat, hanya sampaikan soal ujian nya saja, tidak perlu di jawab."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            },
        ],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 800
    }

    try:
        GPT4V_ENDPOINT = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    res = response.json()
    return res['choices'][0]['message']['content']


def tts(content, audio_path=None):
    if audio_path is None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        audio_path = f"outputs/outputs-speech-{timestr}.wav"
    
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)

    if not LANGUAGE_KEY or not LANGUAGE_ENDPOINT:
        raise ValueError("Please set the LANGUAGE_KEY and LANGUAGE_ENDPOINT environment variables.")

    def authenticate_client():
        ta_credential = AzureKeyCredential(LANGUAGE_KEY)
        text_analytics_client = TextAnalyticsClient(
            endpoint=LANGUAGE_ENDPOINT,
            credential=ta_credential
        )
        return text_analytics_client

    client = authenticate_client()

    def language_detection(client, content):
        try:
            response = client.detect_language(documents=[content], country_hint='us')[0]
            return response.primary_language.iso6391_name
        except Exception as err:
            print(f"Encountered exception: {err}")
            return "unknown"

    detected_language = language_detection(client, content)


    voice_map = {
        'id': 'id-ID-GadisNeural',       # Bahasa Indonesia
        'en': 'en-US-GuyNeural',         # Bahasa Inggris (AS)
        'es': 'es-MX-JorgeNeural',       # Bahasa Spanyol (Meksiko)
        'fr': 'fr-FR-DeniseNeural',      # Bahasa Prancis (Prancis)
        'de': 'de-DE-ConradNeural',      # Bahasa Jerman (Jerman)
        'ar': 'ar-SA-HamedNeural',       # Bahasa Arab (Arab Saudi)
        'zh_chs': 'zh-CN-XiaoxiaoNeural', # Bahasa Mandarin (China)
        'ja': 'ja-JP-NanamiNeural',      # Bahasa Jepang
        'pt': 'pt-BR-FranciscaNeural',   # Bahasa Portugis (Brasil)
        'ru': 'ru-RU-SvetlanaNeural',    # Bahasa Rusia
        'nl': 'nl-NL-ColetteNeural',     # Bahasa Belanda
        'ko': 'ko-KR-SunHiNeural',       # Bahasa Korea
        'tr': 'tr-TR-EmelNeural',        # Bahasa Turki
        'sv': 'sv-SE-HilleviNeural',     # Bahasa Swedia
        'pl': 'pl-PL-ZofiaNeural',       # Bahasa Polandia
        'cs': 'cs-CZ-VlastaNeural',      # Bahasa Ceska (Ceko)
        'hu': 'hu-HU-NoemiNeural',       # Bahasa Hungaria
        'ro': 'ro-RO-EmilNeural',        # Bahasa Rumania
        'sk': 'sk-SK-LukasNeural',       # Bahasa Slovak
        'bg': 'bg-BG-BorislavNeural',    # Bahasa Bulgaria
        'hr': 'hr-HR-GabrijelaNeural',   # Bahasa Kroasia
        'lv': 'lv-LV-NilsNeural',        # Bahasa Latvia
        'it': 'it-IT-ElsaNeural',        # Bahasa Italia
        'vi': 'vi-VN-HoaiMyNeural'       # Bahasa Vietnam
    }

    voice = voice_map.get(detected_language, 'id-ID-GadisNeural')
    speech_config.speech_synthesis_voice_name = voice
    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_synthesizer.speak_text_async(content).get()

    return audio_path


def stt(audio_path):
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    source_language_config = speechsdk.languageconfig.SourceLanguageConfig("id-ID")  
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    speech_recognizer = speechsdk.SpeechRecognizer(  
        speech_config=speech_config, 
        source_language_config=source_language_config, 
        audio_config=audio_config
    )

    result = speech_recognizer.recognize_once()
    return result.text


def grade_choices(question, answer):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "berikan_penilaian",
                "description": "Berika penilaian",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "benar_salah": {
                            "type": "boolean",
                            "description": "Apakah jawaban pengguna benar atau salah",
                        },
                        "penjelasan": {
                            "type": "string",
                            "description": "Jelaskan jawaban yang benar",
                        },
                    },
                    "required": ["benar_salah", "penjelasan"],
                },
            },
        }
    ]

    client = AzureOpenAI(api_version="2024-02-01")
    completion = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah seorang guru yang saat ini sedang mengadakan ujian. Soal berupa pilihan ganda. Berikut adalah soalnya:"
            },
            {
                "role": "system",
                "content": question
            },
            {
                "role": "user",
                "content": answer
            },
            {
                "role": "assistant",
                "content": "Berdasarkan jawaban pengguna di atas ini, berikan penilaian dalam format berikut: Jawaban: BENAR atau SALAH Penjelasan: Jelaskan jawaban yang benar"
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "berikan_penilaian"}}
    )

    args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    is_correct = args['benar_salah']
    explanation = args['penjelasan']

    return is_correct, explanation


def grade_essay(question, answer):
    raise NotImplementedError

    tools = [
        {
            "type": "function",
            "function": {
                "name": "berikan_penilaian",
                "description": "Berika penilaian",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nilai": {
                            "type": "number",
                            "description": "Nilai dari 0-10",
                        },
                        "penjelasan": {
                            "type": "string",
                            "description": "Jelaskan jawaban yang benar",
                        },
                    },
                    "required": ["nilai", "penjelasan"],
                },
            },
        }
    ]

    client = AzureOpenAI(api_version="2024-02-01")
    completion = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah seorang guru yang saat ini sedang mengadakan ujian. Soal berupa esai. Berikut adalah soalnya:"
            },
            {
                "role": "system",
                "content": question
            },
            {
                "role": "user",
                "content": answer
            },
            {
                "role": "assistant",
                "content": "Berdasarkan jawaban pengguna di atas ini, berikan penilaian dalam format berikut: Nilai: 0-10 Penjelasan: Jelaskan jawaban yang benar"
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "berikan_penilaian"}}
    )

    args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    grade = args['nilai']
    explanation = args['penjelasan']

    return grade, explanation
