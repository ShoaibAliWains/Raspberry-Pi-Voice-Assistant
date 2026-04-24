# -*- coding: utf-8 -*-

import io
import json
import os
import re
import time
import wave
import audioop
import tempfile
import subprocess

from vosk import Model, KaldiRecognizer
from google import genai
from google.cloud import texttospeech


VOSK_MODEL_PATH = os.path.expanduser("~/vosk-model-small-cs-0.4-rhasspy")

USB_INPUT_DEVICE = "plughw:CARD=Device,DEV=0"
USB_OUTPUT_DEVICE = "plughw:CARD=Device,DEV=0"

INPUT_RATE = 16000
TARGET_RATE = 16000
CHANNELS = 1

WAKE_ALIASES = [
    "armor",
    "armore",
    "amor",
    "armour",
    "amor",
    "amore",
    "armr",
    "haló armor",
    "halo armor",
]

WAKE_LISTEN_SECONDS = 2.5
QUESTION_SECONDS = 5.0

GEMINI_MODEL = "gemini-2.5-flash"
TTS_VOICE_NAME = "cs-CZ-Chirp3-HD-Achird"

DEBUG = True
POST_TTS_COOLDOWN = 1.0


def debug_print(*args):
    if DEBUG:
        print(*args)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("[unk]", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáčďéěíňóřšťúůýž]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_setup():
    if not os.path.isdir(VOSK_MODEL_PATH):
        raise RuntimeError(f"VOSK model nebyl nalezen: {VOSK_MODEL_PATH}")

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Chybí GEMINI_API_KEY")

    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred or not os.path.exists(cred):
        raise RuntimeError("Chybí nebo neplatí GOOGLE_APPLICATION_CREDENTIALS")

    texttospeech.TextToSpeechClient()


def record_wav_with_arecord(seconds: float, device: str) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        temp_wav = f.name

    try:
        whole_seconds = max(1, int(round(seconds)))

        cmd = [
            "arecord",
            "-D", device,
            "-f", "S16_LE",
            "-r", str(INPUT_RATE),
            "-c", str(CHANNELS),
            "-d", str(whole_seconds),
            temp_wav
        ]

        debug_print("Nahrávám:", " ".join(cmd))
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        with open(temp_wav, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def wav_to_pcm_and_resample(wav_bytes: bytes, dst_rate: int) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    if channels == 2:
        pcm = audioop.tomono(pcm, sampwidth, 0.5, 0.5)
        channels = 1

    if src_rate != dst_rate:
        pcm, _ = audioop.ratecv(
            pcm,
            sampwidth,
            channels,
            src_rate,
            dst_rate,
            None
        )

    return pcm


def recognize_pcm_with_vosk(model: Model, pcm_bytes: bytes) -> str:
    rec = KaldiRecognizer(model, TARGET_RATE)
    rec.SetWords(False)

    chunk_size = 4000
    for i in range(0, len(pcm_bytes), chunk_size):
        rec.AcceptWaveform(pcm_bytes[i:i + chunk_size])

    result = json.loads(rec.FinalResult())
    return normalize_text(result.get("text", "").strip())


def contains_wake_word(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False

    words = text.split()

    for alias in WAKE_ALIASES:
        alias = normalize_text(alias)
        if alias in text:
            return True

    return False


def should_process_text(text: str) -> bool:
    text = normalize_text(text)

    if not text:
        return False
    if len(text) < 2:
        return False
    if text in ["a", "s", "z", "k", "v", "jo", "hm", "ehm", "ano", "ne"]:
        return False

    return True


def ask_gemini(question: str) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = (
        "Odpovídej pouze česky. "
        "Odpovídej stručně, přirozeně a užitečně. "
        "Nevypisuj dlouhé odstavce. "
        f"Dotaz uživatele: {question}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if hasattr(response, "text") and response.text:
        return response.text.strip()

    return "Promiň, teď se mi nepodařilo odpovědět."


def speak_text(text: str):
    client = texttospeech.TextToSpeechClient()

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="cs-CZ",
            name=TTS_VOICE_NAME
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(response.audio_content)
        temp_wav = f.name

    try:
        cmd = ["aplay", "-D", USB_OUTPUT_DEVICE, temp_wav]
        debug_print("Přehrávám:", " ".join(cmd))
        subprocess.run(cmd, check=False)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def wait_for_wake_word(model: Model):
    print("\nČekám na slovo 'armor'...")

    while True:
        wav_bytes = record_wav_with_arecord(WAKE_LISTEN_SECONDS, USB_INPUT_DEVICE)
        pcm = wav_to_pcm_and_resample(wav_bytes, TARGET_RATE)
        text = recognize_pcm_with_vosk(model, pcm)

        print("Wake slyším:", repr(text))

        if contains_wake_word(text):
            print("Wake slovo zachyceno.")
            return


def record_question_text(model: Model) -> str:
    print("Poslouchám dotaz...")
    wav_bytes = record_wav_with_arecord(QUESTION_SECONDS, USB_INPUT_DEVICE)
    pcm = wav_to_pcm_and_resample(wav_bytes, TARGET_RATE)
    text = recognize_pcm_with_vosk(model, pcm)

    print("Rozpoznaný dotaz:", repr(text))
    return text


def ask_once(model: Model):
    wait_for_wake_word(model)

    try:
        speak_text("Ano?")
    except Exception as e:
        print("Chyba TTS při potvrzení:", e)

    time.sleep(0.3)

    user_text = record_question_text(model)

    if should_process_text(user_text):
        print("\nUživatel:", user_text)
        try:
            answer = ask_gemini(user_text)
            print("Asistent:", answer)
            speak_text(answer)
        except Exception as e:
            print("Chyba při odpovědi:", e)
            try:
                speak_text("Promiň, došlo k chybě.")
            except Exception:
                pass
    else:
        print("Dotaz nebyl rozpoznán dostatečně dobře.")
        try:
            speak_text("Promiň, nerozuměl jsem.")
        except Exception:
            pass

    time.sleep(POST_TTS_COOLDOWN)


def main():
    check_setup()

    print("Načítám Vosk model...")
    model = Model(VOSK_MODEL_PATH)

    print("Asistent běží.")
    print("Musíš pokaždé znovu říct 'armor'.")

    while True:
        try:
            ask_once(model)
        except KeyboardInterrupt:
            print("\nUkončuji asistenta...")
            break
        except Exception as e:
            print("Hlavní chyba:", e)
            time.sleep(1)


if __name__ == "__main__":
    main()
