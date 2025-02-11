import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import pyaudio
import wave
import os
from dotenv import load_dotenv
from openai import OpenAI
import speech_recognition as sr
from jiwer import wer
from rouge_score import rouge_scorer
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Lower sample rate for faster processing
CHUNK = 1024
OUTPUT_FILENAME = "continuous_recording.wav"
TRANSCRIPTION_FILE = "transcription_summary.txt"
SUMMARY_FILE = "summary.txt"
stop_recording = False
executor = ThreadPoolExecutor()

# Function to record audio
def record_audio(app_instance):
    global stop_recording
    stop_recording = False
    app_instance.update_status("Recording started...")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    with wave.open(OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        while not stop_recording:
            data = stream.read(CHUNK)
            wf.writeframes(data)  

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    app_instance.update_status("Recording ended.")
    asyncio.run(process_audio(app_instance))


# Main processing function
async def process_audio(app_instance):
    transcribed_text = await transcribe_audio_async()
    if transcribed_text:
        summary_text = await generate_summary_async()
        accuracy = calculate_transcription_accuracy(transcribed_text)
        reference_file = "lung.txt"
        try:
            with open(reference_file, "r", encoding="utf-8") as f:
                reference_text = f.read().strip()
        except Exception as e:
            reference_text = "Error reading reference file"
        
        rouge_scores = evaluate_summary(reference_text, summary_text)

        app_instance.update_summary(summary_text)
        app_instance.update_accuracy(accuracy)
        app_instance.update_rouge_score(rouge_scores)

# Asynchronous transcription function
async def transcribe_audio_async():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, transcribe_audio, OUTPUT_FILENAME)

# Asynchronous summary generation
async def generate_summary_async():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, generate_summary)


# Function to transcribe audio using OpenAI Whisper
import re

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        
        transcribed_text = transcription.text
        transcribed_text = re.sub(r"[^a-zA-Z0-9\s]", "", transcribed_text)

        with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
            f.write(transcribed_text + "\n\n")
        
        return transcribed_text
    except Exception as e:
        return f"Error during transcription: {e}"

# Function to generate summary using GPT-4
def generate_summary():
    try:
        with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
            transcribed_text = f.read().strip()

        if transcribed_text:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI medical assistant"},
                    {"role": "user", "content": f"Suggest what disease the patient exhibits based on the transcription(output format: The patient exhibits symptoms of(disease related to transcrption)):\n {transcribed_text}"}
                ],
                max_tokens=100  # Limit response length
            )

            summary_text = response.choices[0].message.content.strip()
            with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                f.write(summary_text)
            return summary_text
        return "No transcription available."
    except Exception as e:
        return f"Error generating summary: {e}"

# Function to calculate transcription accuracy using WER
def calculate_transcription_accuracy(system_transcript):
    try:
        reference_file="lung_sym.txt"
        with open(reference_file, "r", encoding="utf-8") as f:
            reference_transcript = f.read().strip()
            reference_transcript=re.sub(r"[^a-zA-Z0-9\s]","",reference_transcript)

        if not system_transcript.strip():
            return "Transcription Accuracy: No valid transcription"

        
        reference_transcript = reference_transcript.lower()
        system_transcript = system_transcript.lower()

        # Print for debugging
        print("Reference:", reference_transcript)
        print("Transcribed:", system_transcript)

        error_rate = wer(reference_transcript, system_transcript)
        accuracy = max(0, 1 - error_rate) * 100
        return f"Transcription Accuracy: {accuracy:.2f}%"
    except Exception as e:
        return f"Error calculating accuracy: {e}"

# Function to evaluate summary quality using ROUGE score
def evaluate_summary(reference_summary, generated_summary):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)

        formatted_scores = (
            f"ROUGE Scores\n"
            f"rouge1 - Precision: {scores['rouge1'].precision * 100:.2f}%, Recall: {scores['rouge1'].recall * 100:.2f}%, F1: {scores['rouge1'].fmeasure * 100:.2f}%\n\n"
            f"rouge2 - Precision: {scores['rouge2'].precision * 100:.2f}%, Recall: {scores['rouge2'].recall * 100:.2f}%, F1: {scores['rouge2'].fmeasure * 100:.2f}%\n\n"
            f"rougeL - Precision: {scores['rougeL'].precision * 100:.2f}%, Recall: {scores['rougeL'].recall * 100:.2f}%, F1: {scores['rougeL'].fmeasure * 100:.2f}%"
        )
        return formatted_scores
    except Exception as e:
        return f"Error calculating ROUGE score: {e}"

class AmbientListening(MDApp):
    def build(self):
        self.layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)
        self.status_label = MDLabel(text="Press start to record.", halign='center')
        self.start_btn = MDRaisedButton(text="Start Recording", on_release=self.start_recording, pos_hint={"center_x": 0.5})
        self.stop_btn = MDRaisedButton(text="Stop Recording", on_release=self.stop_recording, pos_hint={"center_x": 0.5})
        self.summary_label = MDLabel(text="Summary will appear here.", halign='center')
        self.accuracy_label = MDLabel(text="Transcription Accuracy: N/A", halign='center')
        self.rouge_label = MDLabel(text="ROUGE Scores: N/A", halign='center')
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.start_btn)
        self.layout.add_widget(self.stop_btn)
        self.layout.add_widget(self.summary_label)
        self.layout.add_widget(self.accuracy_label)
        self.layout.add_widget(self.rouge_label)
        return self.layout
    def start_recording(self, instance):
        threading.Thread(target=record_audio, args=(self,), daemon=True).start()

    def stop_recording(self, instance):
        global stop_recording
        stop_recording = True

    def update_status(self, status_text):
        self.status_label.text = status_text  

    def update_summary(self, summary_text):
        self.summary_label.text = summary_text

    def update_accuracy(self, accuracy_text):
        self.accuracy_label.text = accuracy_text

    def update_rouge_score(self, rouge_text):
        self.rouge_label.text = rouge_text

if __name__ == "__main__":
    AmbientListening().run()
