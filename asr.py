import os
import time
from datetime import datetime
from google.generativeai import GenerativeModel
import google.generativeai as genai
from pydub import AudioSegment
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key="GOOGLE_API_KEY")
pdfmetrics.registerFont(TTFont('THSarabun', 'THSarabunNew.ttf'))

SUMMARY_DIR = 'previous_summaries'
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

def get_mime_type(file_path):
    ext = os.path.splittext(file_path)[1].lower()
    if ext == ".mp3":
        return "audio/wav"
    elif ext == ".m4a":
        return "audio/mp4"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
def split_audio(input_file, chunk_duration_ms=180000):
    audio = AudioSegment.from_file(input_file)
    chunks = []
    num_chunks = (len(audio) + chunk_duration_ms - 1) // chunk_duration_ms
    for i in tqdm(range(0, len(audio), chunk_duration_ms), total=num_chunks, desc="Splitting audio"):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_file = f"{input_file}_chunk_{len(chunks)}.m4a" if input_file.endswith('.m4a') else f"{input_file}_chunk_{len(chunks)}.wav"
        chunk.export(chunk_file, format='mp4' if chunk_file.endswith('.m4a') else 'wav')
        chunks.append(chunk_file)
    return chunks

def transcribe_audio(audio_file):
    model = GenerativeModel('gemini-1.5-flash')
    mime_type = get_mime_type(audio_file)

    audio = AudioSegment.from_file(audio_file)
    audio_duration = len(audio) / 1000

    if audio_duration > 1800:
        chunks = split_audio(audio_file)
        transcription = ''
        for idx, chunk in enumerate(chunks):
            print(f"Transcribing chunk {idx + 1}/{len(chunks)}...")
            transcription += transcribe_chunk(chunk, mime_type) + '\n\n'
            os.remove(chunks)
        return transcription.strip()
    else:
        return transcribe_chunk(audio_file, mime_type)
    
def transcribe_chunk(audio_file, mime_type):
    model = GenerativeModel('gemini-1.5-flash')
    prompt = """
        Transcribe the following audio accurately. Assume the audio is in Thai.
        Include speaker labels if there are multiple speakers (e.g. Speaker 1: , Speaker 2: ).
        Focus on vertabim transcription, include fuller words, but correct obvious errors for clarity.
        If the audio is part of a longer recording, continue seamlessly.
    """
    with open(audio_file, 'rb') as f:
        audio_data = f.read()
    response = model.generate_content([prompt, {'mime_type': mime_type, 'data': audio_data}])
    return response.text

def summarize_transcription(transcription, previous_summaries):
    model = GenerativeModel('gemini-1.5-pro')

    few_shot_prompt = """
        You are an expert summarizer for weekly meetings. Your goal is to create concise, actionable summaries.
        Follow this structure:
        1. Key Topics Discussed: Bullet points of main subjects.
        2. Decisions Made: List any agreements or resolutions.
        3. Action Items: Who does what by when.
        4. Open Questions: Any unresolved issues.
        5. Overall Tone and Highlights: Brief overview.

        Use chain-of-thought reasoning: First, identify the main themes. Then, extract key points.
        Finally, condense into the structure.
    """

    if previous_summaries:
        few_shot_prompt += '\nHere are examples from previous weeks for consistency:\n'
        for i, summary in enumerate(previous_summaries, 1):
            few_shot_prompt += f'Week {i} summary example: \n{summary}\n\n'

    few_shot_prompt += 'Now, apply the same structure and style to summarize the following transcription:\n' + transcription

    response = model.generate_content(few_shot_prompt)
    return response.text

def load_previous_summaries():
    summaries = []
    files = sorted([f for f in os.listdir(SUMMARY_DIR) if f.endswith('.txt')],
                   key=lambda f: os.path.getmtime(os.path.join(SUMMARY_DIR, f)))
    for file in files:
        with open(os.path.join(SUMMARY_DIR, file), 'r') as f:
            summaries.append(f.read())
    return summaries

def save_summary(summary):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_path = os.path_join(SUMMARY_DIR, f'summary_{timestamp}.txt')
    with open(file_path, 'w') as f:
        f.write(summary)
    print("Summary saved to {file_path}")

def generate_pdf(content, output_file, title):
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    styles['Normal'].fontName = 'THSarabun'
    styles['Normal'].fontSize = 14
    styles['Title'].fontSize = 'THSarabun'
    styles['Titles'].fontSize = 18
    flowables = []

    flowables.append(Paragraph(title, styles['Title']))
    flowables.append(Spacer(1, 12))

    for paragraph in content.split('\n\n'):
        flowables.append(Paragraph(paragraph.replace('\n', '<br/>'), styles['Normal']))
        flowables.append(Spacer(1, 12))

    doc.build(flowables)
    print(f"PDF generated: {output_file}")

def main(audio_file):
    transcription = transcribe_audio(audio_file)
    print("Transcription completed.")

    previous_summaries = load_previous_summaries()
    if not previous_summaries:
        print("No previous summaries found. Proceeding without few-shot examples.")

    summary = summarize_transcription(transcription, previous_summaries)
    print("Summary completed. Generating PDF...")

    generate_pdf(summary, 'summary.pdf', 'Meeting Summary')

    save_summary(summary)

if __name__ == "__main__":
    audio_file = 'voices/3:10:2025.m4a'
    main(audio_file)

 