import os
import time
from datetime import datetime
from google import genai
from google.genai import types, errors  # type: ignore # Added 'errors' import
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

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

pdfmetrics.registerFont(TTFont('THSarabun', 'fonts/Sarabun-Regular.ttf'))

SUMMARY_DIR = 'previous_summaries'
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

def get_mime_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp3":
        return "audio/wav"
    elif ext == ".m4a":
        return "audio/mp4"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# FIX 1: Reduced chunk duration to 10 minutes (600,000ms) to avoid Token Limit errors
def split_audio(input_file, chunk_duration_ms=600000): 
    audio = AudioSegment.from_file(input_file)
    chunks = []
    num_chunks = (len(audio) + chunk_duration_ms - 1) // chunk_duration_ms
    for i in tqdm(range(0, len(audio), chunk_duration_ms), total=num_chunks, desc="Splitting audio"):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_file = f"{input_file}_chunk_{len(chunks)}.m4a" if input_file.endswith('.m4a') else f"{input_file}_chunk_{len(chunks)}.wav"
        chunk.export(chunk_file, format='mp4' if chunk_file.endswith('.m4a') else 'wav')
        chunks.append(chunk_file)
    return chunks

def transcribe_chunk(audio_file, mime_type, max_retries=5): # Increased retries
    model_name = 'gemini-2.5-flash' 
    
    prompt = """
        Transcribe the following audio accurately. Assume the audio is in Thai.
        Include speaker labels if there are multiple speakers (e.g. Speaker 1: , Speaker 2: ).
        Focus on verbatim transcription, include filler words, but correct obvious errors for clarity.
        If the audio is part of a longer recording, continue seamlessly.
    """
    
    with open(audio_file, 'rb') as f:
        audio_data = f.read()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=audio_data, mime_type=mime_type)
                ]
            )
            
            if response.text:
                return response.text
            else:
                print(f"Warning: No text returned for {audio_file}.")
                return ""

        # FIX 2: Catch specific Google API errors (429 Resource Exhausted)
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60  # Wait 1 full minute to reset quota
                print(f"⚠️ Quota exceeded (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"❌ API Error on {audio_file}: {e}")
                time.sleep(5)
                
        except Exception as e:
            print(f"Error transcribing {audio_file} on attempt {attempt + 1}: {e}")
            time.sleep(5)
            
    print(f"Failed to transcribe {audio_file} after {max_retries} attempts.")
    return ""

def transcribe_audio(audio_file):
    mime_type = get_mime_type(audio_file)

    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        return None

    audio_duration = len(audio) / 1000
    if audio_duration <= 0:
        print("Error: Audio file is empty or invalid.")
        return None

    # Logic: if > 10 mins (600s), split it.
    if audio_duration > 600:
        chunks = split_audio(audio_file)
        transcription = ''
        failed_chunks = []
        for idx, chunk in enumerate(chunks):
            print(f"Transcribing chunk {idx + 1}/{len(chunks)}...")
            
            chunk_transcription = transcribe_chunk(chunk, mime_type)
            
            if chunk_transcription:
                transcription += chunk_transcription + '\n\n'
            else:
                failed_chunks.append(idx + 1)
                print(f"Warning: No transcription for chunk {idx + 1}")
            
            # FIX 3: Clean up and SLEEP to prevent bursting limits
            try:
                os.remove(chunk)
                print("Sleeping 10s to cool down API...")
                time.sleep(10) 
            except OSError as e:
                print(f"Error deleting chunk {chunk}: {e}")
                
        if failed_chunks:
            print(f"Warning: Failed to transcribe chunks: {failed_chunks}")
        return transcription.strip() if transcription else None
    else:
        return transcribe_chunk(audio_file, mime_type)

def summarize_transcription(transcription, previous_summaries):
    if transcription is None or not transcription.strip():
        print("Error: Transcription is empty or None. Cannot summarize.")
        return None

    model_name = 'gemini-2.5-flash'

    # REPLACED: New complex prompt structure
    system_instruction = """
    You are an expert Chief of Staff and professional meeting minute taker. 
    Your goal is to produce a comprehensive, high-detail meeting report that captures not just what was said, but the nuance, context, and specific data points.
    
    The input text is a transcription of a meeting (likely in Thai). 
    Please analyze it deeply and generate a report in Thai (unless the transcription is 100% English) using the following strict structure:

    # 1. Executive Summary (บทสรุปผู้บริหาร)
    * **Objective:** A 2-3 sentence overview of why the meeting took place.
    * **Key Outcomes:** The top 3 most critical results or strategic shifts from this meeting.

    # 2. Detailed Discussion Log (บันทึกการหารือแบบละเอียด)
    *Analyze the transcript and group the conversation into distinct Topics. For EACH topic, provide:*
    * **Topic:** [Topic Name]
    * **Context:** Briefly explain the background or the problem being discussed.
    * **Key Discussion Points:** Detail the arguments raised. Who supported what? What were the concerns? (e.g., "Speaker A proposed X, but Speaker B argued it was too expensive").
    * **Specific Data:** EXTRACT all specific numbers, prices, dates, or KPIs mentioned.
    * **Conclusion:** How did this specific topic end? (Agreed, Deferred, or Disagreement).

    # 3. Decisions & Rationale (มติที่ประชุมและเหตุผล)
    * **Decision:** [What was decided]
    * **Rationale:** [Why was this decided? What data or argument supported this choice?]

    # 4. Action Items (งานที่ต้องดำเนินการต่อ)
    * [ ] **[Responsible Person]**: [Specific Task] (Deadline: [Date if mentioned]) - *Priority: [High/Medium/Low]*

    # 5. Parking Lot & Unresolved Issues (ประเด็นคงค้าง)
    * List items that were raised but not resolved or deferred to a future meeting.

    **CRITICAL INSTRUCTION:** - Do not be vague. Do not write "They discussed the budget." 
    - Instead write: "They discussed the Q3 marketing budget variance of 50,000 THB."
    - If speakers are identified in the transcript, attribute key quotes to them.
    """

    # We add the transcription to the user message
    user_content = "Here is the meeting transcription:\n\n" + transcription

    # Optional: logic to include previous summaries if you want continuity
    if previous_summaries:
        user_content = f"Note: This is a recurring meeting. Here are summaries from the last {len(previous_summaries)} meetings for context on ongoing issues:\n\n" + "\n".join(previous_summaries[-3:]) + "\n\n" + user_content

    try:
        response = client.models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction, # Using system instruction for the "Role"
                temperature=0.3 # Lower temperature for more factual/precise results
            ),
            contents=user_content
        )
        
        if response.text:
            return response.text
        else:
            print("Error: No valid summary returned from model.")
            return None
    except Exception as e:
        print(f"Error summarizing transcription: {e}")
        return None

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
    file_path = os.path.join(SUMMARY_DIR, f'summary_{timestamp}.txt')
    with open(file_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to {file_path}")

def generate_pdf(content, output_file, title):
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Register font (Keep your existing font registration)
    styles['Normal'].fontName = 'THSarabun'
    styles['Normal'].fontSize = 14
    styles['Normal'].leading = 18 # Add line spacing for readability
    
    styles['Title'].fontName = 'THSarabun'
    styles['Title'].fontSize = 20
    styles['Title'].alignment = 1 # Center align title

    # Create a custom style for headers (lines starting with #)
    from reportlab.lib.enums import TA_LEFT
    header_style = styles['Heading2']
    header_style.fontName = 'THSarabun'
    header_style.fontSize = 16
    header_style.textColor = 'black'

    flowables = []
    flowables.append(Paragraph(title, styles['Title']))
    flowables.append(Spacer(1, 20))

    # Split by lines to process formatting manually
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 6))
            continue

        # Handle Markdown-style Headers (#)
        if line.startswith('#'):
            clean_line = line.replace('#', '').strip()
            flowables.append(Spacer(1, 10))
            flowables.append(Paragraph(clean_line, header_style))
            flowables.append(Spacer(1, 5))
        
        # Handle simple bolding **text** -> <b>text</b> for ReportLab
        else:
            # ReportLab uses HTML-like tags for bolding. 
            # We replace ** with <b> and </b> alternately.
            # A simple regex or replace loop is needed for perfect markdown, 
            # but a quick fix is just replacing the ** with empty string if you don't want to parse logic,
            # OR simple manual replace:
            formatted_line = line.replace('**', '<b>', 1).replace('**', '</b>', 1) 
            # (Note: The line above only fixes one bold pair per line. For robust markdown parsing, you'd need a markdown library, 
            # but usually just rendering the text is fine).
            
            flowables.append(Paragraph(formatted_line, styles['Normal']))

    doc.build(flowables)
    print(f"PDF generated: {output_file}")

def main(audio_file):
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} does not exist.")
        return

    transcription = transcribe_audio(audio_file)
    if not transcription:
        print("Error: Transcription failed or is empty. Exiting.")
        return

    print("Transcription completed.")
    print("Transcription content:", transcription[:500] + "..." if len(transcription) > 500 else transcription)

    previous_summaries = load_previous_summaries()
    if not previous_summaries:
        print("No previous summaries found. Proceeding without few-shot examples.")

    summary = summarize_transcription(transcription, previous_summaries)
    if not summary:
        print("Error: Summary generation failed. Exiting.")
        return

    print("Summary completed. Generating PDF...")
    generate_pdf(summary, 'SomYing_Summary.pdf', 'Lecture Summary')
    save_summary(summary)

if __name__ == "__main__":
    audio_file = 'voices/มหาวิทยาลัยเกษตรศาสตร์.m4a'
    main(audio_file)