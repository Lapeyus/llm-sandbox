import os
from swarm import Swarm, Agent
from dotenv import load_dotenv
import subprocess
from pydub import AudioSegment
import re
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Get necessary environment variables
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME_LARGE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Swarm client
client = Swarm()

# Initialize Agents
narrative_agent = Agent(
    name="Podcast Stylist",
    instructions=(
        "Using the provided text, create a podcast-style monologue presented by a single narrator. Since the listener cannot see the text, it’s essential to describe non-readable elements like diagrams and tables clearly and concisely. Ensure the explanation is thoughtful, engaging, and easy for the audience to follow, adopting a conversational style. Avoid adding any non-verbal cues or stage directions (e.g., ‘[Intro music fades out]’, ‘[Pause]’), as the Text-to-Speech agent will read them aloud if you do. Your response must include all entities and topics from the original text"
    ),
    model=OPENAI_MODEL_NAME
)

lda_agent = Agent(
    name="lda",
    instructions=(
        "extract a list of the main themes or themes within the text using the Latent Dirichlet Allocation (LDA) technique in this text. respond with a list of the topics obtained from the LDA results."
    ),
    model=OPENAI_MODEL_NAME
)

ent_agent = Agent(
    name="lda",
    instructions=(
        "extract a list of all entities present in the text"
    ),
    model=OPENAI_MODEL_NAME
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to process files in a folder
def process_folder(base_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r') as file:
                        content = file.read()

                    # ent_response = client.run(
                    #     agent=ent_agent,
                    #     messages=[{"role": "user", "content": content}],
                    # )
                    # lda_response = client.run(
                    #     agent=narrative_agent,
                    #     messages=[{"role": "user", "content": content}],
                    # )
                    # context_variables = {"entities": ent_response, "topics": lda_response}
                    # print(context_variables)

                    # Generate narrative response
                    narrative_response = client.run(
                        agent=narrative_agent,
                        messages=[{"role": "user", "content": content}],
                        # context_variables=context_variables,
                    )
                    narrative = narrative_response.messages[-1]["content"]

                    # Save the final content to a text file
                    output_file = os.path.join(output_folder, filename)
                    with open(output_file, 'w') as output:
                        output.write(narrative)
                    logging.info(f"Saved podzcast-style response to {output_file}")

                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")

# TTS Function to process text with voice cues
def process_tts_with_voices(text, output_file):
    """
    Converts text with voice change cues into audio using macOS TTS engine.
    """
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    voice_segments = re.split(r'(\[Voice: [^\]]+\])', text)
    combined_audio = AudioSegment.empty()

    for segment in voice_segments:
        # Check for voice cue
        voice_match = re.match(r'\[Voice: ([^\]]+)\]', segment)
        if voice_match:
            current_voice = voice_match.group(1).strip()
            continue  # Skip the voice cue itself

        # Skip empty segments
        if not segment.strip():
            continue

        # Create temporary text file for TTS processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_text_file:
            temp_text_file.write(segment.encode('utf-8'))
            temp_text_file_path = temp_text_file.name

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name

        subprocess.run(
            ['say', '-f', temp_text_file_path, '-o', temp_audio_file_path, '--data-format=LEF32@22050']
        )

        # Append the generated audio segment
        segment_audio = AudioSegment.from_wav(temp_audio_file_path)
        combined_audio += segment_audio

        # Clean up temporary files
        os.remove(temp_text_file_path)
        os.remove(temp_audio_file_path)

    # Export the combined audio to the final output file
    combined_audio.export(output_file, format="wav")
    logging.info(f"Generated audio saved to {output_file}")

# Main processing
if __name__ == "__main__":
    input_folder = "./txt"  # Folder with input text files
    output_text_folder = "./podcast/text_responses"  # Folder for text responses
    output_audio_folder = "./podcast/audio_files"  # Folder for audio files
    final_audio_output = "./podcast/final_podcast.wav"  # Combined audio file

    # Process text files into podcast narratives
    process_folder(input_folder, output_text_folder)

    # Generate audio for each text response
    combined_audio = AudioSegment.empty()
    with ThreadPoolExecutor() as executor:
        futures = []
        for text_file in os.listdir(output_text_folder):
            if text_file.endswith('.txt'):
                text_path = os.path.join(output_text_folder, text_file)
                output_audio_path = os.path.join(output_audio_folder, text_file.replace('.txt', '.wav'))

                with open(text_path, 'r') as file:
                    text_content = file.read()
                    futures.append(executor.submit(process_tts_with_voices, text_content, output_audio_path))

        for future in futures:
            future.result()  # Wait for all threads to complete

    # Append to combined audio
    for audio_file in os.listdir(output_audio_folder):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(output_audio_folder, audio_file)
            combined_audio += AudioSegment.from_wav(audio_path)

    # Export the combined podcast audio
    combined_audio.export(final_audio_output, format="wav")
    logging.info(f"Final podcast audio saved to {final_audio_output}")
