from PIL import Image
import requests
import base64
import json
import io
import os
from pdf2image import convert_from_path

# Constants
OUTPUT_FOLDER = "output_texts"
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
CHECKPOINT_FILE = os.path.join(OUTPUT_FOLDER, "progress.json")
MODEL = "gemma3:12b" #'mistral-small3.1' #

def preprocess_image(image):
    """
    Converts an image to a standard RGB PNG format and encodes it to base64.
    """
    try:
        img = image.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def image_to_text_gemma(image, prompt="Extract all text from this image and translate it to spanish", server_url="http://localhost:11434/api/generate"):
    """
    Performs OCR using the Gemma model via Ollama's REST API.
    """
    try:
        encoded_string = preprocess_image(image)
        if not encoded_string:
            print("Error encoding image.")
            return None

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "images": [encoded_string],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2}
        }

        response = requests.post(server_url, json=payload)
        response.raise_for_status()

        json_data = response.json()
        return json_data.get("response", "").strip()

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None

def load_checkpoint():
    """Loads the processing checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_data):
    """Saves progress in a checkpoint file to resume later."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=4)

def process_pdf(pdf_path):
    """
    Converts a PDF into images, processes each image with OCR,
    saves text outputs per page, and concatenates into a final text file.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    print(f"Processing PDF: {pdf_path}")

    # Load checkpoint progress
    checkpoint = load_checkpoint()
    processed_pages = checkpoint.get("processed_pages", {})

    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    all_text = []

    for i, image in enumerate(images):
        page_num = i + 1
        image_filename = os.path.join(IMAGE_FOLDER, f"page_{page_num}.png")
        text_filename = os.path.join(OUTPUT_FOLDER, f"page_{page_num}.json")

        # Skip processing if already completed
        if str(page_num) in processed_pages:
            print(f"Skipping already processed page {page_num}")
            with open(text_filename, "r", encoding="utf-8") as f:
                all_text.append(f.read())
            continue

        print(f"Processing page {page_num}...")

        # Save image to file
        image.save(image_filename, format="PNG")

        # Extract text from image
        extracted_text = image_to_text_gemma(image)

        if extracted_text:
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            all_text.append(extracted_text)

            # Update checkpoint
            processed_pages[str(page_num)] = text_filename
            save_checkpoint({"processed_pages": processed_pages})

        else:
            print(f"Warning: No text extracted from page {page_num}")

    # Save concatenated text into a final file
    final_text_file = os.path.join(OUTPUT_FOLDER, "final_output.json")
    with open(final_text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    print(f"Processing complete. Output saved to {final_text_file}")

if __name__ == "__main__":
    pdf_file = "book.pdf"
    process_pdf(pdf_file)
