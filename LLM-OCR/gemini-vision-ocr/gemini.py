import os
import json
from PIL import Image
from pdf2image import convert_from_path
from google import genai
from google.genai import types

OUTPUT_FOLDER = "output_texts"
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
CHECKPOINT_FILE = os.path.join(OUTPUT_FOLDER, "progress.json")

def save_checkpoint(checkpoint_data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=4)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def preprocess_image(image, image_filename):
    image.save(image_filename, format="PNG")
    return image_filename

def image_to_text_gemini(image_filename):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        uploaded_file = client.files.upload(file=image_filename)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text="""
                        Extract all readable text, tranlate to spanish from this image and structure it in a JSON format with the following schema:
                        {
                            "texto": "translated body text"
                        }
                    """),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_modalities=["text"],
            response_mime_type="application/json",
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=generate_content_config,
        )

        return json.loads(response.text) if response.text else None
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None

def read_json_safe(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read JSON file {filepath}: {e}")
        return None

def process_pdf(pdf_path):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    print(f"Processing PDF: {pdf_path}")

    checkpoint = load_checkpoint()
    processed_pages = checkpoint.get("processed_pages", {})

    images = convert_from_path(pdf_path)
    all_text = []

    for i, image in enumerate(images):
        image_filename = os.path.join(IMAGE_FOLDER, f"page_{i + 1}.png")
        text_filename = os.path.join(OUTPUT_FOLDER, f"page_{i + 1}.json")

        if str(i) in processed_pages:
            print(f"Skipping already processed image {i + 1}")
            page_data = read_json_safe(text_filename)
            if page_data:
                all_text.append(page_data)
            continue

        print(f"Processing image {i + 1}...")
        preprocess_image(image, image_filename)
        extracted_text = image_to_text_gemini(image_filename)

        if extracted_text:
            with open(text_filename, "w", encoding="utf-8") as f:
                json.dump(extracted_text, f, indent=4)
            all_text.append(extracted_text)
            processed_pages[str(i)] = text_filename
            save_checkpoint({"processed_pages": processed_pages})
        else:
            print(f"Warning: No text extracted from image {i + 1}")

    final_text_file = os.path.join(OUTPUT_FOLDER, "final_output.json")
    with open(final_text_file, "w", encoding="utf-8") as f:
        json.dump(all_text, f, indent=4)

    print(f"Processing complete. Output saved to {final_text_file}")

if __name__ == "__main__":
    pdf_file = "book.pdf"
    process_pdf(pdf_file)