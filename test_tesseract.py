import pytesseract
from pdf2image import convert_from_path
import os
from dotenv import load_dotenv

load_dotenv('.env', override=True)

def extract_text_from_pdf(pdf_path, output_dir="extracted_texts"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output text file path based on PDF filename
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_txt_path = os.path.join(output_dir, f"{pdf_filename}_extracted.txt")
    
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=300)
    
    full_text = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng')
        full_text.append(f"----- Page {i+1} -----\n{text}\n")
    
    # Save extracted text to file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(full_text))
    
    print(f"Text extracted to {output_txt_path}")
    return full_text, output_txt_path

def process_all_pdfs(data_dir=os.getenv("DATA_DIR"), output_dir="extracted_texts"):
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    # Get all PDF files in the data directory
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    extracted_texts = {}
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"Processing {pdf_path}...")
        text, output_path = extract_text_from_pdf(pdf_path, output_dir)
        extracted_texts[pdf_file] = {
            "text": text,
            "output_path": output_path
        }
    
    return extracted_texts

if __name__ == "__main__":
    # Process all PDFs in the ./data directory
    extracted_texts = process_all_pdfs()
    for pdf_file, data in extracted_texts.items():
        print(f"\nPDF: {pdf_file}")
        print(f"Output Path: {data['output_path']}")
        print(f"First page text (preview): {data['text'][0][:200]}...")