import pytesseract
from pdf2image import convert_from_path
import os

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Comment out on Linux/Mac

def extract_text_from_pdf(pdf_path, output_txt_path="output_tesseract.txt"):
    images = convert_from_path(pdf_path, dpi=300)
    
    full_text = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng')
        full_text.append(f"----- Page {i+1} -----\n{text}\n")
    
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(full_text))
    
    print(f"Text extracted to {output_txt_path}")
    return full_text


pdf_path = './data/circular1.pdf' 
extracted_text = extract_text_from_pdf(pdf_path)

print(extracted_text[1])