import pytesseract
from pdf2image import convert_from_path
import os
import glob
from PyPDF2 import PdfReader
import re
import json
from datetime import datetime

def extract_pdf_metadata(pdf_path):
    """Extract metadata from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            info = reader.metadata
            
            # Basic metadata
            metadata = {
                "title": info.get('/Title', ''),
                "author": info.get('/Author', ''),
                "subject": info.get('/Subject', ''),
                "creator": info.get('/Creator', ''),
                "producer": info.get('/Producer', ''),
                "creation_date": info.get('/CreationDate', ''),
                "modification_date": info.get('/ModDate', ''),
                "pages": len(reader.pages),
                "filename": os.path.basename(pdf_path),
                "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 2)
            }
            
            # Format dates if present
            for date_field in ['creation_date', 'modification_date']:
                if metadata[date_field] and metadata[date_field].startswith('D:'):
                    # Parse PDF date format (D:YYYYMMDDHHmmSSOHH'mm')
                    date_str = metadata[date_field][2:14]  # Extract YYYYMMDDHHmm
                    try:
                        parsed_date = datetime.strptime(date_str, '%Y%m%d%H%M')
                        metadata[date_field] = parsed_date.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass  # Keep original if parsing fails
            
            # Extract document-specific fields (custom function)
            custom_metadata = extract_custom_fields(reader)
            metadata.update(custom_metadata)
            
            return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {}

def extract_custom_fields(pdf_reader):
    """Extract document-specific metadata from PDF text content"""
    custom_fields = {}
    
    # Extract text from first few pages
    text = ""
    for i in range(min(3, len(pdf_reader.pages))):  # First 3 pages
        text += pdf_reader.pages[i].extract_text()
    
    # Look for circular number
    circular_match = re.search(r'Circular\s+No\.?\s*([A-Za-z0-9\-\.\/]+)', text)
    if circular_match:
        custom_fields['circular_number'] = circular_match.group(1).strip()
    
    # Look for dates in the format DD Month YYYY
    date_patterns = re.finditer(r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', text)
    dates = [match.group(0) for match in date_patterns]
    if dates:
        custom_fields['document_dates'] = dates
    
    # Look for references to specific acts or regulations
    acts = re.findall(r'(?:Act|Regulation),?\s+\d{4}', text)
    if acts:
        custom_fields['regulations'] = acts
    
    return custom_fields

def extract_text_with_metadata(pdf_path, output_txt_path=None, metadata_path=None):
    """Extract both text content and metadata from PDF"""
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output paths if not provided
    filename = os.path.basename(pdf_path)
    filename_no_ext = os.path.splitext(filename)[0]
    
    if output_txt_path is None:
        output_txt_path = os.path.join(output_dir, f"{filename_no_ext}.txt")
        
    if metadata_path is None:
        metadata_path = os.path.join(output_dir, f"{filename_no_ext}_metadata.json")
    
    # Get metadata
    metadata = extract_pdf_metadata(pdf_path)
    
    # Save metadata to JSON file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Extract text using OCR
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return [], metadata
    
    full_text = []
    # Add metadata as header
    metadata_header = f"--- Document Metadata: {filename} ---\n"
    for key, value in metadata.items():
        if isinstance(value, str) or isinstance(value, (int, float)):
            metadata_header += f"{key}: {value}\n"
        elif isinstance(value, list):
            metadata_header += f"{key}: {', '.join(str(v) for v in value)}\n"
    full_text.append(metadata_header)
    
    # Process each page
    for i, image in enumerate(images):
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            full_text.append(f"----- {filename}: Page {i+1} -----\n{text}\n")
        except Exception as e:
            print(f"Error extracting text from page {i+1}: {str(e)}")
            full_text.append(f"----- {filename}: Page {i+1} -----\nError extracting text\n")
    
    # Write to file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(full_text))
    
    print(f"Text and metadata extracted to {output_txt_path}")
    print(f"Metadata saved separately to {metadata_path}")
    
    return full_text, metadata

def process_all_pdfs(data_dir='./data'):
    """Process all PDFs in the data directory"""
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(data_dir, '*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return {}, ""
        
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    all_metadata = {}
    combined_text_path = os.path.join(output_dir, "combined_text.txt")
    
    with open(combined_text_path, 'w', encoding='utf-8') as combined_file:
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            print(f"Processing {filename}...")
            
            # Generate output paths
            output_txt_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            metadata_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
            
            # Extract text and metadata
            text, metadata = extract_text_with_metadata(pdf_path, output_txt_path, metadata_path)
            
            # Store metadata
            all_metadata[filename] = metadata
            
            # Add to combined text file with clear document separators
            combined_file.write(f"\n\n--- BEGIN DOCUMENT: {filename} ---\n\n")
            
            # Read the individual file to ensure we're getting what was written to disk
            with open(output_txt_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
                combined_file.write(content)
                
            combined_file.write(f"\n\n--- END DOCUMENT: {filename} ---\n\n")
    
    # Save all metadata to a combined file
    all_metadata_path = os.path.join(output_dir, "all_metadata.json")
    with open(all_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
        
    print(f"All texts combined into {combined_text_path}")
    print(f"All metadata combined into {all_metadata_path}")
    
    return all_metadata, combined_text_path

# Example usage
if __name__ == "__main__":
    all_metadata, combined_text_path = process_all_pdfs('./data')
    
    print("\nProcessed PDF files:")
    for filename, metadata in all_metadata.items():
        print(f"- {filename}")
        print(f"  Title: {metadata.get('title', 'N/A')}")
        print(f"  Author: {metadata.get('author', 'N/A')}")
        print(f"  Circular: {metadata.get('circular_number', 'N/A')}")