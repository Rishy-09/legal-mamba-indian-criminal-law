import fitz  # PyMuPDF
import re
import os
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data"
PDF_FILES = {
    "bns": "bns.pdf",
    "bnss": "bnss.pdf",
    "bsa": "bsa.pdf"
}
OUTPUT_FILE = os.path.join(DATA_DIR, "final_train.txt")

# --- 1. THE PDF CLEANER (Generalized V3) ---
def clean_pdf(pdf_path, law_name):
    print(f"-> Processing {law_name.upper()} ({pdf_path})...")
    if not os.path.exists(pdf_path):
        print(f"   ❌ Missing file: {pdf_path}")
        return ""

    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        text = page.get_text()
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Standard Garbage Filters
            if re.match(r'^Page \d+', line) or re.match(r'^\d+$', line): continue
            if "GAZETTE" in line or "MINISTRY" in line: continue
            if re.search(r'\. \. \. \.', line): continue # TOC dots
            cleaned_lines.append(line)
        full_text.append("\n".join(cleaned_lines))

    raw_text = "\n".join(full_text)

    # Surgical Cut: Find "1. Short title" and cut everything before
    # Matches "1. Short title" loosely to catch minor formatting diffs
    match_start = re.search(r"1\.\s*Short title", raw_text, re.IGNORECASE)
    if match_start:
        raw_text = raw_text[match_start.start():]
    
    # Surgical Cut: Find End (Repeal or Objects)
    match_end = re.search(r"STATEMENT OF OBJECTS", raw_text)
    if match_end:
        raw_text = raw_text[:match_end.start()]

    # Cleanup Hyphens and Spacing
    raw_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', raw_text)
    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
    
    print(f"   ✅ Cleaned {law_name}: {len(raw_text)} chars")
    return f"--- START {law_name.upper()} ---\n{raw_text}\n--- END {law_name.upper()} ---\n\n"


def fetch_judgments(n=1000):
    print(f"-> Fetching {n} Legal Judgments from HuggingFace...")
    try:
        # NEW RELIABLE DATASET
        # This dataset is already chunked and clean for LLM training
        dataset = load_dataset("vihaannnn/Indian-Supreme-Court-Judgements-Chunked", split="train", streaming=True)
        
        judgments_text = ""
        count = 0
        
        for item in tqdm(dataset):
            if count >= n: break
            
            # The text is in the 'text' column
            text = item.get('text', '')
            
            if len(text) > 1000: 
                judgments_text += f"--- JUDGMENT {count+1} ---\n{text}\n\n"
                count += 1
                
        print(f"   ✅ Fetched {count} judgments.")
        return judgments_text
    except Exception as e:
        print(f"   ⚠️ HuggingFace Error: {e}")
        return ""

# --- 3. MASTER EXECUTION ---
def main():
    final_corpus = ""
    
    # Step A: Clean PDFs
    for name, filename in PDF_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        final_corpus += clean_pdf(path, name)

    # Step B: Add Judgments
    final_corpus += fetch_judgments(n=500)

    # Step C: Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_corpus)
        
    print(f"\n✅ MASTER CORPUS READY: {OUTPUT_FILE}")
    print(f"Total Size: {len(final_corpus)/1024/1024:.2f} MB")
    print("Next Step: Retrain tokenizer on 'final_train.txt'")

if __name__ == "__main__":
    main()