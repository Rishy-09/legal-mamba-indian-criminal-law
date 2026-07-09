import re
import os

# CONFIGURATION
INPUT_FILE = "data/final_train.txt"
OUTPUT_FILE = "data/final_train_normalized.txt"

def clean_corpus():
    print(f"-> Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: File not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    original_len = len(text)
    print(f"   Original Size: {original_len} chars")

    # --- PHASE 1: REMOVE GAZETTE & PDF ARTIFACTS ---
    print("-> Scrubbing Gazette metadata...")
    
    # 1. Remove ID Codes & Registration Numbers
    text = re.sub(r'xxxGID[A-Z]+xxx', '', text)  # xxxGIDHxxx
    text = re.sub(r'CG-DL-E-[\d-]+', '', text)   # CG-DL-E-25122023
    text = re.sub(r'REGISTERED NO\..*', '', text, flags=re.IGNORECASE)
    
    # 2. Remove Common Gazette Headers
    garbage_phrases = [
        r"EXTRAORDINARY",
        r"PUBLISHED BY AUTHORITY",
        r"PART II — Section 1",
        r"LEGISLATIVE DEPARTMENT",
        r"MINISTRY OF LAW",
        r"NEW DELHI, .*", 
        r"SEPARATE PAGING IS GIVEN.*",
    ]
    for phrase in garbage_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)

    # 3. Remove Visual Separators (Long underscores)
    # Matches any sequence of 3 or more underscores
    text = re.sub(r'_{3,}', '', text)

    # 4. Remove Recurring Page Headers found in your file
    text = re.sub(r'\[Part II—', '', text)
    text = re.sub(r'Sec\. 1\]', '', text)

    # --- PHASE 2: REMOVE ENCODING/HINDI ARTIFACTS ---
    print("-> Removing encoding artifacts...")
    # Specific patterns found in your file analysis
    artifacts = [
        r"vlk/kkj\.k",  # Garbled Hindi
        r"Hkkx II",     # Part II in Hindi
        r"lañ \d+\]",   # Numbering artifact
        r"izkf/kdkj ls",
        r"izdkf'kr",
        r"¼'kd½",       # Saka era marker
        r"\[k\.M",
    ]
    for art in artifacts:
        text = re.sub(art, '', text)

    # --- PHASE 3: LEGAL NORMALIZATION ---
    print("-> Normalizing Legal Terms...")
    
    # Expand "u/s" -> "under Section"
    text = re.sub(r'\b[uU]/[sS]\.?\s+', 'under Section ', text)
    
    # Standardize "Sec." -> "Section"
    text = re.sub(r'\b[sS]ec\.\s+', 'Section ', text)
    
    # Fix Act Acronyms
    text = text.replace("Cr.P.C.", "CrPC")
    text = text.replace("I.P.C.", "IPC")

    # --- PHASE 4: WHITESPACE CLEANUP ---
    # Collapse multiple newlines into two (to keep paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)

    new_len = len(text)
    print(f"-> Cleanup Complete. Removed {original_len - new_len} chars of noise.")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved clean corpus to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_corpus()