import re
import os

# CONFIGURATION
INPUT_FILE = "data/final_train_normalized.txt" # We clean the already normalized file
OUTPUT_FILE = "data/final_train_v3.txt"

def final_polish():
    print(f"-> Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: File not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    original_line_count = len(lines)
    cleaned_lines = []

    # --- THE KILL LIST (Specific patterns from your file) ---
    garbage_patterns = [
        r"vlk/kkj\.k",              # "Extraordinary" in garbled font
        r"ubZ fnYyh",               # "New Delhi" in garbled font
        r"lkseokj",                 # "Monday" in garbled font
        r"jftLVªh lañ",             # "Registered No." in Hindi
        r"bl Hkkx esa",             # "In this part" in Hindi
        r"Hkkx II",                 # "Part II"
        r"ikS\"k",                  # Hindu calendar month
        r"No\. 53\]",               # Specific Gazette number
        r"Mhñ ,yñ",                 # "DL" (Delhi) in Hindi
        r"— 1",                     # Artifact from "Part II - Section 1"
        r"GIDH",                    # GIDH code
    ]

    print("-> Scrubbing stubborn artifacts...")
    
    for line in lines:
        is_garbage = False
        
        # Check against patterns
        for pattern in garbage_patterns:
            if re.search(pattern, line):
                is_garbage = True
                break
        
        # Check for lines that are just symbols (e.g. "()")
        if line.strip() == "()":
            is_garbage = True

        if not is_garbage:
            cleaned_lines.append(line)

    # Rejoin text
    text = "".join(cleaned_lines)
    
    # Final cleanup of multiple newlines created by deleting lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    print(f"-> Cleanup Complete.")
    print(f"   Removed {original_line_count - len(cleaned_lines)} garbage lines.")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved Final Corpus to: {OUTPUT_FILE}")

if __name__ == "__main__":
    final_polish()