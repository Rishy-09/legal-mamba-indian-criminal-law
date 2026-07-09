import google.generativeai as genai
import time
import json
import os
import re  # Added Regex for cleaning
from tqdm import tqdm

# PASTE YOUR KEY HERE

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
input_file = "data/bns_clean_v3.txt"
output_file = "data/bns_instruction_dataset.jsonl"

# Configure Gemini with the NEW model you found
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

system_instruction = """
You are an expert in Indian Law (BNS 2023).
Generate 3 high-quality "Instruction-Response" pairs based EXCLUSIVELY on the provided text.
Format as a JSON LIST: [{"instruction": "...", "response": "..."}]
Rules:
1. No Markdown formatting.
2. No intro/outro text.
3. Use valid JSON (escape quotes properly).
"""

def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def clean_json_string(text):
    """The Janitor: Extracts the JSON list from messy text."""
    try:
        # 1. Try to find the JSON block inside markdown ```json ... ```
        match = re.search(r"```json\s*(.*)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        
        # 2. Find the first '[' and last ']' to strip surrounding text
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx : end_idx + 1]
        
        return text
    except:
        return text

def generate_synthetic_data():
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    chunks = chunk_text(raw_text)
    print(f"-> Text split into {len(chunks)} chunks. Starting generation...")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        # Added tqdm for progress bar
        for i, chunk in enumerate(tqdm(chunks)):
            try:
                # Construct Prompt
                prompt = f"{system_instruction}\n\n--- LEGAL TEXT ---\n{chunk}\n\n--- JSON OUTPUT ---"
                
                # Call API
                response = model.generate_content(prompt)
                
                # JANITOR STEP: Clean the response
                raw_response = response.text
                clean_response = clean_json_string(raw_response)
                
                # Parse JSON
                data_batch = json.loads(clean_response)
                
                # Save
                for entry in data_batch:
                    json.dump(entry, f_out)
                    f_out.write('\n')
                
                # Sleep to match Free Tier limits (15 requests/min = 4s sleep)
                time.sleep(4) 

            except json.JSONDecodeError:
                # If JSON is still bad, just skip this chunk and DON'T CRASH
                # print(f" [Skipping Bad JSON in Chunk {i}] ", end="") 
                pass 
            except Exception as e:
                # Print other errors but keep going
                print(f"\n[!] Error on chunk {i}: {e}")
                time.sleep(5)

    print(f"\n✅ Dataset Generation Complete! Saved to {output_file}")

if __name__ == "__main__":
    if "PASTE_YOUR" in API_KEY:
        print("❌ Please paste your API Key inside the script first!")
    else:
        generate_synthetic_data()