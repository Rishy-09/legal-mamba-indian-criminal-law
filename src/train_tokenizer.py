from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
import os

# CONFIGURATION
INPUT_TEXT_FILE = "data/final_train_v3.txt"
INPUT_JSON_FILE = "data/bns_instruction_dataset.jsonl"
OUTPUT_TOKENIZER = "data/tokenizer.json"

def train_custom_tokenizer():
    print("-> Preparing data for tokenizer...")
    
    # 1. GATHER ALL TEXT
    all_text = ""
    
    # Load The Master Corpus (BNS + BNSS + BSA + Judgments)
    if os.path.exists(INPUT_TEXT_FILE):
        print(f"   Loading {INPUT_TEXT_FILE}...")
        with open(INPUT_TEXT_FILE, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
    else:
        print(f"❌ Error: {INPUT_TEXT_FILE} not found! Check your folder.")
        return
        
    # Load Instruction Data (If you have it)
    if os.path.exists(INPUT_JSON_FILE):
        print(f"   Loading {INPUT_JSON_FILE}...")
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_text += data['instruction'] + "\n" + data['response'] + "\n"
                except:
                    pass
    else:
        print("   (Skipping JSONL - file not found, using text only)")

    # Save temporary file for the trainer
    temp_file = "data/tokenizer_dump.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(all_text)
        
    print(f"-> Corpus Stats: {len(all_text)} characters collected.")

    # 2. CONFIGURE TOKENIZER (Byte Pair Encoding)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # 3. TRAIN IT
    # vocab_size=8192: The "Goldilocks" size for domain-specific SLMs
    trainer = BpeTrainer(
        vocab_size=8192, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    print("-> Training Tokenizer... (This uses Rust, it's fast)")
    tokenizer.train(files=[temp_file], trainer=trainer)

    # 4. SAVE IT
    tokenizer.save(OUTPUT_TOKENIZER)
    print(f"✅ Success! Custom Tokenizer saved to: {OUTPUT_TOKENIZER}")
    
    # 5. TEST IT
    # We test with a word that appears in Judgments but NOT in BNS
    test_sentence = "The Appellant filed a petition under the Evidence Act."
    encoded = tokenizer.encode(test_sentence)
    print("\n--- TEST ---")
    print(f"Input: {test_sentence}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs:    {encoded.ids}")

    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    train_custom_tokenizer()