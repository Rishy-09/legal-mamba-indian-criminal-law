import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import time
import os

from mamba import Mamba, ModelArgs

# --- CONFIGURATION (Student SLM Tier) ---
BATCH_SIZE = 8       # Reduced to 8 to fit the larger model in GPU memory
BLOCK_SIZE = 256     # Increased context window for better sentence structure
LR = 3e-4
MAX_STEPS = 300     # You might want to increase this to 10,000 later
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# for visualization
loss_history = []

# Paths
# Change these paths in your notebook configuration cell
TXT_PATH = "data/final_train_v3.txt"
TOKENIZER_PATH = "data/tokenizer.json"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- DATASET ---
class TextDataset(Dataset):
    def __init__(self, txt_path, tokenizer_path, block_size):
        print(f"-> Loading {txt_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        print("-> Tokenizing (this might take a moment)...")
        # Ensure we just take the ids
        ids = self.tokenizer.encode(text).ids
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        print(f"-> Loaded {len(self.data)} tokens.")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Random sampling
        rand_start = torch.randint(0, len(self.data) - self.block_size, (1,)).item()
        chunk = self.data[rand_start : rand_start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def train():
    print(f"-> Using device: {DEVICE}")
    
    # 1. Prepare Data
    dataset = TextDataset(TXT_PATH, TOKENIZER_PATH, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 2. Setup Model
    vocab_size = dataset.tokenizer.get_vocab_size()
    
    # Use the Magnified Config
    model = Mamba(
        ModelArgs(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=24
        )
    ).to(DEVICE)
    
    # Print Parameter Count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Model Size: {num_params / 1_000_000:.2f} Million Parameters")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # ENHANCEMENT: Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=1e-5)

    # 3. Training Loop
    model.train()
    start_time = time.time()
    iter_loader = iter(dataloader)
    
    for step in range(MAX_STEPS):
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            x, y = next(iter_loader)
            
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x)
        
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.reshape(B*T, C), y.reshape(B*T))

        loss_history.append(loss.item())
        
        loss.backward()
        
        # ENHANCEMENT: Gradient Clipping (Prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step() # Update LR
        
        # Logging
        if step % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            dt = time.time() - start_time
            print(f"Step {step}/{MAX_STEPS} | Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Time: {dt:.2f}s")

        # ENHANCEMENT: Save Checkpoints
        if step % 50 == 0 and step > 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"model_step_{step}.pth")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss_history": loss_history,
            }, save_path)
            print(f"-> Checkpoint saved: {save_path}")

    # 4. Final Save
    print("-> Saving final model...")
    torch.save({
        "step": MAX_STEPS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_history": loss_history,
    }, "model_final.pth")
    print("-> Done!")

if __name__ == "__main__":
    train()