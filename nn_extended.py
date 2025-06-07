# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import torch.nn.functional as F
import torch.nn as nn

# %%
# Load the DataFrame
df = pd.read_csv("./data.csv") # Make sure your data.csv is in the correct path

# --- Data Preparation ---
# 1. Clean the 'toponym' column: convert to string, strip whitespace, convert to lowercase
df['toponym'] = df['toponym'].astype(str).str.strip().str.lower()

# 2. Convert each toponym string into a list of tokens, adding <S> and <E>
#    Each character will be a separate token
#    We create a new column 'toponym_tokens' to store these lists
df['toponym_tokens'] = df['toponym'].apply(lambda x: ['<S>'] + list(x) + ['<E>'])

# Get all unique tokens from your processed data
all_tokens = []
for tokens_list in df['toponym_tokens']:
    all_tokens.extend(tokens_list)

# Create a sorted list of unique tokens (your vocabulary)
vocab = sorted(list(set(all_tokens)))

stoi = {token: i for i, token in enumerate(vocab)}

# Create mapping from integer to token (itos) for debugging/display
itos = {i: token for token, i in stoi.items()}

# Print the vocabulary size
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
print(f"stoi mapping: {stoi}")

# %%
# Define your context window size
block_size = 3 # Example: uses 3 previous characters to predict the next

# Initialize lists to store our input sequences (X) and target characters (Y)
X, Y = [], []

# Iterate through each preprocessed toponym (list of tokens)
for tokens_list in df['toponym_tokens']:
    # Pad the beginning of the context with the start-of-sequence token ID
    # This ensures every sequence has a full 'block_size' context, even at the start.
    context = [stoi['<S>']] * block_size

    # Iterate through each token in the current toponym, including <E>
    # The current 'token_id' will be our target (Y) for the current 'context' (X)
    for token_str in tokens_list:
        # Convert the current token to its integer ID
        token_id = stoi[token_str]

        # Add the current context (as a list of integers) to X
        X.append(context)

        # Add the current token's ID (which is the next character) to Y
        Y.append(token_id)

        # Update the context by sliding the window:
        # Remove the oldest character from the left and add the new character (token_id) to the right
        context = context[1:] + [token_id]

# Convert the Python lists to PyTorch tensors
X = torch.tensor(X)
Y = torch.tensor(Y)

print(f"\nShape of X (inputs): {X.shape}")
print(f"Shape of Y (targets): {Y.shape}")
print(f"Data type of X: {X.dtype}")
print(f"Data type of Y: {Y.dtype}")

# Example of the first few X and Y pairs
print("\nFirst 10 X, Y pairs:")
for i in range(20):
    context_chars = ''.join(itos[idx.item()] for idx in X[i])
    target_char = itos[Y[i].item()]
    print(f"Context: {context_chars} ---> Target: {target_char}")

# %%
# Training split, dev/validation split, test split
# 80%, 10%, 10%

# Set a random seed for reproducibility in data splitting
random.seed(42) # Using random module for shuffling indices before splitting

# Create a permutation of indices
indices = list(range(X.shape[0]))
random.shuffle(indices)

n1 = int(0.8 * len(indices))
n2 = int(0.9 * len(indices))

Xtr, Ytr = X[indices[:n1]], Y[indices[:n1]]
Xdev, Ydev = X[indices[n1:n2]], Y[indices[n1:n2]]
Xte, Yte = X[indices[n2:]], Y[indices[n2:]]

print(f"\nXtr shape: {Xtr.shape}, Ytr shape: {Ytr.shape}")
print(f"Xdev shape: {Xdev.shape}, Ydev shape: {Ydev.shape}")
print(f"Xte shape: {Xte.shape}, Yte shape: {Yte.shape}")


# %%
class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_hidden=256, block_size=3):
        super().__init__()
        self.block_size = block_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Network architecture
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(n_embd * block_size, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            
            # Layer 2
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            
            # Layer 3
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            
            # Layer 4
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            
            # Layer 5
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(n_hidden, vocab_size)
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embedding layer
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                if layer.weight.shape[0] == vocab_size:  # output layer
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                else:  # hidden layers
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # Changed to relu
        
        # Initialize batch norm layers
        for layer in self.net:
            if isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Embed the input tokens
        emb = self.embedding(x)  # shape: (batch_size, block_size, n_embd)
        
        # Flatten the embeddings for the linear layers
        x = emb.view(emb.shape[0], -1)  # shape: (batch_size, block_size * n_embd)
        
        # Forward pass through the network
        logits = self.net(x)  # shape: (batch_size, vocab_size)
        
        return logits

# %%
# Initialize the model
n_embd = 64  # embedding dimension
n_hidden = 256  # hidden layer dimension
model = CharLanguageModel(vocab_size, n_embd, n_hidden, block_size)

# Print model summary
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# %%
# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
batch_size = 64
max_steps = 100000
eval_interval = 1000

# Move data to device
Xtr, Ytr = Xtr.to(device), Ytr.to(device)
Xdev, Ydev = Xdev.to(device), Ydev.to(device)
Xte, Yte = Xte.to(device), Yte.to(device)

# %%
# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')

for step in range(max_steps):
    # Get a random batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    
    # Forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update parameters
    optimizer.step()
    
    # Track training loss
    train_losses.append(loss.item())
    
    # Evaluation
    if step % eval_interval == 0 or step == max_steps - 1:
        model.eval()
        with torch.no_grad():
            # Training loss
            train_logits = model(Xtr[:1000])
            train_loss = F.cross_entropy(train_logits, Ytr[:1000])
            
            # Validation loss
            val_logits = model(Xdev)
            val_loss = F.cross_entropy(val_logits, Ydev)
            val_losses.append(val_loss.item())
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
        
        print(f"Step {step:6d}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")
        model.train()

# %%
# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(np.linspace(0, len(train_losses), len(val_losses)), val_losses, label='Validation loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.show()

# %%
# Evaluate on test set
with torch.no_grad():
    test_logits = model(Xte)
    test_loss = F.cross_entropy(test_logits, Yte)
    print(f"Test loss: {test_loss:.4f}")

# %%
# Sampling function
def generate_toponym(model, max_len=20, temperature=1.0):
    model.eval()
    with torch.no_grad():
        context = [stoi['<S>']] * block_size
        generated = []
        
        for _ in range(max_len):
            # Get logits
            x = torch.tensor([context], device=device)
            logits = model(x)
            
            # Apply temperature
            logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            
            # Update context and generated sequence
            context = context[1:] + [ix]
            generated.append(ix)
            
            # Stop if we generate end token
            if ix == stoi['<E>']:
                break
                
        # Convert to string
        return ''.join(itos[i] for i in generated if i != stoi['<S>'] and i != stoi['<E>'])

# Generate some examples
print("Generated toponyms:")
for _ in range(10):
    print(generate_toponym(model, temperature=0.7))


