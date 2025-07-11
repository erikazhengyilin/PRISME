import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.model import Autoencoder

# === Settings ===
csv_file = "PRISME/csv_embeddings/prisme.csv"
input_dim = 512   # Adjust based on your CSV columns
batch_size = 32
n_epochs = 100
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Data ===
df = pd.read_csv(csv_file)
gene_names = df["symbol"].values
X = df.drop(columns=["symbol"]).values.astype(np.float32)

# Optional: normalize
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

# === Prepare Dataloader ===
tensor_data = torch.tensor(X).to(device)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model Setup ===
model = Autoencoder(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# === Training Loop ===
best_loss = float("inf")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for (batch,) in tqdm(dataloader):
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)

    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_save_path)

print(f"Training done. Model saved to {model_save_path}")