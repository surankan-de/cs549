# train_mine.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import MINE
from utils import PTCTDataset
import matplotlib.pyplot as plt

def mutual_info_mine(model, loader, device):
    model.eval()
    total = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                pt, ct, _ = batch
            else:
                pt, ct = batch
            pt = pt.to(device)
            ct = ct.to(device)
            t_xy = model(pt, ct)
            idx = torch.randperm(pt.size(0))
            ct_shuf = ct[idx]
            t_x_y = model(pt, ct_shuf)
            mi_batch = t_xy.mean() - torch.log(torch.exp(t_x_y).mean()+1e-8)
            total.append(mi_batch.item())
    return float(np.mean(total))

def train_mine(model, dataset, device='cpu', epochs=10, batch_size=256, lr=1e-4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    mi_history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if len(batch) == 3:
                pt, ct, _ = batch
            else:
                pt, ct = batch
            pt = pt.to(device)
            ct = ct.to(device)
            t_xy = model(pt, ct)
            idx = torch.randperm(pt.size(0))
            ct_shuf = ct[idx]
            t_x_y = model(pt, ct_shuf)
            loss = - (t_xy.mean() - torch.log(torch.exp(t_x_y).mean()+1e-8))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))
        mi_est = mutual_info_mine(model, loader, device)
        mi_history.append(mi_est)
        print(f"Epoch {epoch+1} MI estimate: {mi_est:.6f}")
    return mi_history


if __name__ == "__main__":
    # quick demo using random data
    from data_gen import generate_plaintexts_random, aes_ecb_encrypt_batch
    pts = generate_plaintexts_random(4096, 16)
    key, cts = aes_ecb_encrypt_batch(pts)
    ds = PTCTDataset(pts, cts, pt_len=16, ct_len=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MINE(16, 16, hidden=256)
    hist = train_mine(model, ds, device=device, epochs=5, batch_size=256)
    plt.plot(hist); plt.title("MI estimate"); plt.show()
