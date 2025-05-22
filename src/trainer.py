import pickle, json
from torch.utils.data import Dataset
import pandas as pd
import torch, torch.nn as nn, pytorch_lightning as pl
import json, pickle
from torch.utils.data import DataLoader

def build_vocab(embed_pkl="models/item2vec.pkl", out_json="models/vocab.json"):
  with open(embed_pkl, "rb") as f:
    embed = pickle.load(f)

  # idx 0 = PAD / UNK
  vocab = {"<PAD>": 0}
  for i, pid in enumerate(embed.keys(), start=1):
    vocab[pid] = i

  with open(out_json, "w") as f:
    json.dump(vocab, f)

  print(f"[vocab] |V|={len(vocab)} → {out_json}")


class BasketSeqDataset(Dataset):
    def __init__(self, pkl_path, vocab_json, max_len=30):
        self.data  = pd.read_pickle(pkl_path)
        with open(vocab_json) as f:
            self.vocab = json.load(f)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, pid_list):
        ids = [self.vocab.get(str(pid), 0) for pid in pid_list]
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
          ids += [0] * (self.max_len - len(ids))
        else:
          ids[-1] = ids[-2] 
        return ids

    def __getitem__(self, idx):
        row  = self.data.iloc[idx]
        seq  = self.encode(row["product_id"])

        # make sure we still have at least one input token + 1 label
        if seq.count(0) >= self.max_len - 1:           # degenerate basket
            seq = [0, 0] + [0] * (self.max_len - 2)

        input_ids = seq[:-1]                 # length  (max_len-1)
        target    = seq[-1]                  # scalar
        return torch.LongTensor(input_ids), torch.LongTensor([target])
    
class NextItemBC(pl.LightningModule):
    def __init__(self,
                 vocab_json="models/vocab.json",
                 embed_pkl="models/item2vec.pkl",
                 d_model=128, nhead=8, n_layers=2, lr=1e-3):
        super().__init__()
        with open(vocab_json) as f:
            self.vocab = json.load(f)
        with open(embed_pkl, "rb") as f:
            embed = pickle.load(f)

        self.n_items = len(self.vocab)

        # Build fixed embedding matrix (idx 0 = zeros for PAD/UNK)
        emb_dim = next(iter(embed.values())).shape[0]
        weight = torch.zeros((self.n_items, emb_dim))
        for pid, vec in embed.items():
            weight[self.vocab[pid]] = torch.tensor(vec)

        self.item_emb = nn.Embedding.from_pretrained(weight, freeze=True)

        self.pos_emb  = nn.Embedding(32, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
                                                   nhead=nhead,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, self.n_items)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.lr = lr

    def forward(self, x):
        # x: [B, L] token indices
        positions = torch.arange(x.size(1), device=x.device)
        positions = self.pos_emb(positions)[None, :, :]
        x = self.item_emb(x) + positions
        h = self.transformer(x)
        # Predict next token = last time-step output
        out = self.fc(h[:, -1, :])            # [B, n_items]
        return out

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.squeeze())
        self.log("val_loss", loss, prog_bar=True)

        # Hit@20
        topk = logits.topk(20)[1]
        hit20 = (topk == y).any(dim=1).float().mean()
        self.log("hit20", hit20, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train_bc(batch_size=256, max_epochs=5):
    seqs = pd.read_pickle("data/train.pkl")
    seqs = seqs[seqs["product_id"].str.len() >= 2]
    seqs.to_pickle("data/train.pkl")

    seqs = pd.read_pickle("data/val.pkl")
    seqs = seqs[seqs["product_id"].str.len() >= 2]
    seqs.to_pickle("data/val.pkl")

    seqs = pd.read_pickle("data/test.pkl")
    seqs = seqs[seqs["product_id"].str.len() >= 2]
    seqs.to_pickle("data/test.pkl")

    train_ds = BasketSeqDataset("data/train.pkl", "models/vocab.json")
    val_ds   = BasketSeqDataset("data/val.pkl",   "models/vocab.json")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4)

    model = NextItemBC()
    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        precision=32,
    )
    trainer.fit(model, train_dl, val_dl)
    trainer.save_checkpoint("models/bc.ckpt")
    print("[BC] Saved to models/bc.ckpt")

if __name__ == "__main__":
  build_vocab()
  train_bc()