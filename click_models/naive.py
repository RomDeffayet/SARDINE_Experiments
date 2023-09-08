import torch
from torch import nn
import pytorch_lightning as pl

class NaiveClickModel(pl.LightningModule):
    def __init__(self, cm_lr: float, state_dim: int, num_items: int, slate_size: int, item_embedd_size: int, hidden_size: int, **kwargs):
        super().__init__()

        self.item_embedding = nn.Embedding(num_items, item_embedd_size)
        self.relevance_model = nn.Sequential(
            nn.Linear(state_dim + item_embedd_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.lr = cm_lr
        self.slate_size = slate_size
    
    def forward(self, state, slate):
        new_size = [-1 for _ in range(len(state.shape) + 1)]
        new_size[-2] = self.slate_size
        state = state.unsqueeze(-2).expand(new_size)

        item_embedding = self.item_embedding(slate)
        pred = self.relevance_model(torch.cat([state, item_embedding], dim = -1)).squeeze()
        return pred, pred   # Click probability, relevance probability

    def training_step(self, batch, batch_idx):
        state = batch.observations["state"]
        slate, clicks = batch.actions, batch.observations["clicks"].float()
        batch_size, slate_size = slate.shape

        pred = self(state, slate)[0]
        
        loss = nn.BCEWithLogitsLoss()(pred, clicks)
        self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state = batch.observations["state"]
        slate, clicks = batch.actions, batch.observations["clicks"].float()
        batch_size, slate_size = slate.shape

        pred = self(state, slate)[0]
        
        loss = nn.BCEWithLogitsLoss()(pred, clicks)
        self.log("val/loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)