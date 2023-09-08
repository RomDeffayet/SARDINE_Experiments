import torch
from torch import nn
import pytorch_lightning as pl

class PBM(pl.LightningModule):
    def __init__(self, cm_lr: float, state_dim: int, num_items: int, slate_size: int, item_embedd_size: int, hidden_size: int, **kwargs):
        super().__init__()

        self.item_embedding = nn.Embedding(num_items, item_embedd_size)
        self.relevance_model = nn.Sequential(
            nn.Linear(state_dim + item_embedd_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.propensities = nn.Embedding(slate_size, 1)

        self.lr = cm_lr
        self.slate_size = slate_size
    
    def forward(self, state, slate):
        new_size = [-1 for _ in range(len(state.shape) + 1)]
        new_size[-2] = self.slate_size
        new_state = state.unsqueeze(-2).expand(new_size)
        item_embedding = self.item_embedding(slate)

        relevance_prob = self.relevance_model(torch.cat([new_state, item_embedding], dim = -1))
        
        prop_size = list(state.shape)
        prop_size[-2:] = [-1, -1]
        propensities = nn.Sigmoid()(self.propensities(torch.arange(self.slate_size))).expand(prop_size)
        click_prob = relevance_prob * propensities

        return click_prob.squeeze(), relevance_prob.squeeze()

    def training_step(self, batch, batch_idx):
        state = batch.observations["state"]
        slate, clicks = batch.actions, batch.observations["clicks"].float()

        click_prob, _ = self(state, slate)
        
        loss = nn.BCELoss()(click_prob, clicks)
        self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state = batch.observations["state"]
        slate, clicks = batch.actions, batch.observations["clicks"].float()

        click_prob, _ = self(state, slate)
        
        loss = nn.BCELoss()(click_prob, clicks)
        self.log("val/loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)