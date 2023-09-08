import torch
from torch import nn
import numpy as np

class AbstractStateEncoder(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        if args.state_dim is None:
            self.state_dim = np.array(env.single_observation_space.shape).prod()
        else:
            self.state_dim = args.state_dim
        self.num_items = env.envs[0].unwrapped.num_items
        self.seq_len = args.sampled_seq_len
        self.rec_size = env.single_observation_space["clicks"].shape[0]
        self.num_topics = env.single_observation_space["hist"].shape[0]

        if args.ideal_se:
            item_features = env.envs[0].unwrapped.item_embedd
            self.item_embedd_dim = item_features.shape[1]
            self.item_embeddings = nn.Embedding.from_pretrained(torch.tensor(item_features), freeze=True) # Raw item representations
        else:
            self.item_embedd_dim = args.item_dim_se #32 #16
            self.item_embeddings = nn.Embedding(self.num_items, self.item_embedd_dim)
        self.click_embedd_dim = args.click_dim_se # 2
        self.click_embeddings = nn.Embedding(2, self.click_embedd_dim)

    def reset(self):
        pass

    def step(self, obs):
        pass

    def forward(self, obs):
        pass

class GRUStateEncoder(AbstractStateEncoder):
    def __init__(self, env, args):
        super().__init__(env, args)

        self.num_layers = args.num_layers_se #2
        self.gru = nn.GRU(self.item_embedd_dim + self.rec_size * self.click_embedd_dim + self.num_topics, self.state_dim, batch_first = True, num_layers = self.num_layers)

    def reset(self):
        self.h = torch.zeros(self.num_layers, self.state_dim)

    def step(self, obs):
        items = self.item_embeddings(torch.tensor(obs["slate"])) # (num_envs, rec_size, item_embedd_dim)
        clicks = self.click_embeddings(torch.tensor(obs["clicks"], dtype = torch.long)) # (num_envs, rec_size, click_embedd_dim)
        mean_items = items.mean(dim = -2) # (num_envs, item_embedd_dim)
        clicks = clicks.flatten(start_dim = -2) # (num_envs, rec_size * click_embedd_dim)
        hist = torch.tensor(obs["hist"]) # (num_envs, num_topics)
        out, self.h =  self.gru(torch.cat([mean_items, clicks, hist], dim = -1), self.h)

        return out

    def forward(self, obs):
        items = self.item_embeddings(obs["slate"]) # (batch_size, seq_len, rec_size, item_embedd_dim)
        clicks = self.click_embeddings(obs["clicks"].long()) # (batch_size, seq_len, rec_size, click_embedd_dim)
        mean_items = items.mean(dim = -2) # (batch_size, seq_len, item_embedd_dim)
        clicks = clicks.flatten(start_dim = -2) # (batch_size, seq_len, rec_size * click_embedd_dim)
        hist = obs["hist"] # (batch_size, seq_len, num_topics)

        return self.gru(torch.cat([mean_items, clicks, hist], dim = -1))[0]

class TransformerStateEncoder(AbstractStateEncoder):
    def __init__(self, env, args):
        super().__init__(env, args)

        self.num_envs = env.num_envs
        self.num_layers = args.num_layers_se #2 #1 #2
        self.num_heads = args.num_heads_se #4 #2 #4
        self.dropout_rate = args.dropout_rate_se #0.1
        self.forward_dim = args.forward_dim_se #32 #64
        self.lin = nn.Linear(self.item_embedd_dim + self.rec_size * self.click_embedd_dim + self.num_topics, self.state_dim)
        self.pos_emb = nn.Embedding(self.seq_len, self.state_dim)
        self.pos_emb_getter = torch.arange(self.seq_len, dtype = torch.long)
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        self.emb_norm = nn.LayerNorm(self.state_dim)
        self.mask = ~torch.tril(torch.ones((self.seq_len, self.seq_len), dtype = torch.bool))
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.state_dim, dim_feedforward = self.forward_dim,
                                                   nhead = self.num_heads, dropout = self.dropout_rate, batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = self.num_layers)

    def reset(self):
        self.seq = torch.empty((self.num_envs, 0, self.item_embedd_dim + self.rec_size * self.click_embedd_dim + self.num_topics), dtype=torch.long)

    def step(self, obs):
        items = self.item_embeddings(torch.tensor(obs["slate"])) # (num_envs, rec_size, item_embedd_dim)
        clicks = self.click_embeddings(torch.tensor(obs["clicks"], dtype = torch.long)) # (num_envs, rec_size, click_embedd_dim)
        mean_items = items.mean(dim = -2) # (num_envs, item_embedd_dim)
        clicks = clicks.flatten(start_dim = -2) # (num_envs, rec_size * click_embedd_dim)
        hist = torch.tensor(obs["hist"]) # (num_envs, num_topics)
        obs = torch.cat([mean_items, clicks, hist], dim = -1).unsqueeze(-2) # (num_envs, 1, item_embedd_dim + rec_size * click_embedd_dim + num_topics)
        self.seq = torch.cat([self.seq, obs], dim = -2) # Store the current obs in the history sequence

        # Pass the sequence collected so far through the transformer
        actual_seq_len = self.seq[:, -self.seq_len:, :].size(1) # Actual sequence length, possibly less than seq_len
        out = self.lin(self.seq[:, -actual_seq_len:, :]) # (num_envs, actual_seq_len, state_dim)
        pos = self.pos_emb(torch.arange(actual_seq_len, dtype = torch.long)).unsqueeze(0) # (1, actual_seq_len, state_dim)
        out = self.emb_norm(self.emb_dropout(out + pos)) # (num_envs, actual_seq_len, state_dim)
        #out = out + pos # (num_envs, actual_seq_len, state_dim)
        out = self.transformer(out, mask = self.mask[:actual_seq_len, :actual_seq_len]) # (num_envs, actual_seq_len, state_dim)
        out = out[:, -1, :] # Keep only the output corresponding to the last timestep in the sequence

        return out

    def forward(self, obs):
        items = self.item_embeddings(obs["slate"]) # (batch_size, seq_len, rec_size, item_embedd_dim)
        clicks = self.click_embeddings(obs["clicks"].long()) # (batch_size, seq_len, rec_size, click_embedd_dim)
        mean_items = items.mean(dim = -2) # (batch_size, seq_len, item_embedd_dim)
        clicks = clicks.flatten(start_dim = -2) # (batch_size, seq_len, rec_size * click_embedd_dim)
        hist = obs["hist"] # (batch_size, seq_len, num_topics)
        seq = torch.cat([mean_items, clicks, hist], dim = -1) # (batch_size, seq_len, item_embedd_dim + rec_size * click_embedd_dim + num_topics)

        # Pass the whole sequence through the transformer
        out = self.lin(seq[:, -self.seq_len:, :]) # (batch_size, seq_len, state_dim)
        pos = self.pos_emb(torch.arange(self.seq_len, dtype = torch.long)).unsqueeze(0) # (1, seq_len, state_dim)
        out = self.emb_norm(self.emb_dropout(out + pos)) # (batch_size, seq_len, state_dim)
        #out = out + pos # (batch_size, seq_len, state_dim)
        out = self.transformer(out, mask = self.mask) # (batch_size, seq_len, state_dim)

        return out
