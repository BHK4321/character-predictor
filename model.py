import torch
import torch.nn as nn

class MaskedCharacterPredictor(nn.Module):
    def __init__(self, vocab_size=28, max_len=45, emb_dim=128, hidden_dim=1024, ablate={}):
        super().__init__()
        self.ablate = ablate
        self.char_emb = nn.Embedding(vocab_size, emb_dim)
        self.ctx_emb = nn.Embedding(4, 32)

        self.pattern_cnn = nn.Sequential(
            nn.Conv1d(emb_dim, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2)
        )

        self.encoder = nn.LSTM(emb_dim + 32, hidden_dim, bidirectional=True, batch_first=True)

        self.pos_prior_mlp = nn.Sequential(
            nn.Linear(1 + 1 + 64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 26)
        )

        def decoder():
            return nn.Sequential(
                nn.Linear(hidden_dim*2 + 26, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 26)
            )

        self.left_decoder = decoder()
        self.right_decoder = decoder()
        self.both_decoder = decoder()

    def forward(self, word_state, position_context, word_length, blank_count, next_to_vowel):
        B, L = word_state.size()
        emb = self.char_emb(word_state)
        cnn_feat = self.pattern_cnn(emb.transpose(1, 2)).transpose(1, 2)
        ctx = self.ctx_emb(position_context)
        encoded, _ = self.encoder(torch.cat([emb, ctx], -1))

        pos_scores = []
        for i in range(L):
            is_blank = (word_state[:, i] == 0).float().unsqueeze(1)
            bc = blank_count.unsqueeze(1).float() / L
            pos_input = torch.cat([is_blank, bc, cnn_feat[:, i, :]], -1)
            pos_scores.append(self.pos_prior_mlp(pos_input).unsqueeze(1))
        priors = torch.cat(pos_scores, 1)  # [B, L, 26]

        out = torch.zeros(B, L, 26, device=word_state.device)
        for i in range(L):
            h = encoded[:, i, :]
            ptype = position_context[:, i]
            inp = torch.cat([h, priors[:, i, :]], -1)
            out[ptype==1, i, :] = self.left_decoder(inp[ptype==1])
            out[ptype==2, i, :] = self.right_decoder(inp[ptype==2])
            out[ptype==3, i, :] = self.both_decoder(inp[ptype==3])

        return out
