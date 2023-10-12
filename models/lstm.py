import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, opt, embedding_dim=768, bidirectional=True):

        super(LSTM, self).__init__()
        hidden_dim = opt.lstm_hidden_dim
        dropout = opt.lstm_dropout
        self.hidden_dim = hidden_dim
        self.input_dim = opt.lstm_input_dim if opt.lstm_input_dim else embedding_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_mean_input = opt.lstm_mean_input

        self.ff_hidden_dim = opt.ff_hidden_dim
        self.bn = nn.BatchNorm1d(embedding_dim) if opt.ff_bn else nn.Identity()
        self.dropout = nn.Dropout(opt.ff_dropout) if opt.ff_dropout else nn.Identity()
        if opt.ff_hidden_dim and opt.lstm_input_dim:
            self.linear = nn.Linear(embedding_dim, opt.ff_hidden_dim)
            self.linear2 = nn.Linear(opt.ff_hidden_dim, opt.lstm_input_dim)
            self.act = nn.ReLU()
        elif opt.lstm_input_dim:
            self.linear = nn.Linear(embedding_dim, opt.lstm_input_dim)
        else:
            self.linear = nn.Identity()
        if opt.lstm_mean_input and opt.lstm_input_dim:
            raise ValueError(
                "Can only specify lstm mean input or lstm input dim not both"
            )

        if opt.lstm_mean_input:
            embedding_dim = 1
        if dropout:
            self.lstm = nn.LSTM(
                embedding_dim, hidden_dim, bidirectional=bidirectional, dropout=dropout
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def forward(self, x):
        """Accepts sentence of shape (batch, sequence length, embedd dim)"""
        # sentences, lengths = input

        batch = len(x)
        if self.lstm_mean_input:
            x = x.mean(dim=-1).unsqueeze(-1)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.bn(x)
        if self.ff_hidden_dim:
            x = self.act(x)
            x = self.linear2(x)
        # packed = pack_padded_sequence(
        #     sentences.transpose(0, 1), lengths.int().cpu(), enforce_sorted=False
        # ).to(device)
        # packed_out, _ = self.lstm(packed)

        # out, _ = pad_packed_sequence(packed_out)
        out, _ = self.lstm(x.transpose(0, 1))
        out = out.view(-1, batch, self.num_directions, self.hidden_dim)

        if self.num_directions == 2:
            out1 = out[-1, :, 0, :].squeeze()
            out2 = out[0, :, 1, :].squeeze()
            out = torch.cat([out1, out2], 1)
        else:
            out = out[-1, :, :, :].squeeze()
        return out
