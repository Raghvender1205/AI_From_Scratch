from unicodedata import bidirectional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class CNNLayerNorm(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layernorm = nn.LayerNorm(n_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3).contiguous() # [batch, channel, time, feature]
        x = self.layernorm(x)

        return x.transpose(2, 3).contigous() # [batch, channel, feature, time]
    

class Residual(nn.Module):
    """
    CNN with Residual or skip connections with LayerNorm
    https://arxiv.org/pdf/1603.05027.pdf
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_features):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm1 = CNNLayerNorm(n_features)
        self.layernorm2 = CNNLayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x # [batch, channel, feature, time]
        x = self.layernorm1(x)
        x = F.relu()
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.layernorm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        x += residual

        return x # [batch, channel, feature, time]
    
class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(rnn_dim, hidden_size, num_layers=1, batch_first=batch_first, bidirectional=bidirectional)
        self.layernorm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)

        return x


class DeepSpeech(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_classes, n_features, stride=2, dropout=0.1):
        super(DeepSpeech, self).__init__()
        n_features = n_features // 2
        self.conv1 = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2) # Extact Heirachial Features
        # Residual CNN Layers
        self.rescnn_layers = nn.Sequential(*[
            Residual(32, 32, kernel=3, stride=1, dropout=dropout, n_features=n_features)
            for _ in range(n_cnn_layers)
        ])
        self.ffn = nn.Linear(n_features*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim*2, hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim), # BiRNN returns rnn_dim*2    
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(rnn_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[2], sizes[3]) # [batch, feature, time]
        x = x.transpose(1, 2) # [batch, time, feature]
        x = self.ffn(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)

        return x