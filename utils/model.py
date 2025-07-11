# a pytorch model of autoencoder
import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim1=256, encoding_dim2=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim1),
            nn.LeakyReLU(),
            nn.Linear(encoding_dim1, encoding_dim2),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim2, input_dim)
            # nn.LeakyReLU(),
            # nn.Linear(encoding_dim1, input_dim)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class BlockoutAutoencoder(nn.Module):
    '''
    During inference, times the ratio of blocking to ensure the correct scale
    '''
    def __init__(self, input_dim, encoding_dim1=256, encoding_dim2=64):
        super(BlockoutAutoencoder, self).__init__()
        self.encoding = nn.Linear(input_dim, encoding_dim1)

        self.encoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(encoding_dim1, encoding_dim2),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim2, encoding_dim1),
            nn.LeakyReLU(),
            nn.Linear(encoding_dim1, input_dim),
            )

    def forward(self, x, blockout_weight=1):
        x = self.encoding(x) * blockout_weight
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim1=2048, encoding_dim2=1024, encoding_dim3=512, encoding_dim4=64):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim2),
            nn.LeakyReLU(),
            nn.Linear(encoding_dim2, encoding_dim3),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim3, input_dim)
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x
