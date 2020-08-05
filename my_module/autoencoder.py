from torch import nn, optim

#オートエンコーダの定義
class standard_Autoencoder(nn.Module):
    

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class color_Autoencoder(nn.Module):
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(255, 255//4),
            nn.ReLU(True),
            nn.Linear(255, 255//14),
            nn.ReLU(True),
            nn.Linear(255, 255//28),
            )

        self.decoder = nn.Sequential(
            nn.Linear(255//28, 255//14),
            nn.ReLU(True),
            nn.Linear(255//14, 255//4),
            nn.ReLU(True),
            nn.Linear(255//4, 255),
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
