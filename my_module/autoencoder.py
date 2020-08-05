from torch import nn

#オートエンコーダの定義
class bin_Autoencoder(nn.Module):
    

    def __init__(self):
        super().__init__()
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
    
    def __init__(self,input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size//4),
            nn.ReLU(True),
            nn.Linear(input_size, input_size//14),
            nn.ReLU(True),
            nn.Linear(input_size, input_size//28),
            )

        self.decoder = nn.Sequential(
            nn.Linear(input_size//28, input_size//14),
            nn.ReLU(True),
            nn.Linear(input_size//14, input_size//4),
            nn.ReLU(True),
            nn.Linear(input_size//4, input_size),
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
