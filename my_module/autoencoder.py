from torch import nn

#オートエンコーダの定義
class bin_autoencoder(nn.Module):
    
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

class color_autoencoder(nn.Module):
    
    def __init__(self,input_size):
        super().__init__()
        self.encoder = nn.Sequential(

            nn.Linear(input_size, input_size//2),
            nn.ReLU(True),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(True),
            nn.Linear(input_size//4, input_size//8),
            nn.ReLU(True),
            nn.Linear(input_size//8, input_size//16),
            nn.ReLU(True),
            nn.Linear(input_size//16, input_size//28),

        )

        self.decoder = nn.Sequential(

            nn.Linear(input_size//28, input_size//16),
            nn.ReLU(True),
            nn.Linear(input_size//16, input_size//8),
            nn.ReLU(True),
            nn.Linear(input_size//8, input_size//4),
            nn.ReLU(True),
            nn.Linear(input_size//4, input_size//2),
            nn.ReLU(True),
            nn.Linear(input_size//2, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class lab_autoencoder(nn.Module):
    
    def __init__(self,input_size):
        super().__init__()
        self.encoder = nn.Sequential(

            nn.Linear(input_size, input_size//2),
            nn.ReLU(True),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(True),
            nn.Linear(input_size//4, input_size//8),
            nn.ReLU(True),
            nn.Linear(input_size//8, input_size//16),
            nn.ReLU(True),
            nn.Linear(input_size//16, input_size//28),

        )

        self.decoder = nn.Sequential(

            nn.Linear(input_size//28, input_size//16),
            nn.ReLU(True),
            nn.Linear(input_size//16, input_size//8),
            nn.ReLU(True),
            nn.Linear(input_size//8, input_size//4),
            nn.ReLU(True),
            nn.Linear(input_size//4, input_size//2),
            nn.ReLU(True),
            nn.Linear(input_size//2, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

