from torch import nn

class mask_detector(nn.Module):  # Convolution Neural Network model for Face Mask detection
    def __init__(self):
        super(mask_detector, self).__init__()

        # Note: The following model is the modified version of that from the work of GitHub user JadHADDAD92 [Link: https://github.com/JadHADDAD92/covid-mask-detector/]

        self.convLayer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))
        self.convLayer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)))
        self.convLayer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3,3)), nn.BatchNorm2d(128), nn.ReLU(),nn.MaxPool2d(kernel_size=(2, 2)))
        self.linearLayers = nn.Sequential(nn.Linear(in_features=2048, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=2))
        
        # Initializing layer weights
        for sequential in [self.convLayer1, self.convLayer2, self.convLayer3, self.linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, image):
        # image: dim -> (batch_size, 3, H, W)

        out = self.convLayer1(image)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.reshape(-1, 2048)
        out = self.linearLayers(out)
        return out