import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import os

from feature_pipeline import utils

logger = utils.get_logger(__name__)


class Autoencoder(nn.Module):
    """
    Autoencoder model to extract features from cat and dog images.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Both networks should be symmetric in the dimensions of the layers
        # Encoder Network
        self.encoder = nn.Sequential(nn.Linear(10000, 1024),
                                     nn.ReLU(True),
                                     nn.Linear(1024, 256),
                                     nn.ReLU(True),
                                     nn.Linear(256, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 32)) # Number of features per image will be 32
        # Decoder Network
        self.decoder = nn.Sequential(nn.Linear(32, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 256),
                                     nn.ReLU(True),
                                     nn.Linear(256, 1024),
                                     nn.ReLU(True),
                                     nn.Linear(1024, 10000), # Output is the images dimensions
                                     nn.Tanh()) # Ensuring output is in the range [-1, 1]

    def forward(self, x):
        x = x.view(-1, 10000)  # Explicitly reshape input to (batch_size, 10000)
        x = self.encoder(x)
        x = self.decoder(x)
        # Reshape back to image dimensions
        x = x.view(-1, 1, 100, 100)  # Reshape output to (batch_size, channels, height, width)
        return x

    def encode(self, x):
        x = x.view(-1, 10000)  # Ensure input is flattened
        return self.encoder(x)

    
class AutoencoderTrainer:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.checkpoint_path = "checkpoints/autoencoder_checkpoint.pth"

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")


    def load_checkpoint(self) -> tuple[int, float]:
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Loaded checkpoint from epoch {epoch}, with loss {loss:.4f}")

        return epoch, loss
    

    def save_checkpoint(self, 
                        epoch: int, 
                        best_loss: float) -> None:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': best_loss,
        }, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    
    def load_data(self, image_data: list[Image]) -> DataLoader:
        height, width = 100, 100 # 10000 pixels

        transform = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        tensors = torch.stack([transform(img) for img in image_data])
        tensor_dataset = TensorDataset(tensors, tensors) # Autoencoder is trained to reconstruct the input
        dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

        return dataloader


    def train(self, dataloader: DataLoader) -> None:
        if os.path.isfile(self.checkpoint_path):
            self.load_checkpoint()
        else:
            best_loss = float('inf')

            num_epochs = 20
            self.model.train()
            for epoch in range(num_epochs):
                #logger.info(f"Epoch [{epoch}/{num_epochs}]")
                total_loss = 0
                for data, _ in dataloader:
                    data = data.to(self.device)
                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, data)
                    # Backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(epoch, best_loss)


    def extract_features(self, dataloader):
        self.model.eval()
        all_features = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device) 
                features = self.model.encode(images)
                all_features.append(features.cpu())
                
        # Concatenate all features from all batches
        all_features = torch.cat(all_features, dim=0)
        return all_features
