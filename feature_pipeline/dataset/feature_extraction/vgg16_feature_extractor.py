import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import models, layers, optimizers
import numpy as np
from PIL import Image
import os

from feature_pipeline import utils

logger = utils.get_logger(__name__)


# TODO: Adjust the trainer class, not working for the model right now!!!!

class VGG16FeatureExtractor(nn.Module):
    """
    Autoencoder model to extract features from cat and dog images.
    """
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        # Both networks should be symmetric in the dimensions of the layers
        # Encoder Network
        conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(100, 100, 1))
        
        self.model = models.Sequential()
        self.model.add(conv_base)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        conv_base.trainable = False

        self.model.compile(loss='binary_crossentropy',
                            optimizer=optimizers.RMSprop(),
                            metrics=['acc'])

    
class VGG16FeatureExtractorTrainer:
    def __init__(self) -> None:
        self.model = VGG16FeatureExtractor()
        self.checkpoint_path = "checkpoints/vgg16_feature_extractor_checkpoint.pth"
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

    
    def _preprocess_image(self, img, target_size=(100, 100)) -> np.array:
        img = img.resize(target_size)
        # Convert the image to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        # Convert the image to a numpy array
        img_array = keras_image.img_to_array(img)
        # Add a batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array


    def load_data(self, image_data: list[Image]) -> list[np.array]:
        processed_images = []
        for img in image_data:
            processed_img = self._preprocess_image(img)
            processed_images.append(processed_img)

        return processed_images


    def train(self, dataloader: list[np.array]) -> None:
        history = self.model.fit(train_generator,
                                steps_per_epoch=50,
                                epochs=10,
                                validation_data=validation_generator,
                                validation_steps=50,
                                verbose=1)
        self.model.save(self.checkpoint_path)


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
