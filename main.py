import sys

# Get the Python version information
version_info = sys.version_info

# Print the Python version in a readable format
print(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
print("Full version info:")
print(sys.version)

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pydantic import BaseModel
import uuid
import io

app = FastAPI()
image_dimensions = {'height':256, 'width':256, 'channels':3}

class ImageInput(BaseModel):
    file: UploadFile


class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)


class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])
    
    def init_model(self): 
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        # vanishing gradients
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)
    
meso = Meso4()
meso.load('./weights/Meso4_DF.h5')


@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    
    # Mở ảnh sử dụng PIL
    image = Image.open(io.BytesIO(contents))

    # Đảm bảo ảnh là RGB (3 kênh màu)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ảnh để phù hợp với mô hình
    image = image.resize((256, 256))
    
    # Chuyển đổi ảnh thành mảng numpy
    image_array = np.array(image)

    # Mở rộng kích thước để tạo một batch với kích thước là 1
    prediction = meso.predict(np.expand_dims(image_array, axis=0)) 
    
    return JSONResponse(content={"prediction": prediction.tolist()})


@app.get("/check")
def read_root():
    return {"Hello": "World"}