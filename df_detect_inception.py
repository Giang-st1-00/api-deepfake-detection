from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.models import Model as KerasModel
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.optimizers import Adam
from fastapi.responses import JSONResponse
import numpy as np
import uuid
import cv2
import os

app = FastAPI()
image_dimensions = {'height':256, 'width':256, 'channels':3}

IMGWIDTH = 256

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


class MesoInception4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
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

        return KerasModel(inputs = x, outputs = y)

mesoInception4 = MesoInception4()
mesoInception4.load('./weights/MesoInception_DF.h5')


@app.post("/upload-video/")
async def create_upload_file(video_file: UploadFile = File(...)):
    video_contents = await video_file.read()

    # Tạo tệp tạm thời để lưu trữ dữ liệu video
    temp_file_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(video_contents)

    # Tạo đối tượng VideoCapture từ đường dẫn của tệp tạm thời
    video_capture = cv2.VideoCapture(temp_file_path)

    # Tạo thư mục để lưu ảnh từ video
    output_folder = f"/tmp/{uuid.uuid4()}/"
    os.makedirs(output_folder, exist_ok=True)

    # Đọc từng frame của video
    frame_index = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Xử lý frame tại đây nếu cần
        frame = cv2.resize(frame, (256, 256))

        # Lưu frame vào thư mục đầu ra
        frame_file_path = os.path.join(output_folder, f"{frame_index}.jpg")
        cv2.imwrite(frame_file_path, frame)
        frame_index += 1

    video_capture.release()

    # Xóa tệp tạm thời sau khi đã xử lý
    os.remove(temp_file_path)

    # Tiến hành dự đoán cho mỗi frame
    predictions = []
    for file_name in os.listdir(output_folder):
        if file_name.endswith(".jpg"):
            frame_file_path = os.path.join(output_folder, file_name)
            frame_array = cv2.imread(frame_file_path)
            prediction = mesoInception4.predict(np.expand_dims(frame_array, axis=0))
            predictions.append(float(prediction[0][0]))

    
    threshold = 5  
    count_not_one = sum(1 for pred in predictions if pred != 1)
    percentage_not_one = (count_not_one / len(predictions)) * 100

    # Trả về kết quả dự đoán
    result = {"result": percentage_not_one > threshold, "percent": percentage_not_one, "predictions": predictions}
    return JSONResponse(content=result)
