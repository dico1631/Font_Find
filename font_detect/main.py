import base64
import io
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from tensorflow.keras import backend
from django.conf import settings
import sys
import os
from io import BytesIO
import Algorithmia
import urllib
# import imutils
import requests

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
INPUT_MODEL_PATH = "./model/model.h5"

def detect_who(model, txt_image):
    name = ""
    result = model.predict(txt_image) # 배열형식
    rlist = []

    for i in range(len(result[0])):
        rlist.append(int(result[0][i]*100))
    

    result_msg = f"배달의민족 폰트 가능성: {result[0][0]*100:.3f}% / 카페24 폰트 가능성: {result[0][1]*100:.3f}% / 조선일보 명조체 가능성: {result[0][2]*100:.3f}% / 마포 폰트 가능성: {result[0][3]*100:.3f}% / 나눔손글씨펜 폰트 가능성: {result[0][4]*100:.3f}% / 나눔스퀘어 폰트 가능성: {result[0][5]*100:.3f}%"

    name_number_label = np.argmax(result)
    if name_number_label == 0:
        name = "Baemin"
    elif name_number_label == 1:
        name = "Cafe24"
    elif name_number_label == 2:
        name = "Chosun"
    elif name_number_label == 3:
        name = "Mapo"
    elif name_number_label == 4:
        name = "Nanumpen"
    elif name_number_label == 5:
        name = "NanumSquare"
        
    
    return (name, result_msg, rlist)

def cascade(url):
    input = {
    "input": url,
    "output": "data://.algo/character_recognition/TextDetectionCTPN/temp/receipt.png"
    }
    client = Algorithmia.client('simpwOTVV5icdkd+wRHy1O0ByZC1')
    algo = client.algo('character_recognition/TextDetectionCTPN/0.2.0')
    algo.set_options(timeout=300) # optional
    result=algo.pipe(input).result['boxes']
    return result
    #{'confidence': 0.9713579416275024, 'x0': 61.44, 'x1': 439.68, 'y0': 229.79373046875, 'y1': 284.44853515625}


def url_to_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image_rect = np.array(image)
    return image_rect


def detect(image_url):

    image_rect = np.asarray(Image)
    model = tf.keras.models.load_model(INPUT_MODEL_PATH)
    result_list = []
    result_name = []
    result_img_list = []
    result_percent_list = []

    results = cascade(image_url) #위치가 딕셔너리를 담은 리스트로 리턴
    if len(results) != 0:
        print(f"인식된 얼굴의 수: {len(results)}")

        for result in results: 
            xpos = int(result['x0'])
            xpos2 = int(result['x1'])
            ypos = int(result['y0'])
            ypos2 = int(result['y1'])
        
            print(xpos, xpos2, ypos, ypos2)
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image_rect = np.array(image)


            #이미지 크롭하기
            image_crop=image_rect[ypos:ypos2, xpos:xpos2]

            #모델 넣기전 전처리
            image = cv2.resize(image_crop, (76,76))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(image, cv2.CV_8U)
            laplacian = tf.cast(laplacian, tf.float32)
            laplacian = np.expand_dims(laplacian, axis=(0,3))

            print("라플라시안",laplacian.shape)

            
            name, result_msg, result = detect_who(model, laplacian)
            result_percent_list.append(result)

             # 8) 이미지를 PNG파일로 변환 
            shape_img = image_crop.shape
            send_image = cv2.resize(image_crop, (shape_img[1]*2, shape_img[0]*2))
            is_success, img_buffer = cv2.imencode(".png", send_image)
            if is_success:
                # 이미지 -> 메인 메모리의 바이너리 형태 
                io_buffer = io.BytesIO(img_buffer)
                result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")
                result_img_list.append(result_img)

            # 9) tensorflow에서 session이 닫히지 않는 문제
            backend.clear_session()


            result_list.append(result_msg)
            result_name.append(name)
            
    else:
        print("지정한 이미지 파일에서 문자를 인식할 수 없습니다.")



    return (result_list, result_name, result_img_list, result_percent_list)
    
