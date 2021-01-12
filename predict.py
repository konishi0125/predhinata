import cv2
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras_facenet import FaceNet

name_list = pd.read_csv('./member_list.csv')
name_list = name_list['name'].values.tolist()

def get_face(img, nose_cascade, face_cascade):
    '''
    取り込まれた画像から顔を切り出す
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    if(len(faces) != 0):
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            nose = nose_cascade.detectMultiScale(face)
            if len(nose) != 0:
                out.append(face)
    return out


def get_img_attribute(img):
    '''
    facenetを使って顔画像の特徴量を抽出する
    '''
    IMG_SIZE = 128
    proc_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    proc_img = np.array(proc_img)
    proc_img = proc_img.reshape((1, IMG_SIZE, IMG_SIZE, 3))
    embedder = FaceNet()
    img_attribute = embedder.embeddings(proc_img).reshape((1,512))
    return img_attribute

def pred_member(model, img_attribute):
    '''
    メンバーを予測する
    '''
    result = model.predict(img_attribute)
    member_index = result.argmax()
    return member_index


def pred_face_for_img_path_list(img_path_list, model_path, cascade_path='./cascade'):
    '''
    画像の取り込みからメンバー予測まで
    '''
    result = []
    nose_cascade = cv2.CascadeClassifier(cascade_path+'/haarcascade_mcs_nose.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path+'/haarcascade_frontalface_default.xml')
    model = keras.models.load_model(model_path)
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        face = get_face(img, nose_cascade, face_cascade)
        if len(face) == 0:
            result.append(None)
        else:
            m_list = []
            for f in face:
                img_attribute = get_img_attribute(f)
                member_index = pred_member(model, img_attribute)
                member = name_list[member_index]
                m_list.append(member)
            result.append(m_list)
    return result

def main():
    img_path_list = ['./downloads/hina_kawata/1.1000_1000_102400.jpg']
    model_path = './model/facenet_hinata_model.h5'
    member_list =\
        pred_face_for_img_path_list(img_path_list, model_path, cascade_path='./cascade')
    print(member_list)

if __name__ == "__main__":
    main()
    