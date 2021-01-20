import glob
from tqdm import tqdm 
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras
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
                break
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


def pred_face_for_img_path_list(img_path_list, model_path, cascade_path='./cascade'):
    '''
    画像の取り込みからメンバー予測まで
    '''
    result = []
    nose_cascade = cv2.CascadeClassifier(cascade_path+'/haarcascade_mcs_nose.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path+'/haarcascade_frontalface_default.xml')
    model = keras.models.load_model(model_path)
    for img_path in tqdm(img_path_list):
        img = cv2.imread(img_path)
        face = get_face(img, nose_cascade, face_cascade)
        if len(face) == 0:
            continue
        else:
            m_list = []
            for f in face:
                img_attribute = get_img_attribute(f)
                similar_list = model.predict(img_attribute)[0]
                result.append(similar_list)
    return result

def main():
    '''
    メイン関数
    '''
    #画像のパス　形式はリスト
    img_path_list = glob.glob('')
    model_path = './model/facenet_hinata_model.h5'
    org_similar_list =\
        pred_face_for_img_path_list(img_path_list, model_path, cascade_path='./cascade')
    
    similar_list = np.mean(org_similar_list, axis=0)
    first_member = name_list[similar_list.argsort()[-1]]
    first_persent = similar_list[similar_list.argsort()[-1]]
    second_member = name_list[similar_list.argsort()[-2]]
    second_persent = similar_list[similar_list.argsort()[-2]]
    third_member = name_list[similar_list.argsort()[-3]]
    third_persent = similar_list[similar_list.argsort()[-3]]
    print(f"First   member:{first_member} persent:{first_persent}")
    print(f"Second  member:{second_member} persent:{second_persent}")
    print(f"Third   member:{third_member} persent:{third_persent}")
    

if __name__ == "__main__":
    main()
    