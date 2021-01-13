import face_recognition
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

name_list = pd.read_csv("./member_list.csv")
name_list = name_list["name"].values.tolist()
j = 0
for name in name_list:
    print(name)
    file_list = glob(f"./hinatablogimg/img/{name}/*")
    os.makedirs(f"./hinatablogimg/cut_img/{name}")
    #file_list = file_list[:5]
    i = 0
    for file in tqdm(file_list):
        try:
            image = face_recognition.load_image_file(file)
        except:
            continue
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            image = Image.fromarray(image)
            image.save(f"./hinatablogimg/cut_img/ng/pic_{j}.jpg")
            j += 1
            continue

        for (x1, y1, x2, y2) in face_locations:
            x_start = min(x1, x2)
            x_end = max(x1, x2)
            y_start = min(y1, y2)
            y_end = max(y1, y2)
            face = image[x_start:x_end, y_start:y_end]
            pil_img = Image.fromarray(face)
            pil_img.save(f"./hinatablogimg/cut_img/{name}/{name}_pic_{i}.jpg")
            i += 1
    