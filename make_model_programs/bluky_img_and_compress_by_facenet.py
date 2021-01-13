from keras_facenet import FaceNet
import numpy as np
from PIL import Image

IMG_SIZE = 128

data = np.load("./pic_numpy/pic.npz")
x = data["arr_0"]
y = data["arr_1"]
lng_x = len(x)
i = 1
out_x=[]
out_y=[]
embedder = FaceNet()
for in_x, in_y in zip(x, y):
    print(f'{i}/{lng_x}')
    i += 1
    add_img = embedder.embeddings(in_x.reshape((1,IMG_SIZE,IMG_SIZE,3)))
    out_x.append(add_img[0])
    out_y.append(in_y)
    pilimg = Image.fromarray(in_x)
    for angle in [-20,-15,-10,-5,0,5,10,15,20]:
        add_img = pilimg.rotate(angle, expand=True)
        add_img = add_img.resize((IMG_SIZE, IMG_SIZE))
        add_img = np.array(add_img).reshape((1,IMG_SIZE,IMG_SIZE,3))
        add_img = embedder.embeddings(add_img)
        out_x.append(add_img[0])
        out_y.append(in_y)

out_x = np.array(out_x)
out_y = np.array(out_y)
np.savez("pic_numpy/face_net_pic.npz", out_x, out_y)