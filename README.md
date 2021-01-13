# predhinata
Predict Hinatazata member who is in pictures  
写真に写っている日向坂メンバーを予測する

# 必要なもの

* cascade file
  haarcascade_mcs_nose.xml  
  haarcascade_frontalface_default.xml

# 準備

## cascade file  
download cascade files from opencv  
opencv公式からカスケードファイルをダウンロードする  
haarcascade_frontalface_default.xml  
https://github.com/opencv/opencv/tree/master/data/haarcascades  
haarcascade_frontalface_default.xml  
https://github.com/opencv/opencv_contrib/tree/1311b057475664b6af118fd1a405888bad4fd839/modules/face/data/cascades  
save to folder "cascade"  
「カスケード」フォルダに保存(predict.py内のパスを通せば別の場所でも良い)

## model file
model files are saved to model  
モデルファイルを「モデル」フォルダに入れる(predict.py内のパスを通せば別の場所でも良い)

# Windowsで動かす

下記のコマンドたちをPowerShellで管理者で実行

## 仮想環境作成

```
python -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 画像ファイルのパス指定
predict.py line 73

## 実行

```
.\venv\Scripts\python.exe hinatablogimg\predict.py

```
