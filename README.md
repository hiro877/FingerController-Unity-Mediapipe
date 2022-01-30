# FingerController-Unity-Mediapipe
カメラで認識した指に連動してオブジェクトが動くアプリです。
Mediapipe, Unity, NodeJsを使用してWebsocket通信で実現しました。
・指でオブジェクトを動かす  
![gif1](https://user-images.githubusercontent.com/65473130/151682970-f1bdd9a7-637c-4d36-a0a0-83c8f9104f51.gif)  
・指のカタチ認識
　”１”、”２”、”３”、”４”、”５”、”OK”、キツネ”、”いいね”　※左記以外は”１”とする
![gif2](https://user-images.githubusercontent.com/65473130/151682975-a908408d-53fb-42d9-b028-095e2d31bffd.gif)  

# Reference code
[mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample)  

 # Implementation Description
 ブログに記載してますので見てください！
 [カメラを用いた指認識コントローラーの作り方①リアルタイム同期通信【Unity, Mediapipe, Websocket】](https://www.hiro877.com/entry/unity-fingercontroller1)  
 [カメラを用いた指認識コントローラーの作り方②指認識【Unity, Mediapipe, Websocket】](https://www.hiro877.com/entry/unity-fingercontroller2)  
 
 # Requirement
 [MediaPipe in Python](https://google.github.io/mediapipe/getting_started/python.html)
 
 ```
 pip install mediapipe
 ```
 
 # DEMO

1. JavaScriptsフォルダ内のws.jsを起動する
```
node we.js
```
2. UnityHub→リストに追加でUnityProjectsフォルダを選択し起動する。
3. 1で作成した環境からPythonフォルダ内のclient.pyを実行する。
```
python Python/client.py
```
