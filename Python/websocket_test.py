from websocket import create_connection
import json

data = {
    "action": "move",
    "room_no": 1,
    "user": "test",
    "pos_x": 2.1,
    "pos_y": 0.1,
    "pos_z": 2,
    "way": "mediapipe",
    "range": 0,
}

ws = create_connection("ws://192.168.0.29:3000")
print("Sending 'Hello, World'...")
ws.send(json.dumps(data))
print("Sent")
print("Receiving...")
result =  ws.recv()
print("Received '%s'" % result)
ws.close()
