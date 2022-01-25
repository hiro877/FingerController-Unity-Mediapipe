#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc

import math

from websocket import create_connection
import json

data = {
    "action": "move",
    "room_no": 1,
    "user": "mediapipe",
    "pos_x": 2.1,
    "pos_y": 0.1,
    "pos_z": 0,
    "way": "mediapipe",
    "range": 0,
}

data_connect = {
    "action": "connect",
    "room_no": 1,
    "user": "mediapipe",
    "pos_x": 2,
    "pos_y": 0,
    "pos_z": 2,
    "way": "neutral",
    "range": 0,
}

data_disconnect = {
    "action": 'disconnect',
    "room_no": 1,
    "user": 'mediapipe',
    "pos_x": 0,
    "pos_y": 0,
    "pos_z": 0,
    "way": 'neutral',
    "range": 0
}

def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        ws.send(data)
        ws.close()
    thread.start_new_thread(run, ())

# 2頂点の距離の計算
def calcDistance(p0, p1):
  a1 = p1.x-p0.x
  a2 = p1.y-p0.y
  return math.sqrt(a1*a1 + a2*a2)

# 3頂点の角度の計算
def calcAngle(p0, p1, p2):
  a1 = p1.x-p0.x
  a2 = p1.y-p0.y
  b1 = p2.x-p1.x
  b2 = p2.y-p1.y
  angle = math.acos( (a1*b1 + a2*b2) / math.sqrt((a1*a1 + a2*a2)*(b1*b1 + b2*b2)) ) * 180/math.pi
  return angle

# 指の角度の合計の計算
def cancFingerAngle(p0, p1, p2, p3, p4):
  result = 0
  result += calcAngle(p0, p1, p2)
  result += calcAngle(p1, p2, p3)
  result += calcAngle(p2, p3, p4)
  return result

# 指ポーズの検出
def detectFingerPose(landmarks):
  # 指のオープン・クローズ
  thumbIsOpen = cancFingerAngle(landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]) < 70
  firstFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[5], landmarks[6], landmarks[7], landmarks[8]) < 100
  secondFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[9], landmarks[10], landmarks[11], landmarks[12]) < 100
  thirdFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[13], landmarks[14], landmarks[15], landmarks[16]) < 100
  fourthFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[17], landmarks[18], landmarks[19], landmarks[20]) < 100

  # ジェスチャー
  if (calcDistance(landmarks[4], landmarks[8]) < 0.1 and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen):
    return "OK"
  elif (calcDistance(landmarks[4], landmarks[12]) < 0.1 and calcDistance(landmarks[4], landmarks[16]) < 0.1 and firstFingerIsOpen and fourthFingerIsOpen):
    return "キツネ"
  elif (thumbIsOpen and not firstFingerIsOpen and (not secondFingerIsOpen) and (not thirdFingerIsOpen) and (not fourthFingerIsOpen)):
    return "いいね"
  elif (thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen):
    return "５"
  elif ((not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen):
    return "４"
  elif ((not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and (not fourthFingerIsOpen)):
    return "３"
  elif ((not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and (not thirdFingerIsOpen) and (not fourthFingerIsOpen)):
    return "２"
  elif ((not thumbIsOpen) and firstFingerIsOpen and (not secondFingerIsOpen) and (not thirdFingerIsOpen) and (not fourthFingerIsOpen)):
    return "１"
  else:
    return "１"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default))',
                        type=int,
                        default=1)

    parser.add_argument("--max_num_hands", type=int, default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_complexity = args.model_complexity

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=model_complexity,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # World座標プロット ########################################################
    if plot_world_landmark:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        r_ax = fig.add_subplot(121, projection="3d")
        l_ax = fig.add_subplot(122, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    # Websocket ########################################################
    ws = create_connection("ws://192.168.0.29:3000")
    ws.send(json.dumps(data_connect))
    origin_pos = []
    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        # 描画 ################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # # 手の平重心計算
                # cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # # 外接矩形の計算
                # brect = calc_bounding_rect(debug_image, hand_landmarks)
                # # 描画
                # debug_image = draw_landmarks(debug_image, cx, cy,
                #                              hand_landmarks, handedness)
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)

                # DetectFingerPose ##########################
                landmark_point = []
                index_finger_pos = []
                for index, landmark in enumerate(hand_landmarks.landmark):
                    landmark_point.append(landmark)

                    image_width, image_height = debug_image.shape[1], debug_image.shape[0]
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    if index == 8:  # 人差指：指先
                        cv.circle(debug_image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                        cv.circle(debug_image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
                        index_finger_pos = [landmark_x, landmark_y]
                        
                finger_pose = detectFingerPose(landmark_point)
                print(finger_pose)
                if finger_pose == "１":
                    if origin_pos:
                        cv.circle(debug_image, origin_pos, 5, (0, 0, 255), 2)
                        cv.circle(debug_image, index_finger_pos, 5, (0, 0, 255), 2)
                        finger_x, finger_y = (origin_pos[0]-index_finger_pos[0])/origin_pos[0], (origin_pos[1]-index_finger_pos[1])/origin_pos[1]
                        data["pos_x"] = -1*finger_x
                        data["pos_y"] = finger_y
                        print(data["pos_x"] , data["pos_y"])
                        ws.send(json.dumps(data))
                        result =  ws.recv()
                    else:
                        origin_pos = index_finger_pos
                        cv.circle(debug_image, origin_pos, 5, (0, 0, 255), 2)
                else:
                    origin_pos = []

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # World座標プロット ###################################################
        # if plot_world_landmark:
        #     if results.multi_hand_world_landmarks is not None:
        #         plot_world_landmarks(
        #             plt,
        #             [r_ax, l_ax],
        #             results.multi_hand_world_landmarks,
        #             results.multi_handedness,
        #         )

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            ws.send(json.dumps(data_disconnect))
            result =  ws.recv()
            ws.close()
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Hand Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            # print(landmark_x, landmark_y)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2, cv.LINE_AA)  # label[0]:一文字目だけ

    return image


def plot_world_landmarks(
    plt,
    ax_list,
    multi_hands_landmarks,
    multi_handedness,
    visibility_th=0.5,
):
    ax_list[0].cla()
    ax_list[0].set_xlim3d(-0.1, 0.1)
    ax_list[0].set_ylim3d(-0.1, 0.1)
    ax_list[0].set_zlim3d(-0.1, 0.1)
    ax_list[1].cla()
    ax_list[1].set_xlim3d(-0.1, 0.1)
    ax_list[1].set_ylim3d(-0.1, 0.1)
    ax_list[1].set_zlim3d(-0.1, 0.1)

    for landmarks, handedness in zip(multi_hands_landmarks, multi_handedness):
        handedness_index = 0
        if handedness.classification[0].label == 'Left':
            handedness_index = 0
        elif handedness.classification[0].label == 'Right':
            handedness_index = 1

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            landmark_point.append(
                [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

        palm_list = [0, 1, 5, 9, 13, 17, 0]
        thumb_list = [1, 2, 3, 4]
        index_finger_list = [5, 6, 7, 8]
        middle_finger_list = [9, 10, 11, 12]
        ring_finger_list = [13, 14, 15, 16]
        pinky_list = [17, 18, 19, 20]

        # 掌
        palm_x, palm_y, palm_z = [], [], []
        for index in palm_list:
            point = landmark_point[index][1]
            palm_x.append(point[0])
            palm_y.append(point[2])
            palm_z.append(point[1] * (-1))

        # 親指
        thumb_x, thumb_y, thumb_z = [], [], []
        for index in thumb_list:
            point = landmark_point[index][1]
            thumb_x.append(point[0])
            thumb_y.append(point[2])
            thumb_z.append(point[1] * (-1))

        # 人差し指
        index_finger_x, index_finger_y, index_finger_z = [], [], []
        for index in index_finger_list:
            point = landmark_point[index][1]
            index_finger_x.append(point[0])
            index_finger_y.append(point[2])
            index_finger_z.append(point[1] * (-1))

        # 中指
        middle_finger_x, middle_finger_y, middle_finger_z = [], [], []
        for index in middle_finger_list:
            point = landmark_point[index][1]
            middle_finger_x.append(point[0])
            middle_finger_y.append(point[2])
            middle_finger_z.append(point[1] * (-1))

        # 薬指
        ring_finger_x, ring_finger_y, ring_finger_z = [], [], []
        for index in ring_finger_list:
            point = landmark_point[index][1]
            ring_finger_x.append(point[0])
            ring_finger_y.append(point[2])
            ring_finger_z.append(point[1] * (-1))

        # 小指
        pinky_x, pinky_y, pinky_z = [], [], []
        for index in pinky_list:
            point = landmark_point[index][1]
            pinky_x.append(point[0])
            pinky_y.append(point[2])
            pinky_z.append(point[1] * (-1))

        ax_list[handedness_index].plot(palm_x, palm_y, palm_z)
        ax_list[handedness_index].plot(thumb_x, thumb_y, thumb_z)
        ax_list[handedness_index].plot(index_finger_x, index_finger_y,
                                       index_finger_z)
        ax_list[handedness_index].plot(middle_finger_x, middle_finger_y,
                                       middle_finger_z)
        ax_list[handedness_index].plot(ring_finger_x, ring_finger_y,
                                       ring_finger_z)
        ax_list[handedness_index].plot(pinky_x, pinky_y, pinky_z)

    plt.pause(.001)

    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    main()
