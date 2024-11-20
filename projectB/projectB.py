import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.python.util.numpy_compat import np_array


def normalized(width):
    hand_data=np_array([[lm.x /width, lm.y, lm.z] for lm in hand_landmark.landmark])
    wrist=hand_data[0]
    hand_data -= wrist


    return  hand_data.flatten()  # Return as a 1D array

model_gesture=tf.keras.models.load_model("models/model_rev1.h5")
model_hand=mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

gesture_labels = {
    1: "dislike",
    3: "fist",
    5: "reset",
    6: "one",
    8: "Yeah",
    9: "rock",
}

input_video=cv2.VideoCapture("video/hand.avi")

width,height=int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
size=(width,height)
fps=input_video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter('result.avi', fourcc, fps, size)

gesture_log=[]

with model_hand.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        frame_out = frame.copy()

        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(frame, hand_landmark, model_hand.HAND_CONNECTIONS)
                normalized_joint=normalized(width)
                hand_data=[1]+normalized_joint.tolist()
                hand_data=np.array(hand_data)
                hand_data=np.expand_dims(hand_data, axis=0)
                # print(hand_data.shape)
                predict=model_gesture.predict(hand_data)
                confidence = np.max(predict)
                predict_class=np.argmax(predict)
                print(confidence,predict_class)
                gesture=gesture_labels.get(predict_class,"Unknown")
                if confidence>0.93:
                    if gesture=="reset":
                        gesture_log=[]
                        cv2.putText(frame, gesture, (50, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        cv2.putText(frame_out, gesture, (50, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    elif gesture!="Unknown":
                        if gesture_log:
                            if gesture not in gesture_log:
                                gesture_log.append(gesture)
                                print(gesture_log)
                                cv2.putText(frame, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                                cv2.putText(frame_out, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            else:
                                print(gesture_log)
                                cv2.putText(frame, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                                cv2.putText(frame_out, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 2)
                        else:
                            gesture_log.append(gesture)
                            print(gesture_log)
                            cv2.putText(frame, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                            cv2.putText(frame_out, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        cv2.putText(frame_out, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    cv2.putText(frame_out, ",".join(map(str,gesture_log)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)





        cv2.imshow('Video', frame)
        output.write(frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


input_video.release()
cv2.destroyAllWindows()







