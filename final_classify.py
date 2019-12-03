import cv2
import serial
import keras
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import pickle

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def return_blobs(frame):
    blob_array = []

    return blob_array


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    ser1 = serial.Serial('COM8', 9600)  # <-- fill with the proper com port value

    model_file = 'best_vgg16_25.h5'

    # Gets proper shape and preprocessing function
    input_shape = (224,224)
    preprocess = imagenet_utils.preprocess_input

    model = load_model(model_file)

    print('Model loaded')

    if cap.isOpened():
        while(True) :
            ret_val, img_frame = cap.read() # img_frame is the current frame in the video feed

            # blobs = return_blobs(img_frame)

            img_array = img_to_array(img_frame)
            img_pre = preprocess(img_array)

            final = np.expand_dims(img_pre, axis=0)
            
            probs = model.predict(final)
            if np.argmax(probs) == 2:
                ser1.write('s'.encode())

            # Stop the program on the ESC key
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()


'''
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img_frame = cap.read() # img_frame is the current frame in the video feed
            cv2.imshow("CSI Camera", img_frame) # displays the current video frame

            sign = processFrame(img)
            if(sign == "Stop sign"):
                ser1.write('s'.encode())


            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break'''
