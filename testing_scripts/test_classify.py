import cv2
import keras
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import pickle

def return_blobs(frame, detector):

    # Convert image to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the detector
    #detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    blobs = detector.detect(gray_img)

    # For every blob, carve out 100x100 array
    ROIs = []
    for blob in blobs:
        x, y = int(blob.pt[0]), int(blob.pt[1])

        xmin = max(x - 50, 0)
        xmax = min(x + 50, frame.shape[1])

        ymin = max(y - 50, 0)
        ymax = min(y + 50, frame.shape[0])

        box = frame[ymin:ymax, xmin:xmax]
        ROIs.append(box)

    return ROIs


def main():
    model_file = 'best_vgg16_25.h5'

    # Gets proper shape and preprocessing function
    input_shape = (224,224)
    preprocess = imagenet_utils.preprocess_input

    model = load_model(model_file)

    print('Model loaded')

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    detector = cv2.SimpleBlobDetector_create(params)

    print('Blob detector set up')

    count = 1

    while count < 5:
        file_name = f'testing_pics/img{count}.jpg'
        image = cv2.imread(file_name)

        ROIs = return_blobs(image, detector)

        print(len(ROIs))
       
        count += 1

        stop_sign = False
        for roi in ROIs:
            img_array = img_to_array(roi)
            img_pre = preprocess(img_array)
            final = np.expand_dims(img_pre, axis=0)
            probs = model.predict(final)[0]
            print(probs)
            if probs[2] >= 0.75:
                stop_sign = True
                #break

        if stop_sign:
            print('yuhhhh')
            #break


if __name__ == "__main__":
    main()
