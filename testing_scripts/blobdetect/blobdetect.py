
import cv2
import numpy as np

# Read image
img = cv2.imread('stop.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Params
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(gray)

i = 0
for k in keypoints:
    x, y = int(k.pt[0]), int(k.pt[1])
  
    minx = max(x - 50, 0)
    maxx = min(x + 50, img.shape[1])
    miny = max(y - 50, 0)
    maxy = min(y + 50, img.shape[0])
    box = img[miny:maxy, minx:maxx]

    cv2.imwrite("box{0}.jpg".format(i), box)
    i += 1

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("img.jpg", im_with_keypoints)
