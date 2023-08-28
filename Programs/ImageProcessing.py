#Image Processing

import cv2
from google.colab.patches import cv2_imshow
img = cv2.imread('im1.jfif', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2_imshow(sobelx)
cv2_imshow(sobely)
cv2_imshow(sobelxy)
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
cv2_imshow(edges)
from PIL import Image as im
edge_img = im.fromarray(edges)
edge_img.save('/content/drive/MyDrive/DATASETS/im2.jfif') #Make sure to tailor this path!

"""
#Final Output Only
#Face Detection
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('im1.jfif')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# Draw bounding boxes around the detected faces
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Display the image with bounding boxes
plt.figure(figsize=(20, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Applying Canny edge detection
edges = cv2.Canny(image, 50, 150)

# Display original image and edge image side by side
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title("Edge Image")
plt.imshow(edges, cmap='gray')
plt.show()

"""
