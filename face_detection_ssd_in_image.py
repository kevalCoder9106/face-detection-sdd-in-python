import cv2
import numpy as np

# referencing model path
prototxt_path = "weights/deploy.proto.txt"  # file with text description of the network architecture
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"  # file with learned network

# loading model using readNetFromCaffe
model = cv2.dnn.readNetFromCaffe(prototxt_path,
                                 model_path)  # returns net object (net objects are class allows to create and manipulate comprehensive artificial neural networks.

# read the image
image = cv2.imread("src/photo3.jpg")
# get width and height of the image
h, w = image.shape[:2]

# resizing image using blobFromImage(image,scale,(x,y),bgr values to be subtracted from original values)
processed_image = cv2.dnn.blobFromImage(image, 1.0, (1920, 1080), (106.13, 115.97, 124.96))  # returns a cropped image

# set processed image as input to the model
model.setInput(processed_image)

# run the model using model.forward and use np.squeeze to store the return values at one place
output = np.squeeze(model.forward())

font_scale = 1.0

for i in range(0, output.shape[0]):  # output.shape
    # get the confidence
    confidence = output[i, 2]
    # limit the confidence
    if confidence > 0.5:
        # location of detection
        box = output[i, 3:7] * np.array([w, h, w, h])
        # convert to integers
        start_x, start_y, end_x, end_y = box.astype(np.int)
        # draw the rectangle on faces
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
        # draw percentage
        cv2.putText(image, f"{confidence * 100:.2f}%", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 0, 0), 2)
# show the image
cv2.imshow("image", image)
cv2.waitKey(0)

cv2.imwrite("photo4_detected.png", image)
