import cv2
import numpy as np

# referencing model path
prototxt_path = "weights/deploy.proto.txt"
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# loading model using readNetFromCaffe from dnn in cv2
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# reading source
video = cv2.VideoCapture("src/video.mp4")

while True:
    # read the desired frame
    ret, frame = video.read()
    # get width and height of the frame
    h, w = frame.shape[:2]

    # preprocess the frame: resize and performs mean subtraction --doute
    blob = cv2.dnn.blobFromImage(frame, 1.0, (int(frame.shape[0]/2), int(frame.shape[1]/2)), (106.13, 115.97, 124.96))

    # set the image into the input of the neural network
    model.setInput(blob)

    # perform inference and get the result
    output = np.squeeze(model.forward())
    font_scale = 1.0

    for i in range(0, output.shape[0]):
        # get the confidence
        confidence = output[i, 2]
        # if confidence is above 50%, then draw the surrounding box
        if confidence > 0.5:
            # get the surrounding box coordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(int)
            # draw the rectangle surrounding the face
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
            # draw text as well
            cv2.putText(frame, f"{confidence * 100:.2f}%", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 0, 0), 2)
    # show the image
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
