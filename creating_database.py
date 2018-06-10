import cv2
import numpy as np
import os
import face_recognition
from six.moves import cPickle
from sklearn.svm import OneClassSVM


# create database for storing the image
fileDir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists("Image_Database"):
    os.makedirs("Image_Database")
dbTestDir = os.path.join(fileDir, 'Image_Database')

# capture the image from the cam
def image_capture(dir):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed
            img_name = str(dir + "\\" + str(img_counter) + ".jpg")
            cv2.imwrite(img_name, frame)
            # print("{} written!".format(img_name))
            img_counter += 1
            print("Image capture no ", img_counter)
    cam.release()
    cv2.destroyAllWindows()


def main():
    name = input("enter ur name ")

    # create dir in database of username
    dir = os.path.join(dbTestDir, name)
    os.makedirs(dir)

    # calling the image_capture function
    print("click space to save a pic and Escape to close the cam ")
    image_capture(dir)

if __name__ == '__main__':
    main()