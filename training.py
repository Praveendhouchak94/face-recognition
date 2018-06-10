import cv2
import numpy as np
import os
from six.moves import cPickle
from sklearn.svm import SVC
import face_recognition

# Accessing the database folder
base = os.path.dirname(os.path.abspath(__file__))
image_database = os.path.join(base, "Image_Database")
# initial name_id field to 0
name_id = 0
names = {}
# initial features and label file to empty
features = []
label = []
q = 0

# finding the files that end with png or jpg and adding then to features after doing some processing
for root, dirs, files in os.walk(image_database):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(root,file)
            name = os.path.basename(root).replace(" ","_").lower()
            #print(names)
            if name not in list(names.keys()):
                names[name] = name_id
                name_id += 1

            #print(names)
            pic = cv2.imread(path)

            face_bounding_boxes = face_recognition.face_locations(pic)
            print(face_bounding_boxes)
            top, right, bottom, left = face_bounding_boxes[0]
            features.append(face_recognition.face_encodings(pic, known_face_locations=face_bounding_boxes)[0])
            label.append(names[name])

print(len(features))
print(len(label))

# training the model on the feature and label
clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
model = clf.fit(np.array(features), np.array(label))


# saving the label into label.pickel file
with open("label.pickel", 'wb') as f:
    cPickle.dump(names, f)

# saving the SVM model to testing
cPickle.dump(model, open('model.yml', 'wb'))