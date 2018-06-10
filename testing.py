import cv2
from six.moves import cPickle
import dlib
import time
import datetime
import face_recognition
import numpy as np
import csv
import os


# loading the model file
loaded_model = cPickle.load(open("model.yml", 'rb'))

# loading the labels files
names = {}
with open("label.pickel", "rb") as f:
    names = cPickle.load(f)

# inversing the name dict
names = {x: y for y, x in names.items()}



def tracking_identifing():
    #Open the first webcam

    capture = cv2.VideoCapture(0)

    #Create two opencv named windows
    cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}

    try:
        while True:
            #Retrieve the latest image from the webcam
            rc, frame = capture.read()


            #Check if a key was pressed and if it was Q, then break
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                break

            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest face
            output_frame = frame.copy()
            
            #Increase the framecounter
            frameCounter += 1


            fids_Delete = []
            for fid in faceTrackers.keys():
                tracking_quality = faceTrackers[fid].update(frame)

                #If the tracking quality is good enough, we must delete
                #this tracker

                if tracking_quality < 5:
                    fids_Delete.append(fid)

            for fid in fids_Delete:
                print("Removing fid " + str(fid))
                start_time = time.strftime('%Y-%m-%d T %H:%M:%S',time.localtime(faceNames[fid][1]))
                end_time = time.strftime('%Y-%m-%d T %H:%M:%S', time.localtime(time.time()))

                d = divmod(time.time()-faceNames[fid][1], 86400)  # days
                h = divmod(d[1], 3600)  # hours
                m = divmod(h[1], 60)  # minutes
                s = m[1]  # seconds
                duration = '%d days, %d hours, %d minutes, %d seconds' % (d[0], h[0], m[0], s)
                faceTrackers.pop(fid , None)



                person_data = [{"Name" : faceNames[fid][0], "start_time": start_time, "end_time": end_time,"duration":duration }]

                with open('result.csv', 'a') as csvFile:
                    fields = ['Name', 'start_time', "end_time", "duration"]
                    writer = csv.DictWriter(csvFile, fieldnames=fields)
                    writer.writerows(person_data)

                csvFile.close()

                faceNames.pop(fid, None)

            #Every 5 frames we will check for new face
            if (frameCounter % 10) == 0:
                # findind the faces location in frame
                faces = face_recognition.face_locations(frame)
                for face in faces:
                    top, right, bottom, left = face
                    x = int(left)
                    y = int(top)
                    w = int(right-left)
                    h = int(bottom-top)

                    #centerpoint of face
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchedFid = None


                    #centerpoint of the face is within the box of a tracker
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())
                        #calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        #check if the centerpoint of the face is within the
                        #rectangleof a tracker region. Also, the centerpoint
                        #of the tracker region must be within the region
                        #detected as a face. If both of these conditions hold
                        #we have a match
                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and ( t_y <= y_bar   <= (t_y + t_h)) and
                             ( x   <= t_x_bar <= (x   + w  )) and ( y   <= t_y_bar <= (y   + h  )) ):
                            matchedFid = fid


                    #If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                        print("Creating new tracker " + str(currentFaceID))

                        #Create and store the tracker
                        tracker = dlib.correlation_tracker()

                        # start tracking the new face in input image
                        tracker.start_track(frame, dlib.rectangle(x, y, x+w, y+h))
                        # encoding the the face in 128 length vector
                        faces_encodings = face_recognition.face_encodings(frame, known_face_locations=[face])[0]
                        # finding the face label with the model
                        pred = loaded_model.predict(np.array(faces_encodings).reshape(1, -1))
                        print(pred)
                        # testing for unknown
                        folder_name = str("Image_Database" + "\\" + names[pred[0]])
                        base = os.path.dirname(os.path.abspath(__file__))
                        folder_path_complete = os.path.join(base, folder_name)

                        test_pic = cv2.imread(str(folder_path_complete + "\\" + "1.jpg"))
                        test_faces = face_recognition.face_locations(test_pic)
                        faces_encodings_ = face_recognition.face_encodings(test_pic, known_face_locations=test_faces)[0]
                        # compare the the face to pred the unknown
                        results = face_recognition.compare_faces([faces_encodings], faces_encodings_)
                        print(results)
                        if results[0] == True:
                            faceTrackers[currentFaceID] = tracker
                            faceNames[currentFaceID] = (names[pred[0]], time.time())
                        else:
                            faceTrackers[currentFaceID] = tracker
                            faceNames[currentFaceID] = ("unknown", time.time())
                            print(faceNames)

                        # Increase the currentFaceID counter
                        currentFaceID += 1

            # Now loop over all the trackers to label the faces
            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(output_frame, (t_x, t_y), (t_x + t_w , t_y + t_h), (222, 222, 222), 1)


                if fid in faceNames.keys():
                    cv2.putText(output_frame, str(faceNames[fid][0]),
                                (int(t_x+5), int(t_y-15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)
                    cv2.putText(output_frame, datetime.datetime.now().strftime("%H:%M:%S"),
                                (int(t_x+5), int(t_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)





            # displaying the result in output_frame
            output = cv2.resize(output_frame, (800,800))

            #Finally, we want to show the images on the screen
            cv2.imshow("input", frame)
            cv2.imshow("output", output)

    except KeyboardInterrupt as e:
        pass

    #Destroy all the window
    capture.release()
    cv2.destroyAllWindows()

def main():
    tracking_identifing()



if __name__ == "__main__":
    main()
