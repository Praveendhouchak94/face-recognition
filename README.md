# face-recognition
face recognition using python it also decta


A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. 
 In this i have used various packages such as face_recognition, numpy, sklearn,six, os
 
 
 Inage_Database folder consist of subfolder of known username in face recognition project such as shown below
 
 Image_Database
            |_____anmol
                     |___1.jpg
                     |___2.jpg
                     |___
		     |___
                     |___10.jpg
            |_____praveen
		     |____1.jpg
		     |____2.jpg
		     |____
		     |____10.jpg

first we have to create database of person by using creating_database.py folder will be auto created in Image_Database folder with the name of person

How to run creating_database.py

python3.6 creating_database

After creating database we have to train our model by using SVM(Support Vector Machine)

we have to run training.py file to train the model on features and labels where 
features is 128 length array capture from the input image after detection the face on it
label is index corrosponding the each input user. labels infromation is also stored in label.pickel file
model is save in model.yml file

how to run the training file

python3.6 training.py

we have to test the model on somedate here i have use live data from webcam

To run the testing file

python3.6 testing.py

all the infromation of unknown and known person is stored in result.csv

result.csv file contain the information like: name of person, start time, end time, duration
