# Face-Recognition-OpenCV-

This program demonstrates simple face recognition in C++ using the FaceRecognizer() algorithm provided by the OpenCv library. 

The program uses the at&t database (http://www.cl.cam.ac.uk/research/dtg/attarchive.facedatabase.html) which contains ten different images from 40 different subjects. Of these, 300 images are used to train the model while the other 100 images are stored in a database and are used for testing purposes. Both sets of images are read from a CSV file containing the path of each image as well as a label that identifies each individual. 

	at\s1\1.pgm;0
	at\s1\2.pgm;0
	…
	at\s1\10.pgm;0
	…
	at\s40\1.pgm;39
	at\s40\10.pgm;39
	
The program starts by reading the known and unknown images and their corresponding labels into four different vectors (images are stored into Mat vectors and labels are stored into int vectors). Once this is done 5 random pictures are selected from each database (Mat vector) and put into a new vector for testing purposes (5 images pertain to known subjects while 5 pertain to unknown subjects).  These images are then removed from the original vectors so that the training and testing data do not overlap.  Using the vector containing the known faces and their corresponding labels (295 images) an eigen model is then trained and saved into an external file. The model is saved in order to demonstrate how it can be later loaded into a different program. Once this process is completed the created file is loaded into a new eigen model and a prediction is performed in each of the 10 images contained in the testing vector. A threshold value between 0 and 10 is used to identify whether an input image corresponds to that of a known face or not. If the ‘predict’ function retrieves a class (label) equal to -1, then the face is unknown, otherwise the subject has been recognized. 
