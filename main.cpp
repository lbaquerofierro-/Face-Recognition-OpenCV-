#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//CONSTANTS 

//Path to csv file containint images in the trained database
const string DATABASE_KNOWN = "at_1-30.csv";
//Path to csv file containint images in the set of non-trained images
const string DATABASE_UNKNOWN = "at_31-40.csv";

//This function reads the face database and stores the images and labels into the corresponding vectors
static void readDB(const string& filename, vector<Mat>& images, vector<int>& labels);

int main()
{
	cout << "Face recognition using EIGENFACES" << endl << endl; 

	//Vectors to hold the set of images and lables that will be part of the model
	vector <Mat> images;
	vector <int> labels;

	//Vectors to hold set of images and lables that will not be part of the model
	vector <Mat> unknownImages;
	vector <int> unknownLabels;

	//Read images and labels into Mat objects
	try{
		readDB(DATABASE_KNOWN, images, labels);  //pass file name
		readDB(DATABASE_UNKNOWN, unknownImages, unknownLabels);
	}
	//If the database was not properly read
	catch (Exception& e){
		cout << "Error openning file " << e.msg << endl;
		return 1;
	}

	//The "images" and "unkown" Mat objects must contains 2 picture or more in order for the program to execute correctly 
	if (images.size() <= 1 || unknownImages.size() <= 1){
		cout << "This program requires more than one known and unknow image";
		return 1;
	}

	//Display the initial size of each vector
	cout << "Initial 'images' vector size: " << images.size() << endl;
	cout << "Initial 'unknownImages' vector size: " << unknownImages.size() << endl << endl; 

	//Get sample test images from the "images" and "unknownImages" vectors (for the purpose of this program,random pictures will be used)
	//Remove selected images from the corresponding vectors so that the training and test data don't overlap

	//Create vector with known and unknown images for testing purposes
	vector<Mat>testImages;
	vector<int>testLabels;

	srand(time(NULL));

	for (int i = 0; i < 5; i++){
		int randomNumber0 = rand() % images.size();
		int randomNumber1 = rand() % unknownImages.size();

		//Insert images into test vector
		testImages.push_back(images[randomNumber0]); 
		testLabels.push_back(labels[randomNumber0]); 
		testImages.push_back(unknownImages[randomNumber1]); 
		testLabels.push_back(unknownLabels[randomNumber1]);

		//Delete images from original vector
		images.pop_back();
		labels.pop_back(); 
		unknownImages.pop_back(); 
		unknownLabels.pop_back(); 
	}

	cout << "The number of images that have been added to the testImages vectors is: " << testImages.size() << endl << endl; 


	//Create and train an Eigenface model for face recognition
	cout << "1. TRAIN THE MODEL" << endl; 
	cout << "Training starting..." << endl;

	Ptr<FaceRecognizer> eigenModel = createEigenFaceRecognizer();
	eigenModel->train(images, labels);
	cout << "Training finished..." << endl << endl;

	//Save trained model to external file
	cout << "2. SAVE THE MODEL" << endl; 
	cout << "Saving model to 'eigenfaces.yaml'" << endl << endl; 
	eigenModel->save("eigenfaces.yaml");

	//Create a new model for demonstration purposes
	//This model can be load into different programs 
	cout << "3. LOAD THE SAVED MODEL (NOT NECESSARY)" << endl; 
	Ptr<FaceRecognizer> eigenModel2 = createEigenFaceRecognizer();
	cout << "Loading model from 'eigenfaces.yaml'" << endl << endl; 
	eigenModel2->load("eigenfaces.yaml"); 

	//Recognize faces using the loaded model and the images stored in the test vector 
	cout << "4. TRY TO RECOGNIZE THE TEST IMAGE (FACE)" << endl; 
	cout << "Predict face from model: " << endl; 
	int predictLabel;
	for (int i = 0; i < testImages.size(); i++){ 

		int confidencePredictLabel = -1;
		double confidence = 0.0; 

		//Apply thrshold to the prediction. If the distance to the nearest neighbor (level of confidence) is larger than the threshold (10), the mothod return -1
		double current_threshold = eigenModel->getDouble("threshold"); 
		eigenModel2->set("threshold", 10.0);

		//Try to recognize test images
		predictLabel = eigenModel2->predict(testImages[i]); 

		//confidencePredictLabel and confidence will present different values if a threshold is not used (these variables are not necessary in this function given that a threshold is being used). 
		//In order to see level of confidece, comment out lines 116 and 117, and uncomment line 129. 

		eigenModel2->predict(testImages[i], confidencePredictLabel, confidence);
		
		//Show results
		
		//cout << "        Confidence: " << confidence << endl; 
		cout << "Image " << i << ": ";
		//If the predicted label is equal to -1, the face was not recognized, otherwise it was (1-10)
		if (predictLabel == -1){
			cout << "    Predicted class = " << predictLabel << setw(36) << " -> Unknown face" << endl;
		}
		else{
			cout << "    Predicted class = " << predictLabel << " / Actual label = " << testLabels[i] << setw(3) << " -> (!)Knwon face" << endl; 
		}
	}

	//Terminate the program
	waitKey(); 
	cout << endl; 
	system("pause");
	return 0;
}

static void readDB(const string& filename, vector<Mat>& images, vector<int>& labels){

	ifstream csv_File;
	csv_File.open(filename);
	if (csv_File.is_open()){
		string line, path, img_label;
		char separator = ';';

		int count = 0;
		while (getline(csv_File, line)){
			stringstream lines(line);
			getline(lines, path, separator);
			getline(lines, img_label);

			if (!path.empty() && !img_label.empty()){
				images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
				labels.push_back(atoi(img_label.c_str()));
			}
			count++; 
		}
	}
	else{
		cout << "The input file is not valid";
	}
}