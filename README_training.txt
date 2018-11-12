-----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------
     				 TRAINING MLP RECOGNITION MODULE FOR AUTONOMOUS VEHICLE  
-----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------

This module is a collection of all the data and the files that are used to extract the features from the images and train models of MultiLayer Perceptron. The following is a description of how the dataset is arranged and the code present in each of the files.

														   AUTHOR - Shruti Agrawal
-----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------


Shapes folder contains two important subfolders : Traindata and Testdata. The data in these folders is organized in the same way as follows:
Traindata/Testdata contains 4 folders:
- 'a' which contains circular signs and has subfolders from 1 - 22
- 'b' which contains triangular signs and has subfolders from 1 - 27
- 'c' which contains octagonal sign, namely, the stop sign and thus has only one folder '1'
- 'd' which contains the false positives and thus has one folder '1'

Each of these folders a,b,c,d have a text file, for example, a.txt that contain a list having values 1 through 22.

Each folder containing images say 1 has a text file s1 (s followed by the number of the folder) that contains a list of the images. This can be done by opening that folder on the command prompt and giving the command such as ls -1 > s1.txt

There are 3 MLP models: 

1) The following files are stored in the folder 'shapes':
   
   "hog_shape" for obtaining the HOG descriptor values for finding the shape of the object, i.e 900 values
   
	-We run a loop from a to d
	-Extract the text file (e.g a.txt) and name as shapefile
	-From shape file extract the folder name and access the txt file within the folder (e.g s1.txt) as the signfile
	-From this extract the imagepath
	-We increment a variable c everytime a new image is extracted.
	-Apply sobel filter to the image
	-Resize image to 24x24
	-Declare a HOGDescriptor that will compute 900 HOG values for the image and save each value into file followed by a comma, and the 			last value is followed by the class 'a','b','c','d' that it it belongs to. This file is finally saved as 			trainingset.txt/testset.txt
	-Display the value of c as the total number of images in training/ testing set. 

   "TrainNetwork_shape" that will be used to train an MLP model based on the hog values extracted using hog_shape
	-We extract the values saved in trainingset.txt and save them in a matrix training_set
	-We also extract the values saved in testset.txt and save them in a matrix test_set
	-The following important variables need to be updated in case new samples are added or changes are made:
		TRAININg_SAMPLES: number of training images after we get total number of images displayed after running hog_shape
		TEST_SAMPLES: number of test images after we get total number of images displayed after running hog_shape
		ATTRIBUTES: Number of HOGDescriptor values extracted (currently 900)
		CLASSES: Total number of shapes/classes used. (i.e a,b,c,d)
	-The matrix training_set_classification/test_set_classification has rows = number of training samples/ test samples and columns = the 			classes. It basically stores the value '1' in the cell that represents the class of that sample. 
	-A MLP is declared with the default parameters. Three layers are defined. 1st layer = input layer that contains the HOG values, 2nd 			layer is the hidden layer conatining 100 nodes, and 3rd layer = output layer that defines the output classes.
	-The training_set_classification matrix along with the training_set matrix is fed into the MLP.
-	-After the training is complete the model is stored as param.xml in the 'shapes' folder and the model is named "FindShape".
	-Now we test the accuracy of the model using a testing_set. We run a loop for iterations = number of test samples. For each sample we 			predict the class using the model of MLP and store this data in a 1-D array "classificationResult". This array now contains 			the weightage of each class for that particualr sample. We find the class having maximum weightage and select it as the 		predicted class. We now	check if the class specified in the test_set_classification matrix matches with the predicted class. 			If correct increment correct_class, else increment wrong_class. Using this we find the accuracy.

2) The following files are stored in the folder /shapes/traindata/a or /b
   
   "traindata_a" and "traindata_b" which are used to obtain the HOG descritpor values for the roi that distinctly specifies the sign from the 	 image. "traindata_a" is for circular signs and "traindata_b" is for triangular signs.
	-A file "shape" is opened that contains the path of a.txt/b.txt.
	-We extract the name of the folder for the file 'shape' and then extract the text file of that folder (e.g s1.txt).
	-We extract an image from the corresponding folder. 
	-Resize image to 24x24. Now find the center of the image. From this center crop the image such that an image of half the dimensions is 			obtained from the center.
	-Declare a HOGDescriptor that will compute 400 HOG values for the image and save each value into file followed by a comma, and the 			last value is followed by the folder (e.g 1,2,3...) that it it belongs to. This file is finally saved as 			trainingset.txt/testset.txt
	--Display the value of c as the total number of images in training/ testing set.
	
   "TrainNetwork_a" and "TrainNetwork_b" that will be used to train an MLP model based on the hog values extracted using traindata_a /       	traindata_b
	-This code follows the exact same algorithm. The difference is seen only in the value of four important variables: TRAINING_SAMPLES, 		TEST_SAMPLES, ATTRIBUTES (currently 400), and CLASSES. These values will be set based on which class a, or b this image belongs to.
	-Also the hidden layer in this model will have only 20 nodes.
	-The the models are stored as a_param.xml / b_param.xml in the respective folder a or b and the model is named "SignDetect" for both.

-----------------------------------------------------------NOTE--------------------------------------------------------------------------------

1) The hidden layers for each model were chosen after carefully examining the results for different values. Increasing or decreasing the nodes 	  of the hidden layers decreased the accuracy of the model, and also, increasing the number of nodes makes the model slower.

2)There is an additional folder called "display" in 'shapes' which contains sample images of each sign. The naming convention used is: class  	(a,b,c,d) followed by the folder(1,2,3...). So if new folder containing a new sign altogether is added, a sample image with the specified
  naming convention is also to be added in this folder.

3)The accuracy of the MLP model "TrainNetwork_shape" is 98.911% for all folders of class a,b,c,d which is currently a total of 50 folders.
  The accuracy of the MLP model "TrainNetwork_a" is 89.166% for data in the 22 folders.
  The accuracy of the MLP model "TrainNetwork_b" is 92.079% for data in the 27 folders.

4)I also have implemented a random forest model using the same data set, which has the same storage and naming conventions as the MLP model.  	The xml files are random.xml, a_random.xml and b_random.xml respectively.
  The accuracy of the RF model "random_shape" is 94.918% for all folders of class a,b,c,d which is currently a total of 50 folders.
  The accuracy of the RF model "random_forest_a" is 79.397% for data in the 22 folders.
  The accuracy of the RF model "random_forest_b" is 73.397% for data in the 27 folders.
  Based on these results I have decided to continue with MLP. In real time also MLP model shows better results.

-----------------------------------------------------------EXECUTABLES-------------------------------------------------------------------------

In folder shapes:
1)hog_shape.sh
2)trainNetwork_shape.sh
3)random_shape.sh

In folder a and b:
1) traindata_a.sh / traindata_b.sh
2) TrainNetwork_a.sh / TrainNetwork_b.sh
2) random_forest_a.sh / random_forest_b.sh
