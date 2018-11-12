-----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------
     				 TRAFFIC SIGN DETECTION AND RECOGNITION MODULE FOR AUTONOMOUS VEHICLE  
-----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------

This module is part of a project aiming to create dirverless, self-autonomous vehicle that is fully aware of itself. This vehicle is being conceived at Cube26 Automotive Research branch for the Mahindra Spark The Rise Challenge. The cpp file norm2.cpp is the final code that integrates sign recognition models with sign detection using mser. The code is explained below and there is an additonal readme_training.txt in the folder 'shapes' that explains the dataset and generation of the xml files.


														   AUTHOR - Shruti Agrawal
-----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------


- Global variables: 

ATTRIBUTES: the size of HOGDescriptors used for shape classification
CLASSES: the total number of classes of shape = 4 (i.e a,b,c,d)
CLASSES_A: the total number of signs that are cuircular = 22
CLASSES_B: the total number of signs that are triangular = 27

This code basically uses MSER for detection of signs. It classifies some area as sign, eliminates some of these areas using certain parameters and sends the rest for classification. The models of MLP that are trained are used to classify the signs or classify the roi as a false positive. 

We start from the main function and all the functions are explained as they are called.

Main():

>> declare a matrix 'output' that is of dimensions 640 x 480+100. The 100 is added as that area would be an area added below the image where the detected and recognized signs would be displayed.

>> detectlist is a vector of matrices that will store the list of signs captured and recognized in a single frame.

>>For every image captured from the video resize it to 480x640. Now from this we extract two images:
	GREYimg - grey scale image 
	RBImage - normalized Red/Blue image. This image is obtained by calling the function normalizeBR.

		normalizeBR(): 
		We basically extract the r,b,g value of each pixel. We also calculate the sum of these values. Now we find normR and normB 			which is obtained by dividing the value of that channel by the sum. We multiply by 255 to get value in that range.

>>The grey scale image is used to find white signs in the scene while the normalized image is used to find blue signs. We thus, send each of these images to the function dnr().
	
	dnr(img, out2, detectList): 
	--Here img is the original img captured from the video.
	--out2 is the converted img(GREYimg, RBimg).
	--detectList is the vector of recognized signs.
	
	--First define a vector probable that defines the region which is about 100 pixels less from the bottom of the image. This is done as 		that area mostly consists of the road region and no signs would exist there. Signs would be at a particualr height from the ground. 		From this the image 'out' is obtained.
	--We now apply MSER on the image 'out' and obtain a vector of regions (which is a vector of points) 'mser_points'.
	--We run a loop to get each region of 'mser_points' and form a bounding rectangle 'r'.
	--Now the detections which are obtained are usually very closely touching the sign. Thus, when we send this image for classification, 		there are discrepancies as some important data might be lost. Therefore, we declare a matrix 'r_class' that will hold the image that 		is used for classification. To define 'r_class' we first check that if we add another 20 pixels to the rows and columns of 'r', the 		image should not go out of frame. If it goes out of frame then 'r_class' is set equal to 'r', else 20 pixels are added to rows and 		columns. 
	--After the check is performed we extract 'r_class' from 'img' as we need the original colored image for classification.
	--We also get an image 'roi' by extracting 'r' from 'img'  to check and apply conatraints that will be used to narrow down some 	detections.
	--We calculate the aspect ratio 'aspect', 'area_ratio' which is ratio of area of mser region (total nuber of pixels in area) to area 		of bounding box.
	--Finally we check these constraints on 'roi' The rows and columns are limited to a range of 25-80, aspect ratio should be between 0.5 		and 1.5 and area ratio should be between 0.4 and 1.
	--The MSER area that satisfy these constraints are passed to be classified. The classify() function returns a flag.

		int classify(img, v)
		*Here img is the image 'roi_class' passed from the function dnr.
		*v is the vector detectList which was passed on from main.

'		*First, read the file param.xml which is stored into nnetwork. We will first classify the shape of the sign and label it as 			a,b,c, or d.
		*Resize the image to 24x24 and define a matrix data that will contain the computed 900 HOGDescriptor values.
		*Define a 1-D classOut that will contain the weightage for each of the four class on prediction.
		*Find the class with heighest weightage and set sign corresponding to the index. We thus get sign classified as a,b,c or d.
		*If sign is found to be 'd' or the maximum weightage of the predicted class is less than 0.7, we assume it as false 			positives. We thus return 0 and exit the function.
		*If the sign is not 'c', we need to extract the MLP model of sign obtained (i.e. a or b) and store it in n1.
		*We extract the roi from the center of image so that we distinctly get the required image of the sign. This image is 12x12 in 			size. Matrix 'roi_data' will now store the the computed 400 HOGDescriptor values for this roi.
		*If sign is 'a' we use the matrix Sign_A to store the predicted weights of the class. We find out the class with maximum 			weightage and save the value as maxIndex. Similarly, for 'b' we use matrix Sign_B.
		*If the sign is 'c', the maxIndex is automatically set to '1'.
  		*Now using the value of 'sign' and 'maxIndex' we get a clear idea of what the sign is. We use the getName() function which is 			basically just a collection of switch cases for 'a' and 'b' and consecutive signs. We extract the sample image from the 		display folder for that sign using this information and resize it 50x50. We create a 100x100 matrix that will contain this 			image at the center and use the set_label() function to add text which defines the sign below the image in the matrix. This 			matrix is added to the vector v	(detectList). 
		*Return 1.

	--If flag is 1, we create a rectangle on the original image 'img' to show the detection.

>>We copy the original frame to the matrix 'output'. As the size of the detected matrices is 100x100, only 6 matrices can fit into the bottom of the output matrix. We thus run a loop till the size of detectList and such upto 6 images at a time. If the image number is less than 6, we use it as it y coordinate, else we use image number mod 6 as the y coordinate of the image in the output matrix.

>>Display the 'output' image and frame rate.  

--------------------------------------------------------NOTE----------------------------------------------------------------------------------

1) The parameters for MSER have been chosen after testing thoroughly. However, they can be changed. For a good reference to their meanings you can check the following link: 

2)The constraints for the 'roi' have been set after considerable testing. However, the limit to number of rows and columns can be adjusted based on the requirement of real time data.

3)There is another file 'norm.cpp' in which we have performed all calculations only on the 'RBimg'. The grey scale image is not used. This file is kept in case it needs to be used in future.

4)'norm2.cpp' only has an implementation for the mLP models of the data. In 'norm2.cpp' there is another function, namely, 'classifySign' which is put as a comment and has an implementation for using the random forest models.

5)For displaying the detected signs an implementation of pop up windows, instead of adding those images in the output image, is marked as coomment in case that is more suitable.

6)If there is a requirement to integrate the recognition code with any other detection code, only the function 'classify' is to be copied into the other code and the parameters just need to be changed correspondingly. No other change is required.

-----------------------------------------------------EXECUTABLES------------------------------------------------------------------------------

1)sign_mser.sh to compile and run the code in 'norm2.cpp'
2)norm.sh to compile the code in 'norm.cpp'
