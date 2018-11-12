#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <fstream>
using namespace std;
using namespace cv;



/*string convertInt(int number)
{
    stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}*/
 
void readFile(string datasetPath,string outputfile )
{int c=0;
    fstream file(outputfile.c_str());
    //file.open(outputfile,'W');
    string path=datasetPath+"/b.txt";
		ifstream shape(path.c_str()); //this will open a.txt that contains the indivisual classes of circular signs
		string sign;
		if(shape.is_open())
		while(getline(shape,sign))	
    	{   //sign=convertInt(sign);

	    	string filepath = datasetPath+"/"+sign+"/s"+sign+".txt";
			ifstream sfile(filepath.c_str()); //this will open the textfile of indivisual classes to read image data
			string imagepath;
			
			//c=0;
			if(sfile.is_open())
	        	{	while(getline(sfile,imagepath))
	      				{   //cout<<"\n"+imagepath;
	      					//  puts(imagepath);
							imagepath=datasetPath+"/"+sign+"/"+imagepath;
							
	            			Mat img = imread(imagepath);
							if(!img.data) { cout<<"\n"<<imagepath<<"--CANT***";continue;}
		   					Mat output;
		   					c++;
							 cout<<"\n"<<imagepath;
	            
					
		    				//Applying gaussian blur to remove any noise
	           			 	//GaussianBlur(img,img,Size(5,5),0);
							 /*Mat grey;
						    cvtColor(img, grey, CV_BGR2GRAY);
							 Mat grad_x, grad_y;
							 Mat abs_grad_x, abs_grad_y;
							 int scale = 1;
							int delta = 0;
							int ddepth = CV_16S;
							Mat grad;

							// Gradient X
							Sobel( grey, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
							/// Gradient Y
							Sobel( grey, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
							convertScaleAbs( grad_x, abs_grad_x );
							convertScaleAbs( grad_y, abs_grad_y );
							addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );*/

							 //Selecting ROI from center of the image to get only the concerened area.
							Size size(24,24);
							resize(img,img,size); 
		   					
		   					int lt=50;
					   		Rect roi;
							int x,y;
							int l,b;
							l=img.cols/4;
							b=img.rows/4;
							x=img.cols/2;
							y=img.rows/2;
							roi.x=x-l;
							roi.y=y-b;
							roi.width=l*2;
							roi.height=b*2;
							img =img(roi);
			
							//cout<<">>>>>>>>>"<<img.rows<<"----"<<img.cols;
							//resize(img,img,size);

					  		//Defining HOG parameters
							HOGDescriptor d( Size(12,12), Size(4,4), Size(2,2), Size(2,2), 4);  
						      vector< float> descriptorsValues;  
						      vector< Point> locations;
						      //computing the HOG features  
	      						d.compute( img, descriptorsValues, Size(0,0), Size(0,0));  
	      						cout<<"\n---"<<descriptorsValues.size()<<"--"<<"**--"<<descriptorsValues[2];
								
	      						//Writing the hog descriptors too the file
								for(int i=0;i<descriptorsValues.size();i++)
									file<<descriptorsValues[i]<<",";

	           			 	//writing the label to file
	            			 file<<sign<<"\n"; 
	       			  	}
		 			sfile.close();
	    		}
	  	//else continue;
	    //cout<<"cant open";
	    }  
	file.close();
	cout<<"\n---"<<c<<endl;
}

int main()
{
    cout<<"Reading the training set......\n";
    readFile("/home/cube26/shruti/shapes/traindata/b","/home/cube26/shruti/shapes/traindata/b/trainingset.txt");
    cout<<"\nReading the test set.........\n";
	///readFile("/home/cube26/shruti/shapes/testdata/b","/home/cube26/shruti/shapes/traindata/b/testset.txt");
    cout<<"\noperation completed\n";
    return 0;
}
