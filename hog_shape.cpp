#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <fstream>
using namespace std;
using namespace cv;
 
void readFile(string datasetPath,string outputfile )
{	
	int c=0;
    fstream file(outputfile.c_str());
    //file.open(outputfile,'W');
    for(char sign = 'a'; sign<='d'; sign++)
    	{
    	string shapefilepath = datasetPath+"/"+sign+"/"+sign+".txt";
		string dir = datasetPath+"/"+sign;
		ifstream shapefile(shapefilepath.c_str());
		string signfilepath;
		if(shapefile.is_open())
			{while(getline(shapefile,signfilepath))
				{
				//c=0;
				//waitKey(500);
				string dir2=dir+"/"+signfilepath;
				signfilepath = dir2+"/s"+signfilepath+".txt";
				ifstream signfile(signfilepath.c_str());
				if(signfile.is_open())
					{ 
						string imagepath;
						while(getline(signfile,imagepath))
							{//cout<<"\nshape: "<<sign<<" -- sign: "<<signfilepath;
							imagepath=dir2+"/"+imagepath; 
		            		Mat img = imread(imagepath);
							if(!img.data) { continue;}
			   				c++;
		    				//Applying gaussian blur to remove any noise
//	           			 	GaussianBlur(img,img,Size(5,5),0);
							Mat grey;
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
							addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
							Size size(24,24);
							resize(img,img,size);
				  		           		
							  HOGDescriptor d( Size(24,24), Size(8,8), Size(4,4), Size(4,4), 9);  
						      vector< float> descriptorsValues;  
						      vector< Point> locations;  
		  						d.compute( img, descriptorsValues, Size(0,0), Size(0,0));  
		   						cout<<"\n---"<<descriptorsValues.size()<<"--"<<"**--"<<descriptorsValues[2];
								for(int i=0;i<descriptorsValues.size();i++)
									file<<descriptorsValues[i]<<",";
		       			 	//writing the label to file
		           			 file<<sign<<"\n";
		           			 cout<<"\nIMAGE: "<<imagepath;
		           			 //waitKey(1000); 
		       			    }
		 			    signfile.close();
		 			    cout<<"\n****TOTAL: "<<c<<"****";
	    		    }
	    		}
	    	shapefile.close();			
	  		}
	  		
		}  
	file.close();
	cout<<"\n$$$$$$$$****TOTAL: "<<c<<"****";
	
}

int main()
{
    cout<<"Reading the training set......\n";
    readFile("/home/cube26/shruti/shapes/traindata","/home/cube26/shruti/shapes/trainingset1.txt");
    cout<<"\nReading the test set.........\n";
//   	readFile("/home/cube26/shruti/shapes/testdata","/home/cube26/shruti/shapes/testset1.txt");
    cout<<"\noperation completed\n";
    return 0;
}

//TRAIN - 2711
//TEST - 250