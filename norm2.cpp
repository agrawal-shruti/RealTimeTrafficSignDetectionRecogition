#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>
#include <fstream>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;
#define ATTRIBUTES 900     //Number of hog values per sample for shape recognition.
#define CLASSES 4          //Number of distinct labels for shape recognition.
#define CLASSES_A 22	   //Number of distict labels for subclasses of class A
#define CLASSES_B 27	   //Number of distict labels for subclasses of class B
int num=0;

//Function add label to the window for sign reconition. 
void set_label(cv::Mat& im, cv::Rect r, const std::string label)

{
	//cout<<'C';
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Point pt(r.x + (r.width-text.width)/2, r.y + (r.height+text.height)/2);

    cv::rectangle(
        im, 
        pt + cv::Point(0, baseline), 
        pt + cv::Point(text.width, -text.height), 
        CV_RGB(102,0,102), CV_FILLED
    );

    cv::putText(im, label, pt, fontface, scale, CV_RGB(255,255,0), thickness, 8);
}

//Function to assign the name of the sign as per the predicted class.
String getName(char sign, int type)
{

//cout<<'B';
	String name;
if(sign=='a') //Circle
    switch(type)
        {
            case 1 : name = "Comp. Turn Right Ahead"; break;
            case 2 : name = "Speed Limit: 50"; break;
            case 3 : name = "Right Turn Prohib."; break;
            case 4 : name = "Left Turn Prohib."; break;
            case 5 : name = "No Stopping"; break;
            case 6 : name = "U-Turn Prohib."; break;
            case 7 : name = "Comp. Turn Left Ahead"; break;
            case 8 : name = "Prdestrians Prohib."; break;
            case 9 : name = "No Parking"; break;
            case 10 : name = "Horn Prohib."; break;
            case 11 : name = "No entry"; break;
            case 12 : name = "Cycle Prohib."; break;
            case 13 : name = "All motor vehicles Prohib."; break;
            case 14 : name = "Comp. Ahead Only"; break;
            case 15 : name = "Overtaking Prohib."; break;
            case 16 : name = "Comp. Left"; break;
            case 17 : name = "Comp. Cycle Track"; break;
            case 18 : name = "Comp. Turn Left"; break;
            case 19 : name = "Comp. Turn Right"; break;
            case 20 : name = "Speed Limit: 30"; break;
            case 21 : name = "Speed Limit: 40"; break;
            case 22 : name = "Speed Limit: 20"; break;
          
        }
    else if (sign == 'b') //Triangle
    switch(type)
        {
            case 1 : name = "Narrow Road Ahead"; break;
            case 2 : name = "Pedestrian Crossing"; break;
            case 3 : name = "Men At Work"; break;
            case 4 : name = "Falling Rocks"; break;
            case 5 : name = "Give Way"; break;
            case 6 : name = "Side Road Right"; break;
            case 7 : name = "Side Road Left"; break;
            case 8 : name = "T-Intersection"; break;
            case 9 : name = "Major Road Ahead"; break;
            case 10 : name = "Gap in Median"; break;
            case 11 : name = "Cattle"; break;
            case 12 : name = "Gaurded Level Crossing"; break;
            case 13 : name = "Cycle Crossing"; break;
            case 14 : name = "Ungaurded Level Crossing"; break;
            case 15 : name = "Left Reverse Bend"; break;
            case 16 : name = "Right Reverse Bend"; break;
            case 17 : name = "Cross Road"; break;
            case 18 : name = "Steep Descent"; break;
            case 19 : name = "Steep Ascent"; break;
            case 20 : name = "School Ahead"; break;
            case 21 : name = "Roundabout"; break;
            case 22 : name = "Ferry"; break;
            case 23 : name = "Dangerous Dip"; break;
            case 24 : name = "Staggard Intersection"; break;
            case 25 : name = "Rough Road"; break;
            case 26 : name = "Y-Intersection"; break;
        }
else if (sign == 'c') //Ocatagon
    name = "Stop";
return name;        
}

//Function to extract image features and classify it using MLP. It returns a flag theat is used to specify if it is a false positive or not. Also stores 
//predicted sign in a vector of matrices. 
float classify( Mat &img, std::vector<Mat>& v )
{
    
    //read the model from the XML file and create the neural network.
    CvANN_MLP nnetwork;
    CvFileStorage* storage = cvOpenFileStorage( "/home/cube26/shruti/shapes/param.xml", 0, CV_STORAGE_READ );
    CvFileNode *n = cvGetFileNodeByName(storage,0,"FindShape");

   // cout<<"\nANN model extracted";
    nnetwork.read(storage,n);
    cvReleaseFileStorage(&storage);
 
    // ...Generate cv::Mat data(1,ATTRIBUTES,CV_32S) which will contain the hog data for the digit to be recognized
  
    Mat data(1,ATTRIBUTES,CV_32F);
    Size size(24,24);
    resize(img,img,size);

    //cout<<"A";
    //compute hog values for recognising the shape from a 24x24 image.55
    HOGDescriptor d( Size(24,24), Size(8,8), Size(4,4), Size(4,4), 9);  
    vector< float> descriptorsValues;  
    vector< Point> locations;  
    d.compute( img, descriptorsValues, Size(0,0), Size(0,0));
    //cout<<"---"<<descriptorsValues.size(); 
    for(int i=0;i<descriptorsValues.size();i++)
    {
        data.at<float>(0,i) = descriptorsValues[i]; 
    }
    //cout<<descriptorsValues.size();
    
    char sign='a';
    int maxIndex = 0;
    //create a matrix to store the predictted weightage of each class.
    cv::Mat classOut(1,CLASSES,CV_32F);
    
    //prediction
    nnetwork.predict(data, classOut);
    float value=0.0f;
    //Obtain the class with the heighest value. This shall be the predicted class.  
    float maxValue=classOut.at<float>(0,0);
    int index;
    for( index=0;index<CLASSES;index++)
    {   value = classOut.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex = index;
                if (index==0)
                    sign='a'; 
                    else if(index== 1)
                    sign ='b';
                    else if(index==2)
                    sign = 'c';
                    else if(index == 3) 
                    sign = 'd';
            }
    }
    //If the predicted class is 'd' then it is a false positive. This must return 0.
    if ((sign =='d') || (maxValue < 0.7))
        {   
            cout<<"\n**NOPE";
            return 0;
        }
   /* for(int i=0;i<CLASSES;i++)
        cout<<"\n $$^"<<i+1<<": "<<classOut.at<float>(0,i);*/
    //cout<<"\n SIGN : "<<sign;
    
    std::ostringstream oss;
    //For classes 'a' and 'b'
    if(sign!='c')
    {
    //We extract the respective xml file which contains the model of the ANN   
    oss << "/home/cube26/shruti/shapes/traindata/"<<sign<<"/"<<sign<<"_param.xml";
    string path = oss.str();
   // cout<<"\n"<<path;
    CvANN_MLP ann2;
    CvFileStorage* storage2 = cvOpenFileStorage( path.c_str(), 0, CV_STORAGE_READ );
    CvFileNode *n1 = cvGetFileNodeByName(storage2,0,"SignDetect");
    ann2.read(storage2,n1);
    cvReleaseFileStorage(&storage2);
    //We crop the image from the centre to obtain a 12x12 image that only contains the sign.
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
    //This matrix shall contain the hog values for the roi sample obtained above.
    Mat roi_data(1,400,CV_32F);
   

    //Size size(24,24);
    //resize(img,img,size);
    //We obtain 400 hog values for the roi image
    HOGDescriptor d1( Size(12,12), Size(4,4), Size(2,2), Size(2,2), 4);  
    vector< float> descriptorsValues1;  
    vector< Point> locations1;  
    d1.compute( img, descriptorsValues1, Size(0,0), Size(0,0));
        for(int i=0;i<descriptorsValues1.size();i++)
    {
        roi_data.at<float>(0,i) = descriptorsValues1[i]; 
    }

    cv::Mat sign_A(1,CLASSES_A,CV_32F);
    cv::Mat sign_B(1,CLASSES_B,CV_32F);
//If the shape was classified as 'a' (circle) we find wightage for the 22 subclasses
if(sign=='a')
 {
    ann2.predict(roi_data, sign_A);
    value=0.0f;
     maxIndex = 1; 
     maxValue=sign_A.at<float>(0,0);

    for( index=0;index<CLASSES_A;index++)
    {   value = sign_A.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex = index+1;
            }    
    }


    /*for(int i=0;i<CLASSES_A;i++)
        cout<<"\n--"<<i+1<<"**"<<sign_A.at<float>(0,i);*/
     //cout<<"\n--"<<maxIndex<<" : "<<maxValue;
}
//If the shape was classified as 'b' (triangle) we find wightage for the 27 subclasses
    else if(sign == 'b')
 {  ann2.predict(roi_data, sign_B);
    value=0.0f;
     maxIndex = 1;
     maxValue=sign_B.at<float>(0,0);
          for( index=0;index<CLASSES_B;index++)
    {   value = sign_B.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex = index+1;
            }    
    }
     /*  for(int i=0;i<CLASSES_B;i++)
        cout<<"\n--"<<i+1<<"**"<<sign_B.at<float>(0,i);*/
    //cout<<"\n--"<<maxIndex<<" : "<<maxValue;
    //maxIndex is the predicted class.
 }
 
}
else //if the shape is 'c'
    maxIndex =1;
// if(maxValue>0.9)
//{
	String SignName = getName(sign, maxIndex); //finds the name of the obtained sign.
//cout<<endl<<SignName<<endl;
    oss.str("");
    Mat im2;
   //extracts sample image of the sample sign.
 oss << "/home/cube26/shruti/shapes/display/"<<sign<<maxIndex<<".jpeg";
    string dp = oss.str();
   // cout<<"\n*** "<<dp;
    im2 = imread(dp.c_str());
 
    Size outsize(50,50);
    resize(im2,im2,outsize);
   /* imshow("yo",im2);
    waitKey(0);*/

    //create a section in the original image where the image and label is to be displayed.
    Mat  output = Mat::zeros(im2.rows+50, im2.cols+50, CV_8UC3);
    im2.copyTo(output(Rect(25, 0, im2.cols, im2.rows)));
    //cout<<"\n--output: "<<output.rows<<"--"<<output.cols<<"\n";

    Rect textArea;
    textArea.x = 0;
    textArea.y = 50;
    textArea.width = 100;
    textArea.height= 50;
    set_label(output, textArea, SignName );
    /*imshow("yo",output);
    waitKey(10);*/
    v.push_back(output);
    return 1;
//}
//else return 0;

}
/*int classifySign( Mat &img, std::vector<Mat>& v  )
{
    //read the model from the XML file and create the neural network.
   // CvANN_MLP nnetwork;
     CvRTrees* rtree11 = new CvRTrees;
    CvFileStorage* storage = cvOpenFileStorage( "/home/cube26/shruti/shapes/random.xml", 0, CV_STORAGE_READ );
    CvFileNode *n = cvGetFileNodeByName(storage,0,"FindShape");
    rtree11->read(storage,n);
    cvReleaseFileStorage(&storage);
 
    //your code here
    // ...Generate cv::Mat data(1,ATTRIBUTES,CV_32S) which will contain the pixel
    // ... data for the digit to be recognized
    // ...
   // Mat im2 = img.clone();
    Mat data(1,ATTRIBUTES,CV_32F);
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
 //   readFile(img, data);
    Size size(24,24);
    resize(img,img,size);
    resize(grey,grey,size);
    //cout<<"A";
    HOGDescriptor d( Size(24,24), Size(8,8), Size(4,4), Size(4,4), 9);  
    vector< float> descriptorsValues;  
    vector< Point> locations;  
    d.compute( grey, descriptorsValues, Size(0,0), Size(0,0));
    //cout<<"---"<<descriptorsValues.size(); 
    for(int i=0;i<descriptorsValues.size();i++)
    {
        data.at<float>(0,i) = descriptorsValues[i]; 
    }
    //cout<<descriptorsValues.size();
     double result;
    char sign='a';
    /*int maxIndex = 0;
    cv::Mat classOut(1,CLASSES,CV_32F);
    //prediction
    nnetwork.predict(data, classOut);
    float value=0.0f;
      
    float maxValue=classOut.at<float>(0,0);
    int index;
    for( index=0;index<CLASSES;index++)
    {   value = classOut.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex = index;*/
              /*  result = rtree11->predict(data, Mat());
                if (result==0)
                    sign='a'; 
                    else if(result==1)
                    sign ='b';
                    else if(result==2)
                    sign = 'c';
                    else if (result==3)
                    sign = 'd';
   /*         }
    }*/
   

    /* for(int i=0;i<CLASSES;i++)
        cout<<"\n $$^"<<i+1<<": "<<classOut.at<float>(0,i)<<"::::"<<i;*/
    /*cout<<"\n SIGN : "<<sign;
     std::ostringstream oss;

      if(sign=='d' )
    {
        cout<<"\n**NOPE";
        return 0;
    }
   
    if(sign!='c')
    {
       
    oss << "/home/cube26/shruti/shapes/traindata/"<<sign<<"/"<<sign<<"_random.xml";
    string path = oss.str();
   // cout<<"\n"<<path;
    CvRTrees* rtree = new CvRTrees;
    CvFileStorage* storage2 = cvOpenFileStorage( path.c_str(), 0, CV_STORAGE_READ );
    CvFileNode *n1 = cvGetFileNodeByName(storage2,0,"SignDetect");
    rtree->read(storage2,n1);
	cvReleaseFileStorage(&storage2);

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
    Mat roi_data(1,6400,CV_32F);
 //   readFile(img, data);
   

    //Size size(24,24);
    //resize(img,img,size);

    HOGDescriptor d1( Size(12,12), Size(8,8), Size(1,1), Size(1,1), 4);  
    vector< float> descriptorsValues1;  
    vector< Point> locations1;  
    d1.compute( img, descriptorsValues1, Size(0,0), Size(0,0));
        for(int i=0;i<descriptorsValues1.size();i++)
    {
        roi_data.at<float>(0,i) = descriptorsValues1[i]; 
    }
     //sign='a';
    //int maxIndex = 0;
    //if(sign == 'a')
    cv::Mat sign_A(1,CLASSES_A,CV_32F);
    //else if(sign == 'b')
    cv::Mat sign_B(1,CLASSES_B,CV_32F);

if(sign=='a')
 {      
       result = rtree->predict(roi_data, Mat());

    /*ann2.predict(roi_data, sign_A);
    value=0.0f;
     maxIndex = 1; 
     maxValue=sign_A.at<float>(0,0);

    for( index=0;index<CLASSES_A;index++)
    {   value = sign_A.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex = index+1;
            }    
    }

    for(int i=0;i<CLASSES_A;i++)
        cout<<"\n--"<<<<"**"<<sign_A.at<float>(0,i);*/
    /*    cout<<"\n---------SUBCLASS: "<<result;
}
    else if(sign == 'b')
 {   //double result;
       result = rtree->predict(roi_data, Mat());/*ann2.predict(roi_data, sign_B);
    /*value=0.0f;
     maxIndex = 1;
     maxValue=sign_B.at<float>(0,0);
          for( index=0;index<CLASSES_B;index++)
    {   value = sign_B.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex = index+1;
            }    
    }
       for(int i=0;i<CLASSES_B;i++)
        cout<<"\n--"<<i+1<<"**"<<sign_B.at<float>(0,i);*/
   /*  cout<<"\n---------SUBCLASS: "<<result;
    //maxIndex is the predicted class.
 }
 
}
else 
    result =1;
 
cout<<"\n--The class is: "<<sign<<"--\n"<<" and Sign is: "<<result;
String SignName = getName(sign, result);
cout<<endl<<SignName<<endl;
    oss.str("");
oss << "/home/cube26/shruti/shapes/display/"<<sign<<result<<".jpeg";
    string dp = oss.str();
    cout<<"\n*** "<<dp;
    Mat im2 = imread(dp.c_str());
    Size outsize(100,100);
    resize(im2,im2,outsize);
   /* imshow("yo",im2);
    waitKey(0);*/

 /* Mat  output = Mat::zeros(im2.rows+50, im2.cols+100, CV_8UC3);
    im2.copyTo(output(Rect(50, 0, im2.cols, im2.rows)));
    //cout<<"\n--output: "<<output.rows<<"--"<<output.cols<<"\n";

    Rect textArea;
    textArea.x = 0;
    textArea.y = im2.rows;
    textArea.width = output.cols;
    textArea.height= 50;
    set_label(output, textArea, SignName );
   /* imshow("yo",output);
    waitKey(0);*/
    /*  v.push_back(output);
    return 1;
}*/


Mat normalizeBR(Mat &img)
{
	Vec3b color;
    Mat out = Mat::zeros( img.size(), CV_8UC1 );
    int i =0;
    for(int x=0;x<img.rows;x++)
    {  
        for(int y=0;y<img.cols;y++)
        {
			Vec3b color = img.at<Vec3b>(x,y);
            int b = (int)color[0];
            int g = (int)color[1];
            int r = (int)color[2];
            int sum = b+g+r;
            uchar normR = ((double)r/sum)*255;
            uchar normB = ((double)b/sum)*255;
            uchar normG = ((double)g/sum)*255;
            if(normR>normB /*&& normR>normG*/)
                out.at<uchar>(x,y)=normR;
            else if (normB>normR/* && normB>normG*/)
                out.at<uchar>(x,y)=normB; 
            //else out.at<uchar>(x,y)=normG;

        }
 
    }
    return out;
   
}



int main()
{
    VideoCapture cap("/home/cube26/shruti/SignDetection/BEST1.avi");
    double t0 = getTickCount();
    Mat img;
    cap>>img;
     Mat output = Mat::zeros(480+100, 640, CV_8UC3);
        std::vector<Mat> detectList;

do
{	   //cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	cap>>img;
	detectList.clear();
   Size s(640,480);
    resize(img,img,s,0,0,INTER_CUBIC);
    
    //Mat out2 = normalizeBR(img);
    Mat out2;
    cvtColor(img,out2,CV_BGR2GRAY);
    //Mat out=img.clone();
    Rect probable;      //Setting the limit of the horizon
   /* probable.x=out2.x;
    probable.y=out2.y;*/
    probable.width=out2.cols;
    probable.height=out2.rows-100;
    Mat out = out2(probable);

    imshow("yo",out2);
    imshow("out",out);
    MSER mser;
       FeatureDetector* detector = new MSER();
       //DescriptorExtractor* extractor = new MSER();
				vector<KeyPoint> keypoints2;
		//		Mat descriptors2; 
				Mat out_1;
				//detecting the keypoints and computing the descriptors
				Mat img_keypoints_1;
				detector->detect(out, keypoints2);
				drawKeypoints( out, keypoints2, out_1, Scalar(255,0,0), DrawMatchesFlags::DEFAULT );
		//		extractor->compute(out, keypoints2, descriptors2); 

      vector<vector<Point> > mser_points;
      mser(out, mser_points, Mat());
      //ncout<<"\n---"<<mser_points.size()<<"\n";

       for (int i = 0; i < mser_points.size(); i++)
    {  Rect r = boundingRect(mser_points[i]);

    	Rect r_class;
    
    	//adding padding for classification
    	if(r.x-10>0 && r.y-10>0 && r.x<img.cols-(r.width+20) && r.y<img.rows-(r.height+20))
    	{
	    	r_class.x = r.x-10;
	    	r_class.y = r.y-10;
	    	r_class.width = r.width+20;
	    	r_class.height = r.height+20;
    	}
    	else 
    		r_class = r;
    	
    	Mat r_norm = out(r);
    	Mat roi_class = img(r_class);
        Mat roi = img(r);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
         //indContours( , contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        int aspect = r.height/r.width;
        int mserarea = mser_points[i].size();
        double area_ratio = (double)mserarea/(r.width*r.height);


        if((roi.rows>14 && roi.rows<110) && (roi.cols>14 && roi.cols<110) && (aspect>0.5 && aspect<1.5) && (area_ratio>0.4 && area_ratio <1))
        {	
        	imshow("tttttttttt",roi_class);
    		imshow("t",roi);
        	float flag = classify(roi_class,detectList);

        	//cout<<"\n*********DONE DANA DONE*******";
        	//cout<<"\n-----------------------area_ratio: "<<area_ratio;
        	if(flag==1)
   		     {rectangle(img, r , Scalar(0,255,0),2); //cout<<"\n\n** ROI rows: "<<roi.rows<<" ---ROI cols: "<<roi.cols;
   		 		//cout<<"\n--------"<<flag<<"\n";
   		 		//waitKey(2000);
   		 	}

        }
    }
     char winName[10];
   // cout<<"\n-------------------"<<detectList.size(); 

    //>>>>>>>>>>>>>>>>>>>>>>>>>>ADD THIS PART AND DECLARE NUM=0 GLOBALLY<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    int x;
    /*if(num>detectList.size() && detectList.size()>0 )
    for(x=1;x<num;x++)
        {   sprintf(winName, "Sign %d",x);.00
            destroyWindow(winName);
            //detectList.clear();
        } //////////////////////////////////////////////////////////////
    
    for( x =0; x<detectList.size();x++)
    {   
        sprintf(winName,"Sign %d",x);
        imshow(winName,detectList[x]);
        //cv::waitKey(2000);
    }*/
        
    	img.copyTo(output(Rect(0, 0, img.cols, img.rows)));
    	//cout<<"-------------------"<<detectList.size();
        for( x =0; x<detectList.size() && x<6;x++)
        {   int y;
            if(x<6)
                y=x;
            if(x>=6)
                y=x%6;
        	Mat final = detectList[x];
        	final.copyTo(output(Rect(y*100,img.rows,final.cols,final.rows)));
        }
    //num=detectList.size();
    
        // cout<<"------------------------------------+ "<<img.rows<<" ++ "<<img.cols;
          imshow("mser",output);
          imshow("key",out_1);
          waitKey(100);
          double fps = getTickFrequency() / (getTickCount() - t0);
            printf("\n Frame rate : %2.2f", fps);
}while(cap.grab());
cap.release();
    return 0;
}



