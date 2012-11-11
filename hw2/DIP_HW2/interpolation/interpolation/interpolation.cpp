// interpolation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h" 
#include <opencv2/opencv.hpp>
#include<time.h>

using namespace std;
using namespace cv;

void Resize_bilinear(IplImage *src, IplImage *dst,int scale);
void Rotation_bicubic(IplImage* src, IplImage* dst,int angle);
void bound(int x, int y, float ca, float sa, int *xmin, int *xmax, int *ymin, int *ymax);
CvSize cvsize;
int scale=1;
int angle=0;

int main(int argc, char* argv[])
{
	int height,width,step,channels;

	//從文件中讀入圖像
	IplImage* img = cvLoadImage("D:\\ddd\\Course\\DIP\\picture\\lena400.jpg");
    IplImage* img_bilinear=0;
	IplImage* img_bicubic=0;	

	//如果讀入圖像失敗
	if(!img)
	{
		printf("Could not load imag");
		return -1;
	}

	height = img->height;
	width = img->width;
	step = img->widthStep;
	channels = img->nChannels;
	printf("Processing a %dx%d image with %d channels\n",height,width,channels); 
	
	cvNamedWindow("win0", CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("win0", 100, 100);
	cvShowImage("win0", img);

	waitKey();
	//放大
	printf("please type the amplification factor =>\n"); 
	scanf_s("%d",&scale);

	cvsize.width=(int)(img->width*scale);
    cvsize.height=(int)(img->height*scale);
    img_bilinear=cvCreateImage(cvsize,img->depth,img->nChannels);
	img_bicubic=cvCreateImage(cvsize,img->depth,img->nChannels);
	
	//bilinear放大
	//clock_t start=clock();記錄時間
	cvResize(img,img_bilinear,CV_INTER_LINEAR);
	//Resize_bilinear(img,img_bilinear,scale);
	//printf( "%f seconds\n", (clock()-start) ); 顯示時間
	
	//start=clock();記錄時間
	cvResize(img,img_bicubic,CV_INTER_CUBIC);//bicubic放大
	//printf( "%f seconds\n",(clock()-start) );顯示時間

	cvNamedWindow("win_bilinear", 1);
	cvMoveWindow("win_bilinear", 300, 100);
	cvNamedWindow("win_bicubic", 1);
	cvMoveWindow("win_bicubic", 500, 100);
	
	cvShowImage("win_bilinear", img_bilinear);
    cvShowImage("win_bicubic", img_bicubic);

	waitKey();
	cvDestroyWindow("win_bilinear");
	cvDestroyWindow("win_bicubic");
	
	//縮小	
	printf("please type the shrink ratio =>\n"); 
	scanf_s("%d",&scale);

	cvsize.width=(int)(img->width/scale);
    cvsize.height=(int)(img->height/scale);
    img_bilinear=cvCreateImage(cvsize,img->depth,img->nChannels);
	img_bicubic=cvCreateImage(cvsize,img->depth,img->nChannels);

	cvResize(img,img_bilinear,CV_INTER_LINEAR);
	cvResize(img,img_bicubic,CV_INTER_CUBIC);

	cvNamedWindow("win_bilinear", 1);
	cvMoveWindow("win_bilinear", 300, 100);
	cvNamedWindow("win_bicubic", 1);
	cvMoveWindow("win_bicubic", 500, 100);

	cvShowImage("win_bilinear", img_bilinear);
    cvShowImage("win_bicubic", img_bicubic);

	waitKey();
	cvDestroyWindow("win_bilinear");
	cvDestroyWindow("win_bicubic");

	//旋轉
	printf("please type the angle of rotation =>\n"); 
	scanf_s("%d",&angle);

    int w = img->width;
    int h = img->height;
    float m[6];                  
    CvMat M;

	/****************************
    m[0] = (float)(cos(angle*CV_PI/180));
    m[1] = (float)(sin(angle*CV_PI/180));
    m[3] = -m[1];
    m[4] = m[0];
    m[2] = w*0.5f;  
    m[5] = h*0.5f;  

	M = cvMat( 2, 3, CV_32F, m);
    cvGetQuadrangleSubPix(img,img_bilinear, &M); 
	****************************/
		
	int newheight =int (fabs(( sin(angle*CV_PI/180)*w)) + fabs((cos(angle*CV_PI/180)*h)));
	int newwidth  =int (fabs(( sin(angle*CV_PI/180)*h)) + fabs((cos(angle*CV_PI/180)*w)));    
	img_bilinear=cvCreateImage(cvSize(newwidth,newheight),img->depth,img->nChannels);
	img_bicubic=cvCreateImage(cvSize(newwidth,newheight),img->depth,img->nChannels);
	float ca,sa;
	int xmin,xmax,ymin,ymax,sx,sy; 
	angle=-angle;
	ca = (float)cos((double)(angle)*CV_PI/180.0); 
	sa = (float)sin((double)(angle)*CV_PI/180.0);  
	xmin = xmax = ymin = ymax = 0;        
	bound(w-1,0,ca,sa,&xmin,&xmax,&ymin,&ymax);        
	bound(0,h-1,ca,sa,&xmin,&xmax,&ymin,&ymax);        
	bound(w-1,h-1,ca,sa,&xmin,&xmax,&ymin,&ymax);
	sx = xmax-xmin+1;
	sy = ymax-ymin+1;
	m[0] = ca;
	m[1] = sa; 
	m[2] =-(float)xmin;
	m[3] =-m[1];   
	m[4] = m[0];     
	m[5] =-(float)ymin;
	M = cvMat( 2, 3, CV_32F, m);
	
	cvWarpAffine( img, img_bilinear,&M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0));
	cvWarpAffine( img, img_bicubic,&M,CV_INTER_CUBIC+CV_WARP_FILL_OUTLIERS,cvScalarAll(0));

	cvNamedWindow("win_bilinear", 1);
	cvMoveWindow("win_bilinear", 300, 100);
	cvNamedWindow("win_bicubic", 1);
	cvMoveWindow("win_bicubic", 500, 100);

    cvShowImage("win_bilinear", img_bilinear);
	cvShowImage("win_bicubic", img_bicubic);
    waitKey();

	cvDestroyWindow("win0");
	cvDestroyWindow("win_bilinear");
	cvDestroyWindow("win_bicubic");
	cvReleaseImage(&img);
    cvReleaseImage(&img_bilinear);
	cvReleaseImage(&img_bicubic);

	return 0;
}

void bound(int x, int y, float ca, float sa, int *xmin, int *xmax, int *ymin, int *ymax)
{       
int rx,ry;       
rx = (int)floor(ca*(float)x+sa*(float)y);   
ry = (int)floor(-sa*(float)x+ca*(float)y); 
if (rx<*xmin) *xmin=rx;
if (rx>*xmax) *xmax=rx;   
if (ry<*ymin) *ymin=ry; 
if (ry>*ymax) *ymax=ry;
}


void Resize_bilinear(IplImage* src, IplImage* dst,int scale)
{
int w;
int h;
int w0 = src->width;
int h0 = src->height;
int w1;
int h1;
int i;

double a=0;
double b=0;

CvScalar f0;
CvScalar f1;
CvScalar f2;
CvScalar f3;
CvScalar g;

for(w=0;w<(w0-1)*scale;w++)
	{
		for(h=0;h<(h0-1)*scale;h++)
		{
			w1=int(w/scale);
			h1=int(h/scale);
		    a=1.0*w/scale-1.0*w1;
			b=1.0*h/scale-1.0*h1;
			f0=cvGet2D(src,h1,w1);
			f1=cvGet2D(src,h1,w1+1);
			f2=cvGet2D(src,h1+1,w1);
			f3=cvGet2D(src,h1+1,w1+1);
			for(i=0;i<4;i++)
			{
			g.val[i]=(1-a)*(1-b)*f0.val[i]+a*(1-b)*f1.val[i]+(1-a)*b*f2.val[i]+a*b*f3.val[i];
			}
			
			cvSet2D(dst,h,w,g);
		}
	}
}


void Rotation_bicubic(IplImage* src, IplImage* dst,int angle)
{
int w;
int h;
int w0 = src->width;
int h0 = src->height;
int i;
int j;
int k;
int l;

double x;
double y;
double a;
double b;

for(w=0;w<(w0-1);w++)
	{
		for(h=0;h<(h0-1);h++)
		{		   	  
			double r[9];
			double x1[3];
				
			r[0] = (float)(cos(angle*CV_PI/180));
            r[1] = (float)(sin(angle*CV_PI/180));
            r[2] = w0*0.5f-0.5*w0*cos(angle*CV_PI/180)-0.5*h0*sin(angle*CV_PI/180); 
			r[3] = -r[1];
            r[4] = r[0];            
            r[5] = h0*0.5f-0.5*h0*cos(angle*CV_PI/180)+0.5*w0*sin(angle*CV_PI/180);
			r[6] = 0;
			r[7] = 0;
			r[8] = 1;
			
			x1[0] = 1.0*w;
			x1[1] = 1.0*h;
			x1[2] = 1.0;

			CvMat *X0 = cvCreateMat( 3, 1, CV_64FC1);			
			//CvMat *R = cvCreateMat( 3, 3, CV_64FC1); 
			//cvInitMatHeader(R,3,3, CV_64FC1,r);
			CvMat *R = &cvMat( 3, 3, CV_64FC1,r);
			CvMat *X1 = &cvMat( 3, 1, CV_64FC1,x1);

			cvMatMul(R,X1,X0);
		    
			x=cvmGet(X0,0,0);
			y=cvmGet(X0,1,0);		

			i=int(x);
			j=int(y);
			
			a=x-1.0*i;
			b=y-1.0*j;

			double s[4];
			for(k=0;k<=3;k++)
			{				
			if(fabs(a+1-k)>=0 && fabs(a+1-k)<1)
			  s[k]=1-pow(2*fabs(a+1-k),2)+pow(fabs(a+1-k),3);			
			else if(fabs(a+1-k)>=1 && fabs(a+1-k)<2)
			  s[k]=4-8*fabs(a+1-k)+pow(5*fabs(a+1-k),2)-pow(fabs(a+1-k),3);
			else if(fabs(a+1-k)>=2)
			  s[k]=0;
			}
			CvMat *A = cvCreateMat( 1,4, CV_64FC1); 
			cvInitMatHeader(A,1,4, CV_64FC1,s);

			for(k=0;k<=3;k++)
			{				
			if(fabs(b+1-k)>=0 && fabs(b+1-k)<1)
			  s[k]=1-pow(2*fabs(b+1-k),2)+pow(fabs(b+1-k),3);			
			else if(fabs(b+1-k)>=1 && fabs(b+1-k)<2)
			  s[k]=4-8*fabs(b+1-k)+pow(5*fabs(b+1-k),2)-pow(fabs(b+1-k),3);
			else if(fabs(b+1-k)>=2)
			  s[k]=0;
			}
			CvMat *B = cvCreateMat( 1,4, CV_64FC1); 
			cvInitMatHeader(B,1,4, CV_64FC1,s);

			double m[16];
			CvScalar f;
		    CvScalar g;
			int u;
			int v;
			
			for(l=0;l<3;l++)
			{
			  for(u=i-1;u<=i+2;u++)
	          {
		        for(v=j-1;v<=j+2;v++)
			    {
			      for(k=0;k<16;k++)
			      {
			        f=cvGet2D(src,v,u);
				    m[k]=f.val[l];				 
			      }
				  CvMat *M = cvCreateMat( 4,4, CV_64FC1); 
			      cvInitMatHeader(M,4,4, CV_64FC1,m);				
				  CvMat *N = cvCreateMat( 1,4, CV_64FC1); 
				  CvMat *G = cvCreateMat( 1,1, CV_64FC1); 
				  cvMatMul(A,M,N);
				  cvMatMul(N,B,G);

				  g.val[l]=cvmGet(G, 0, 0);
			    }
			  }
			}

			cvSet2D(dst,h,w,g);
	    }  
	}
}
