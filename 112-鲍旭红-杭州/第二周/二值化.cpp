#include "opencv2/opencv.hpp"



using namespace std;
using namespace cv;



void myThreshold(Mat img, Mat& dst,int minV,int maxV)
{
	uchar *pSrc = img.data;
	int step = img.step / sizeof(uchar);
	/*建立一个和原图像相同大小的单通道黑色背景图像*/
	dst = Mat::zeros(img.size(), CV_8UC1);
	int channels = img.channels();//图片通道
	for (int i = 0; i < img.rows; i++)
	{
		uchar *pData = dst.ptr(i);
		int i_step = i * step;
		for (int j = 0; j < img.cols; j++)
		{
			int j_channels = j*channels;
			int value = pSrc[i_step + j_channels];
			/*满足阈值条件则像素点填充255*/
			if ((value > minV) && (value < maxV))
			{
				pData[j] = 255;
			}
		}
	}
}


#if 1
int main(void)
{
	string path = "D:\\八斗清华班\\【2】数学基础&数字图像\\八斗学院作业\\lena.bmp";
	Mat srcImg = imread(path);
	Mat grayImg;
	myThreshold(srcImg, grayImg,23,100);
	imshow("srcImg", srcImg);
	imshow("grayImg", grayImg);
	waitKey(0);
}
#endif
