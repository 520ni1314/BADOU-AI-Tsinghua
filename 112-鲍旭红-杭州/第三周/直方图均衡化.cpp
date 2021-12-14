#include "opencv2/opencv.hpp"
#include"opencv2\core\core_c.h"



using namespace std;
using namespace cv;




void histogramEqualization(Mat srcImg, Mat &dstImg)
{
	/*�Ҷ�ֱ��ͼֻ��Ե�ͨ����ͼ*/
	if ((srcImg.type() != CV_8UC1) || !srcImg.data)
	{
		return;
	}
	dstImg = Mat::zeros(srcImg.size(), srcImg.type());
	int cols = srcImg.cols;
	int rows = srcImg.rows;
	double total = cols * rows;
	double a[256] = { 0 };
	int max = 0;
	int min = 255;
	/*�����ۼ�ֱ��ͼ*/
	for (int i = 0; i < rows; i++)
	{
		uchar *pSrc = srcImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int temp = pSrc[j];
			if (temp > max)
			{
				max = temp;
			}
			else if (temp < min)
			{
				min = temp;
			}
			a[temp]++;
		}
	}
	for (int i = 0; i < 256; i++)
	{
		a[i] = a[i] / total;
	}
	double tmp = a[0];
	for (int i = 1; i < 256; i++)
	{
		a[i] += tmp;
		tmp = a[i];
	}
	for (int i = 0; i < rows; i++)
	{
		uchar *pSrc = srcImg.ptr<uchar>(i);
		uchar *pDst = dstImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int temp = pSrc[j];
			pDst[j] = a[temp] * (max - min) + min;
		}
	}
}






#if 1
int main(void)
{
	string path = "D:\\�˶��廪��\\��2����ѧ����&����ͼ��\\�˶�ѧԺ��ҵ\\lena_c.bmp";
	Mat srcImg = imread(path);
	Mat grayImg;
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	Mat dstImg;
	histogramEqualization(grayImg, dstImg);
	imshow("srcImg", srcImg);
	imshow("grayImg", grayImg);
	imshow("dstImg", dstImg);
	waitKey(0);
}
#endif
