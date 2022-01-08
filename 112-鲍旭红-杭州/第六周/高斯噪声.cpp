#include <fstream>
#include <iostream>
#include <stdlib.h>
#include<vector>
#include <limits>
#include<math.h>
#include <algorithm> // std::move_backward
#include <random> // std::default_random_engine
#include <chrono> // std::chrono::system_clock
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

double generateGaussianNose(double mu, double sigma)
{
	//����һ���ر�С��ֵ
	const double epsilon = numeric_limits<double>::min();//����Ŀ�����������ܱ�ʾ����ƽ�1��������1�Ĳ�ľ���ֵ
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flagΪ�٣������˹�������
	if (!flag)
	{
		return z1 * sigma + mu;
	}
	double u1, u2;
	//�����������
	do 
	{
		u1 = rand()*(1.0 / RAND_MAX);
		u2 = rand()*(1.0 / RAND_MAX);
	} while (u1 < epsilon);
	//flagΪ�湹���˹�������X
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z1 * sigma + mu;
}



//Ϊͼ����Ӹ�˹����
void addGaussianNoise(Mat srcImg, Mat& dstImg, double mu, double sigma)
{
	dstImg = srcImg.clone();
	int channels = srcImg.channels();//��ȡͼ���ͨ��
	int rows = srcImg.rows;			//ͼ�������
	int cols = srcImg.cols * channels;//ͼ���������
	//�ж�ͼ���������
	if (srcImg.isContinuous())//�жϾ����Ƿ��������������������൱��ֻ��Ҫ����һ��һά���� 
	{
		cols *= rows;
		rows = 1;
	}
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			//��Ӹ�˹����
			double gaussian = generateGaussianNose(mu, sigma) * 32;
			//printf("gaussian = %.4f\n", gaussian);
			int val = srcImg.ptr<uchar>(i)[j] + gaussian;
			//printf("val = %d\n", val);
			if (val > 255)
			{
				val = 255;
			}
			if (val < 0)
			{
				val = 0;
			}
			dstImg.ptr<uchar>(i)[j] = (uchar)val;
		}
	}

}


int main(void)
{
	string img_path = "F:\\�˶��廪��\\lena\\temp\\lena.bmp";
	Mat img = imread(img_path);
	if (!img.data)
	{
		return -1;
	}
	imshow("img", img);
	Mat dstImg;
	addGaussianNoise(img, dstImg,0.3,0.9);
	imshow("dstImg", dstImg);
	waitKey(0);
}