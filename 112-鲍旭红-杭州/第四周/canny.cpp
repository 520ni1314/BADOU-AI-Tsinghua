#include<iostream>
#include"opencv2\opencv.hpp"



using namespace std;
using namespace cv;


//���ɸ�˹�����kernel
void Gussian_kernel(int kernel_size, int sigma, Mat &kernel)
{
	const double PI = 3.1415926;
	int m = kernel_size / 2;
	kernel = Mat(kernel_size, kernel_size, CV_32FC1);
	float s = 2 * sigma*sigma;
	uchar *pKernel = NULL;
	for (int i = 0; i < kernel_size; i++)
	{
		pKernel = kernel.ptr<uchar>(i);
		for (int j = 0; j < kernel_size; j++)
		{
			int x = i - m;
			int y = j - m;
			pKernel[j] = exp(-(x*x + y*y) / s) / (PI *s);
		}
	}
}




/*
�����ݶ�ֵ�ͷ���
imageSource ԭʼ�Ҷ�ͼ
imageX X�����ݶ�ͼ��
imageY Y�����ݶ�ͼ��
gradXY �õ���ݶȷ�ֵ
pointDirection �ݶȷ���Ƕ�
*/
void GradDirection(const Mat srcImg, Mat &imgX, Mat &imgY, Mat &gradXY, Mat &theta)
{
	imgX = Mat::zeros(srcImg.size(), CV_32SC1);
	imgY = Mat::zeros(srcImg.size(), CV_32SC1);
	gradXY = Mat::zeros(srcImg.size(), CV_32SC1);
	theta = Mat::zeros(srcImg.size(), CV_32SC1);
	int rows = srcImg.rows;
	int cols = srcImg.cols;
	int step = srcImg.step;
	int sttepXY = gradXY.step;
	uchar *PX = imgX.data;
	uchar *PY = imgY.data;
	uchar *P = srcImg.data;
	uchar *XY = gradXY.data;
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int a00 = P[(i - 1) * step + j - 1];
			int a01 = P[(i - 1) * step + j];
			int a02 = P[(i - 1) * step + j + 1];

			int a10 = P[i * step + j - 1];
			int a11 = P[i * step + j];
			int a12 = P[i * step + j + 1];

			int a20 = P[(i + 1) * step + j - 1];
			int a21 = P[(i + 1) * step + j];
			int a22 = P[(i + 1) * step + j + 1];

			double gradY = double(a02 + 2 * a12 + a22 - a00 - 2 * a10 - a20);
			double gradX = double(a00 + 2 * a01 + a02 - a20 - 2 * a21 - a22);

			imgX.at<int>(i,j) = abs(gradX);
			imgY.at<int>(i,j) = abs(gradY);
			if (gradX == 0)
			{
				gradX = 0.000000000001;
			}
			theta.at<int>(i,j) = atan(gradY / gradX) * 57.3;
			theta.at<int>(i, j) = (theta.at<int>(i,j) + 360) / 360;
			gradXY.at<int>(i, j) = sqrt(gradX*gradX + gradY*gradY);
		}
	}
	convertScaleAbs(imgX, imgX);
	convertScaleAbs(imgY, imgY);
	convertScaleAbs(theta, theta);
	convertScaleAbs(gradXY, gradXY);
}



/*�ֲ��Ǽ���ֵ����
���Ÿõ��ݶȷ��򣬱Ƚ�ǰ��������ķ�ֵ��С�����õ����ǰ�����㣬����
���õ�С��ǰ����һ�㣬����Ϊ0
srcImg:����õ����ݶ�ͼ��
dstImg:����ķǼ���ֵ����ͼ��
theta:ÿ�����ص���ݶȷ���Ƕ�
imgX:X�����ݶ�
imgY:Y�����ݶ�
*/
void NonlocalMaxValue(Mat srcImg, Mat &dstImg, Mat &theta, Mat &imgX, Mat &imgY)
{
	dstImg = srcImg.clone();
	int cols = srcImg.cols;
	int rows = srcImg.rows;
	for (int i = 1; i < rows - 1; i++)
	{
		uchar *pSrcImg0 = srcImg.ptr<uchar>(i - 1);
		uchar *pSrcImg = srcImg.ptr<uchar>(i);
		uchar *pSrcImg1 = srcImg.ptr<uchar>(i + 1);
		int *pTheta = theta.ptr<int>(i);
		uchar *pX = imgX.ptr<uchar>(i);
		uchar *pY = imgY.ptr<uchar>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (pSrcImg[j] == 0)
			{
				continue;
			}
			int g00 = pSrcImg0[j - 1];
			int g01 = pSrcImg0[j];
			int g02 = pSrcImg0[j + 1];

			int g10 = pSrcImg[j - 1];
			int g11 = pSrcImg[j];
			int g12 = pSrcImg[j + 1];

			int g20 = pSrcImg1[j - 1];
			int g21 = pSrcImg1[j];
			int g22 = pSrcImg1[j + 1];

			int direction = pTheta[j];//�õ��ݶȵĽǶ�ֵ
			int g1 = 0;
			int g2 = 0;
			int g3 = 0;
			int g4 = 0;
			double tmp1 = 0.0;//���������ص��ֵ�õ��ĻҶ���
			double tmp2 = 0.0;
			double weight = fabs((double)pY[j] / (double)pX[j]);
			if (weight == 0)
			{
				weight = 0.0000001;
			}
			if (weight > 1)
			{
				weight = 1 / weight;
			}
			if ((0 <= direction && direction < 45) || (180 < direction && (direction < 225)))
			{
				tmp1 = g10*(1 - weight) + g20*weight;
				tmp2 = g02 * weight + g12 * (1 - weight);
			}
			else if ((45 <= direction && direction < 90) || 225 <= direction &&direction < 270)
			{
				tmp1 = g01 * (1 - weight) + g02 * weight;
				tmp2 = g20 * (weight)+g21 * (1 - weight);
			}
			else if ((90 <= direction && direction < 135) || 270 <= direction &&direction < 315)
			{
				tmp1 = g00 * weight + g01 * (1 - weight);
				tmp2 = g21 * (1 - weight) + g22 * weight;
			}
			else if ((135 <= direction && direction < 180) || 315 <= direction &&direction < 360)
			{
				tmp1 = g00 * weight + g10 * (1 - weight);
				tmp2 = g12 * (1 - weight) + g22 * weight;
			}
			if ((pSrcImg[j] < tmp1) || (pSrcImg[j] < tmp2))
			{
				pSrcImg[j] = 0;
			}
		}
	}
}




/*˫��ֵ����
���ƣ�ָ��һ������ֵA,һ������ֵB,һ��ȡBΪͼ������Ҷȼ��ֲ�70%����BΪ1.5��2����С��A��
�Ҷ�ֵС��A�ģ���Ϊ0���Ҷ�ֵ����B�ģ�תΪ255
*/
void DoubleThreshold(Mat &srcImg, const double lowTH, const double highTH)
{
	int cols = srcImg.cols;
	int rows = srcImg.rows;
	for (int i = 0; i < rows; i++)
	{
		uchar *pSrcImg = srcImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			double tmp = pSrcImg[j];
			tmp = (tmp > highTH) ? (255) : tmp;
			tmp = (tmp < lowTH) ? (0) : tmp;
			pSrcImg[j] = tmp;
		}
	}
}



/*
���Ӵ���
�Ҷ�ֵ����A��B֮��ģ���������ص��ٽ���8�����Ƿ��лҶ�ֵΪ255�ģ�
��û��255�ģ���ʾ����һ�������ľֲ�����ֵ�㣬�����ų�����Ϊ0
����255����ʾ ������������Ե�����ӣ���Ϊ255
֮���ظ�ִ�в��裬ֱ�����������һ�����ص�

��������㷨����ֵΪ255�����ص�����ҵ���Χ����Ҫ��ĵ㣬������Ҫ��ĵ���Ϊ255��
Ȼ���޸�i,j������ֵ��i,j���л��ˣ��ڸı���i,j�����ϼ���Ѱ��255��Χ����Ҫ��ĵ㡣
����������255�ĵ��޸�����ٰ�����������˵�ľֲ�����ֵ����Ϊ0
*/
void DoubleThresholdLink(Mat &srcImg, double lowTH, double highTH)
{
	int cols = srcImg.cols;
	int rows = srcImg.rows;
	for (int i = 1; i < rows - 1; i++)
	{
		uchar *pSrcImg = srcImg.ptr<uchar>(i);
		for (int j = 1; j < cols; j++)
		{
			double pix = pSrcImg[j];
			if (pix != 255)
			{
				continue;
			}
			bool change = false;
			for (int k = -1; k <= 1; k++)
			{
				uchar *pSrcImg1 = srcImg.ptr<uchar>(i + k);
				for (int u = -1; u <= 1; u++)
				{
					if ((k == 0) || (u == 0))
					{
						continue;
					}
					double tmp = pSrcImg1[j + u];
					if ((tmp >= lowTH) && (tmp <= highTH))
					{
						pSrcImg1[j + u] = 255;
						change = true;
					}
				}
			}
			if (change)
			{
				if (i > 1)
				{
					i--;
				}
				if (j > 2)
				{
					j -= 2;
				}
			}
		}
	}
	for (int i = 0; i < rows; i++)
	{
		uchar *pSrcImg = srcImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			if (pSrcImg[j] != 255)
			{
				pSrcImg[j] = 0;
			}
		}
	}
}



int main()
{
	Mat imgGray = imread("D:\\lena\\lena512color.tiff", 0);
	imshow("imgGray", imgGray);
	waitKey();
}