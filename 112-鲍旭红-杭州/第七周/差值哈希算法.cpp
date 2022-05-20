#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2\imgproc\imgproc.hpp"



using namespace cv;
using namespace std;



string difference_hash(Mat src)
{
	Mat resize_8_9;
	resize(src, resize_8_9, Size(8, 9));
	Mat grayImg;
	if (src.channels() != 1)
	{
		cvtColor(resize_8_9, grayImg, CV_BGR2GRAY);
	}
	int channels = grayImg.channels();
	int rows = grayImg.rows;
	int cols = grayImg.cols * channels;//ͼ��������
									   /*�ж�ͼ���������*/
	if (grayImg.isContinuous())
	{
		/*���ͼ���������൱��ֻ����һ��һά����*/
		cols *= rows;
		rows = 1;
	}
	/*���ֵ��ϣ*/
	string hash = "";
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols - 1; j++)
		{
			uchar val1 = grayImg.ptr<uchar>(i)[j];
			uchar val2 = grayImg.ptr<uchar>(i)[j + 1];
			if (val1 > val2)
			{
				hash += "1";
			}
			else
			{
				hash += "0";
			}
		}
	}
	return hash;
}



int main(void)
{
	string path1 = "F:\\�˶��廪��\\lena\\temp\\lena.bmp";
	string path2 = "F:\\�˶��廪��\\lena\\temp\\lena.jpg";
	Mat srcImg1 = imread(path1);
	Mat srcImg2 = imread(path2);
	string hash1 = difference_hash(srcImg1);
	string hash2 = difference_hash(srcImg2);
	cout << hash1 << "\n" << hash2 << endl;
	int similarity_rate = 0;
	for (int i = 0; i < hash1.length(); i++)
	{
		if (hash1[i] == hash2[i])
		{
			similarity_rate++;
		}
	}
	cout << "����ͼƬ��������: " << similarity_rate << "%" << endl;
	system("pause");
}