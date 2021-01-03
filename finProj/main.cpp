#include "head.h"
#include "face_binary_cls.cpp"
#define MAX_FLOAT_SIZE 49160
using namespace cv;

int now_size;
float* BGRaINI(Mat img,float* in)
{
    int posi = 1;
    Mat image(img.rows, img.cols, CV_8UC3);
    for (int i = 0; i < img.rows; ++i) {
        Vec3b* p2 = img.ptr<Vec3b>(i);
        for (int j = 0; j < img.cols; ++j) {
            in[posi] = float(p2[j][2]);
            in[posi + 16384] = float(p2[j][1]);
            in[posi + 32768] = float(p2[j][0]);
            posi++;
        }
    }
    return in;
}
float* conV(float* in, int level) {
    conv_param conv = conv_params[level]; 
    int ouSize;
    int convPosi = 0;
    if (conv.stride == 2) ouSize = now_size / 2;
    else ouSize = now_size;
    float* ou = new float[conv.out_channels*ouSize*ouSize];
    float* window = new float[10];
    for (int i = 1; i < 10; i++)
         window[i] = conv.p_weight[convPosi++];
    int* windowPosi = new int[10];
    

    for (int col = 0; col < ouSize; col++)
    {
        #pragma region windowPosi initial
        windowPosi[1] = col * 2 * now_size;
        windowPosi[2] = col * 2 * now_size +1;
        windowPosi[3] = col * 2 * now_size +2;
        windowPosi[4] = (col * 2 + 1) * now_size;
        windowPosi[5] = (col * 2 + 1) * now_size+1;
        windowPosi[6] = (col * 2 + 1) * now_size+2;
        windowPosi[7] = (col * 2 + 2) * now_size;
        windowPosi[8] = (col * 2 + 2) * now_size+1;
        windowPosi[9] = (col * 2 + 2) * now_size+2;
#pragma endregion
        for (int row = 1; row < ouSize; row++)
        {
            for (int i = 1; i < 10; i++)
            {
                ou[windowPosi[i]] *= window[i];
                windowPosi[i] = windowPosi[i] + conv.stride;
            }
        }
    }

    
    return ou;
}
int main() {
	Mat image = imread("1.jpg");
    now_size = 128;
    float* thep = new float[MAX_FLOAT_SIZE];
    thep = BGRaINI(image,thep);
    conV(thep, 1);

    //fir maxPool
	imshow("myPic",image);
	waitKey(0);
	return 0;
}

