#include "head.h"
#include "face_binary_cls.cpp"
#include<exception>
#define MAX_FLOAT_SIZE 65600
using namespace cv;

int now_size;
float* BGRaINI(Mat img,float* in)
{
    int posi = 0;
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
    int ouSize;//the size of output
    int conv_add = 0;
    if (conv.stride == 2) ouSize = now_size / 2;
    else ouSize = now_size;
    float* ou = new float[(conv.out_channels)*ouSize*ouSize]{ 0 };
    float* window = new float[10];
    int ou_size = 0;
    for (; ou_size < conv.out_channels; ou_size++)//16
    {
        std::cout << ou_size << std::endl;
        int ou_add = 0;
        for (int inpu_size = 0; inpu_size < conv.in_channels; inpu_size++)//3
        {
            std::cout << ou_add << std::endl;
            ou_add = ou_size * ouSize * ouSize;
            window[0] = 0;
            for (int i = 1; i < 10; i++) {//initial window content
                window[i] = conv.p_weight[conv_add++];

            }
            int* windowPosi = new int[11]{0};
            windowPosi[10] = 0;
            for (int col = 0; col < ouSize; col++)
            {
                if(conv.pad == 0)
                {
                    //pad1 initial
                    windowPosi[1] = col * 2 * now_size + inpu_size * now_size * now_size;
                    windowPosi[2] = col * 2 * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[3] = col * 2 * now_size + 2 + inpu_size * now_size * now_size;
                    windowPosi[4] = (col * 2 + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[5] = (col * 2 + 1) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[6] = (col * 2 + 1) * now_size + 2 + inpu_size * now_size * now_size;
                    windowPosi[7] = (col * 2 + 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[8] = (col * 2 + 2) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[9] = (col * 2 + 2) * now_size + 2 + inpu_size * now_size * now_size;
                    
                }
                else{
                    //pad 2
                    windowPosi[1] = -1; //col * 2 * now_size + inpu_size * now_size * now_size;
                    windowPosi[2] = col * 2 * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[3] = col * 2 * now_size + 2 + inpu_size * now_size * now_size;
                    windowPosi[4] = -1; //(col * 2 + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[5] = (col * 2 + 1) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[6] = (col * 2 + 1) * now_size + 2 + inpu_size * now_size * now_size;
                    windowPosi[7] = -1; //(col * 2 + 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[8] = (col * 2 + 2) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[9] = (col * 2 + 2) * now_size + 2 + inpu_size * now_size * now_size;
                    
                    if (col == 0) {
                        windowPosi[2] = -1;
                        windowPosi[3] = -1;
                    }
                    
                }

                for (int row = 1; row < ouSize+1; row++)
                {
                    for (int i = 1; i < 10; i++)
                    {
                        if (windowPosi[i] >=0) {
                            ou[ou_add] += in[windowPosi[i]] * window[i];
                        }
                        else {
                            if (col == 0) { windowPosi[1] = -3; windowPosi[2] = -3;windowPosi[3] = -3;}
                            else {windowPosi[i] = windowPosi[i + 1] - 1;}
                        }
                        windowPosi[i] = windowPosi[i] + conv.stride;
                    }
                    ou_add++;
                }
            }
        }
        std::cout << ou_add << std::endl;
    }
    
    return ou;
}
int main() {
	Mat image = imread("1.jpg");
    now_size = 128;
    float* thep = new float[MAX_FLOAT_SIZE];
    thep = BGRaINI(image,thep);
    thep = conV(thep, 0);
    std::cout << thep[0];
    std::cout << thep[65536]<<" ";
    std::cout << thep[65535] << " ";
    std::cout << thep[65534] << " ";
    std::cout << thep[65533] << " ";
    //fir maxPool
	imshow("myPic",image);
    //thep = conV(thep, 1);
    //thep = conV(thep, 2);
	waitKey(0);
	return 0;
}

