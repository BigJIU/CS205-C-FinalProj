#include "head.h"
#include "face_binary_cls.cpp"
#include<exception>
#include <algorithm>
#include<cmath>
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
    else ouSize = 30;
    float* ou = new float[(conv.out_channels)*ouSize*ouSize]{ 0 };
    float* window = new float[10];
    int ou_size = 0;
    int ou_add = 0;
    for (; ou_size < conv.out_channels; ou_size++)//16
    {
        
        for (int inpu_size = 0; inpu_size < conv.in_channels; inpu_size++)//3
        {
            //std::cout << ou_add << std::endl;
            ou_add = ou_size * ouSize * ouSize;
            window[0] = 0;
            for (int i = 1; i < 10; i++) {//initial window content
                window[i] = conv.p_weight[conv_add++];
            }
            int* windowPosi = new int[11]{0};
            windowPosi[10] = 0;
            int loopNum = ouSize;
            for (int col = 0; col < loopNum; col++)
            {
                if(level == 1)
                {
                    //level 1 initial
                    windowPosi[1] = col * now_size + inpu_size * now_size * now_size;
                    windowPosi[2] = col * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[3] = col * now_size + 2 + inpu_size * now_size * now_size;
                    windowPosi[4] = (col + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[5] = (col + 1) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[6] = (col + 1) * now_size + 2 + inpu_size * now_size * now_size;
                    windowPosi[7] = (col + 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[8] = (col + 2) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[9] = (col + 2) * now_size + 2 + inpu_size * now_size * now_size;
                    
                }
                else{
                    //pad 2
                    windowPosi[1] = -1; //col * 2 * now_size + inpu_size * now_size * now_size;
                    windowPosi[2] = (col * 2 - 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[3] = (col * 2 - 1) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[4] = -1; //(col * 2 + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[5] = (col * 2 ) * now_size + inpu_size * now_size * now_size;
                    windowPosi[6] = (col * 2 ) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[7] = -1; //(col * 2 + 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[8] = (col * 2 + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[9] = (col * 2 + 1) * now_size + 1 + inpu_size * now_size * now_size;
                    
                    if (col == 0) {
                        windowPosi[2] = -1;
                        windowPosi[3] = -1;
                    }
                    
                }
                else {
                    //level 2
                    windowPosi[1] = -100; //col * 2 * now_size + inpu_size * now_size * now_size;
                    windowPosi[2] = -100;
                    windowPosi[3] = (col * 2 - 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[4] = -100; //(col * 2 + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[5] = -100;
                    windowPosi[6] = (col * 2 - 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[7] = -100; //(col * 2 + 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[8] = -100;
                    windowPosi[9] = (col * 2) * now_size + inpu_size * now_size * now_size;

                    if (col == 0) {
                        windowPosi[2] = -100;
                        windowPosi[3] = -100;
                    }
                }
                for (int row =  1; row < loopNum + 1; row ++)
                {
                    //std::cout << "ou[" << ou_add << "] = ";
                    for (int i = 1; i < 10; i++)
                    {
                        if (windowPosi[i] >=0) {
                            ou[ou_add] += in[windowPosi[i]] * window[i];
                            //std::cout << in[windowPosi[i]] * window[i] <<"("<< windowPosi[i] <<")+";
                        }
                        else {
                            if (col == 0) { windowPosi[1] = -3; windowPosi[2] = -3;windowPosi[3] = -3;}
                            else {windowPosi[i] = windowPosi[i + 1] - 1;}
                            //std::cout << 0 << "(" << -1 << ")+";
                        }
                        
                        
                    }
                    //std::cout << " = "<<ou[ou_add]<<" at ("<<col<<","<<row-1<<")\n";
                    ou[ou_add] += conv.p_bias[ou_size]/conv.in_channels;
                    
                    ou_add++;
                }
            }
        }
    }
    
    delete[] in;
    return ou;
}
float* reLu(float* in,int size) {
    float* ou = new float[size];
    for (int i = 0; i < size; i++)
    {
        if (in[i] < 0) { ou[i] = 0; }
        else { ou[i] = in[i]; }
    }
    delete[] in;
    return ou;
}
float* maxPool(float* in,int l) {
    float* ou = new float[l*now_size * now_size / 4]{0};
    int ou_add = 0;
    for (int i = 0; i < l; i++)
    {
        for (int col = 0; col < now_size; col++)
        {
            for (int row = 0; row < now_size; row++)
            {
                ou[ou_add] = in[col * now_size + row+i*now_size*now_size]> ou[ou_add]? in[col * now_size + row + i * now_size * now_size] : ou[ou_add];
                ou[ou_add] = in[col * now_size + row + 1 + i * now_size * now_size]> ou[ou_add]? in[col * now_size + row + 1 + i * now_size * now_size] : ou[ou_add];
                ou[ou_add] = in[(col + 1) * now_size + row + i * now_size * now_size]> ou[ou_add]? in[(col + 1) * now_size + row + i * now_size * now_size] > ou[ou_add] : ou[ou_add];
                ou[ou_add] = in[(col + 1) * now_size + row+1 + i * now_size * now_size]> ou[ou_add]? in[(col + 1) * now_size + row + 1 + i * now_size * now_size] : ou[ou_add];
                ou_add++;
                row++;
            }
            col++;
        }
    }
    delete[] in;
    return ou;
}
float* flat(float* in) {
    float* ou = new float[2];
    float* w = fc_params->p_weight;
    float* bias = fc_params->p_bias;
    std::cout <<"\nflat[2048]:"<< in[2048];
    std::cout << "\nflat[2047]:" << in[2047];
    float a = bias[0];
    float b = bias[1];
    int j = 0;
    int ii = 0;
        
    for (int i = 0; i < 2048; i++)
    {
        //a += in[i] * w[ii++];
        //b += in[i] * w[ii++];
        a += in[i] * w[ii];
        b += in[i] * w[ii+2048];
        ii++;
        std::cout << "\nthe i: " << i << " is " << in[i];
        //std::cout << "With i:" << i << " a:" << a << " b:" << b << "\n";
    }
    ou[0] = a;
    ou[1] = b;
    return ou;
}
int main() {
	Mat image = imread("face.jpg");
    //imshow("myPic", image); 
    //waitKey(0);
 /*   conv_param conv = conv_params[0];
    int n = 0;
    for (int i = 0; i < 16*3; i++)
    {
        std::cout << conv.p_weight[n++] << " ";
        std::cout << conv.p_weight[n++] << " ";
        std::cout << conv.p_weight[n++] << "\n";
        std::cout << conv.p_weight[n++] << " ";
        std::cout << conv.p_weight[n++] << " ";
        std::cout << conv.p_weight[n++] << "\n";
        std::cout << conv.p_weight[n++] << " ";
        std::cout << conv.p_weight[n++] << " ";
        std::cout << conv.p_weight[n++] << "\n";
        std::cout <<  "\n";
    }*/
    now_size = 128;
    float* thep = new float[MAX_FLOAT_SIZE];
    thep = BGRaINI(image,thep);

    thep = conV(thep, 0);//over no problem output: 16x64x64
    thep = reLu(thep, 16 * 64 * 64);//over no problem output: 16x64x64
    now_size = 64;
    thep = maxPool(thep,16);//16x32x32

    now_size = 32;
    thep = conV(thep, 1);//may right 32x30x30
    now_size = 30;
    thep = reLu(thep, 32 * 30 * 30);
    thep = maxPool(thep,32);//may right 32x15x15
    ////test start
    //for (int i = 0; i < 2048; i++)//4096 65536
    //{
    //    std::cout << "posi: " << i << " num:" << thep[i] << std::endl;
    //}
    ////std::cout << "posi: " << 28800 << " num:" << thep[28800] << std::endl;
    ////std::cout << "posi: " << 2048 << " num:" << thep[2048] << std::endl;
    ////test con
    now_size = 15;
    thep = conV(thep, 2);//32x8x8
    //test start
    for (int i = 0; i < 7200; i++)//4096 65536
    {
        std::cout << "posi: " << i << " num:" << thep[i] << std::endl;
    }
    //std::cout << "posi: " << 28800 << " num:" << thep[28800] << std::endl;
    std::cout << "posi: " << 7200 << " num:" << thep[7200] << std::endl;
    //test con
    thep = reLu(thep,32*16*16);

    thep = flat(thep);
    std::cout << "\n a:" << thep[0];
    std::cout << "\n b:" << thep[1];
    double a = 400;
    double b = 471;
    float e = 2.71828182845f;
    float ae = exp(a);
    float be = exp(b);
    float ans = ae/(ae+be);
    std::cout << "\na+b:" << ae+be;
    std::cout << "\nae:" << ae;
    std::cout << "\nbe:" << be;
    std::cout << "\n400e:" << exp(471.728);
    std::cout << "\n" << ans;
	

	return 0;
}

