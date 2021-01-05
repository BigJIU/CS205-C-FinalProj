#include "head.h"
#include "face_binary_cls.cpp"
#include<exception>
#include <chrono>
#include <algorithm>
#include<cmath>
#define MAX_FLOAT_SIZE 65600
using namespace std;
using namespace chrono;
using namespace cv;


int now_size;
float* BGRaINI(Mat img, float* in)
{
    int posi = 0;
    Mat image(img.rows, img.cols, CV_8UC3);
    for (int i = 0; i < img.rows; ++i) {
        Vec3b* p2 = img.ptr<Vec3b>(i);
        for (int j = 0; j < img.cols; ++j) {
            in[posi] = float(p2[j][2])/255.f;
            in[posi + 16384] = float(p2[j][1]) / 255.f;
            in[posi + 32768] = float(p2[j][0]) / 255.f;
            //std::cout << "\n r g b" << in[posi] << in[posi + 16384] << in[posi + 32768];
            posi++;
        }
    }
    return in;
}

float* conV(float* in, int level) {
    conv_param conv = conv_params[level];
    int ouSize;//the size of output
    int conv_add = 0;
    if (level == 0) ouSize = now_size / 2;
    else if (level == 1) ouSize = 30;
    else ouSize = 8;
    float* ou = new float[(conv.out_channels) * ouSize * ouSize]{ 0 };
    float* window = new float[10];
    int ou_size = 0;
    int ou_add = 0;
    for (; ou_size < conv.out_channels; ou_size++)//16
    {
        for (int i = 0; i < ouSize * ouSize; i++)
        {
            ou[ou_size * ouSize * ouSize + i] += conv.p_bias[ou_size];
        }
        for (int inpu_size = 0; inpu_size < conv.in_channels; inpu_size++)//3
        {
            //std::cout << ou_add << std::endl;
            ou_add = ou_size * ouSize * ouSize;
            window[0] = 0;
            for (int i = 1; i < 10; i++) {//initial window content
                window[i] = conv.p_weight[conv_add++];
            }

            int* windowPosi = new int[11]{ 0 };
            windowPosi[10] = 0;
            int loopNum = ouSize;
            for (int col = 0; col < loopNum; col++)
            {
                if (level == 1)
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
                else  {
                    //level 0 2
                    windowPosi[1] = -100; //col * 2 * now_size + inpu_size * now_size * now_size;
                    windowPosi[2] = (col * 2 - 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[3] = (col * 2 - 1) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[4] = -100; //(col * 2 + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[5] = (col * 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[6] = (col * 2) * now_size + 1 + inpu_size * now_size * now_size;
                    windowPosi[7] = -100; //(col * 2 + 2) * now_size + inpu_size * now_size * now_size;
                    windowPosi[8] = (col * 2 + 1) * now_size + inpu_size * now_size * now_size;
                    windowPosi[9] = (col * 2 + 1) * now_size + 1 + inpu_size * now_size * now_size;

                    if (col == 0) {
                        windowPosi[2] = -1;
                        windowPosi[3] = -1;
                    }

                }

                for (int row = 1; row < loopNum + 1; row++)
                {
                    //if (level == 2)std::cout << inpu_size<< ": ou[" << ou_add << "] = ";
                    for (int i = 9; i > 0; i--)
                    {
                        if (conv.pad==1&&row == loopNum) {if(i == 9 || i == 3 || i == 6)continue;}
                        if (conv.pad == 1 && col == loopNum-1) { if (i == 9 || i == 8 || i == 7)continue; }
                        if (windowPosi[i] >= 0) {
                            ou[ou_add] += in[windowPosi[i]] * window[i];
                            //if (level == 2)std::cout << in[windowPosi[i]] * window[i] <<"("<< windowPosi[i] <<")+";
                            windowPosi[i] = windowPosi[i] + conv.stride;  
                        }
                        else {
                            windowPosi[i] = windowPosi[i + 1] - 1;
                            if (col == 0) { windowPosi[1] = -100; windowPosi[2] = -100; windowPosi[3] = -100; }
                            //if (level == 2)std::cout << 0 << "(" << -1 << ")+";
                        }


                    }
                    //if(level == 2)std::cout << " = "<<ou[ou_add]<<" at ("<<col<<","<<row-1<<")\n";
                    ou_add++;
                }
            }
        }
    }

    delete[] in;
    return ou;
}
float* reLu(float* in, int size) {
    float* ou = new float[size];
    for (int i = 0; i < size; i++)
    {
        if (in[i] < 0) { ou[i] = 0; }
        else { ou[i] = in[i]; }
    }
    delete[] in;
    return ou;
}
float* maxPool(float* in, int l) {
    float* ou = new float[l * now_size * now_size / 4]{ 0 };
    int ou_add = 0;
    for (int i = 0; i < l; i++)
    {
        for (int col = 0; col < now_size; col++)
        {
            for (int row = 0; row < now_size; row++)
            {
                ou[ou_add] = in[col * now_size + row + i * now_size * now_size] > ou[ou_add] ? in[col * now_size + row + i * now_size * now_size] : ou[ou_add];
                ou[ou_add] = in[col * now_size + row + 1 + i * now_size * now_size] > ou[ou_add] ? in[col * now_size + row + 1 + i * now_size * now_size] : ou[ou_add];
                ou[ou_add] = in[(col + 1) * now_size + row + i * now_size * now_size] > ou[ou_add] ? in[(col + 1) * now_size + row + i * now_size * now_size]  : ou[ou_add];
                ou[ou_add] = in[(col + 1) * now_size + row + 1 + i * now_size * now_size] > ou[ou_add] ? in[(col + 1) * now_size + row + 1 + i * now_size * now_size] : ou[ou_add];
                //if(l==32)std::cout << "\nfor " << in[col * now_size + row + i * now_size * now_size] << " " << in[col * now_size + row + 1 + i * now_size * now_size] << " " << in[(col + 1) * now_size + row + i * now_size * now_size] << " " << in[(col + 1) * now_size + row + 1 + i * now_size * now_size] << " we choose:" << ou[ou_add];
                ou_add++;
                row++;
            }
            col++;
        }
    }
    delete[] in;
    return ou;
}
float* flat(float* in,int level) {
    
    fc_param fc = fc_params[level];
    float* ou = new float[fc.out_features];
    float* w = fc.p_weight;
    float* bias = fc.p_bias;
    for (int i = 0; i < fc.out_features; i++)
    {
        ou[i] = bias[i];
    }
    int w_add = 0;
    for (int ouu = 0; ouu < fc.out_features; ouu++)
    {
        for (int inn = 0; inn < fc.in_features; inn++)
        {
            ou[ouu] += in[inn] * w[w_add++];
        }
    }
    return ou;
}
void SoftMax(float* fcl, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += exp(fcl[i]);
    }
    for (int i = 0; i < size; i++)
    {
        fcl[i] = exp(fcl[i]) / sum;
    }
}
int main() {
    char* inpu_name = new char[30];
    cin >> inpu_name;
    Mat image = imread(inpu_name);
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
    auto start = std::chrono::steady_clock::now();
    thep = BGRaINI(image, thep);

    thep = conV(thep, 0);//over no problem output: 16x64x64
    thep = reLu(thep, 16 * 64 * 64);//over no problem output: 16x64x64
    now_size = 64;
    thep = maxPool(thep, 16);//16x32x32

    now_size = 32;
    thep = conV(thep, 1);//may right 32x30x30
    now_size = 30;

    thep = reLu(thep, 32 * 30 * 30);

    thep = maxPool(thep, 32);//may right 32x15x15
    now_size = 15;
    thep = conV(thep, 2);//32x8x8
    now_size = 8;
    thep = reLu(thep, 32 * 8 * 8);

    thep = flat(thep,0);
    SoftMax(thep, fc_params->out_features);

    auto end = std::chrono::steady_clock::now();
    printf("%s%.4f\n", "Score of background: ", thep[0]);
    cout << "---------------------------------------" << endl;
    printf("%s%.4f\n", "Score of face      : ", thep[1]);
    cout << "---------------------------------------" << endl;
    printf("%s%lld%s", "calculation takes ", duration_cast<std::chrono::milliseconds>(end - start).count(), " ms\n");
    imshow("the_pic", image);
    waitKey(0);

////test start
//for (int i = 1800; i < 2700; i++)//4096 65536
//{
//    std::cout << "posi: " << i << " num:" << thep[i] << std::endl;
//}
////std::cout << "posi: " << 28800 << " num:" << thep[28800] << std::endl;
////std::cout << "posi: " << 2048 << " num:" << thep[2048] << std::endl;
////test con

    return 0;
}