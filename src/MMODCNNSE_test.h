//
// Created by VIRNECT on 2018-11-20.
//


#ifndef MMODCNN_SPRINGEDITION_DLIBSE_TEST_H
#define MMODCNN_SPRINGEDITION_DLIBSE_TEST_H
#ifdef _MSC_VER
#define DLL_MACRO extern "C" __declspec(dllexport)
#endif
#ifdef __linux__
#define DLL_MACRO
#endif
#include"MMODCNNSE_model.h"
#include<dlib/opencv.h>
DLL_MACRO void* MMODCNNLoad(char* dat);
DLL_MACRO int MMODCNNDetectFromFile(char* img_path,void* _net,float threshold,float* result,char** label,int result_sz);
DLL_MACRO int MMODCNNDetectFromDLibImage(dlib::matrix<dlib::rgb_pixel>& img,void* _net,float threshold,float* result,char** label,int result_sz);
DLL_MACRO int MMODCNNDetectFromCvMat(void* cvmat,void* _net,float threshold,float* result,char** label,int result_sz);
DLL_MACRO void MMODCNNRelease(void* _net);

#endif //MMODCNN_SPRINGEDITION_DLIBSE_TEST_H
