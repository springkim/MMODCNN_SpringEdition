//
// Created by VIRNECT on 2018-11-20.
//

#ifndef DLIB_SPRINGEDITION_DLIBSE_MODEL_H
#define DLIB_SPRINGEDITION_DLIBSE_MODEL_H
#include<iostream>
#include<algorithm>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include<dlib/data_io.h>
#include"tinyxml2.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef _MSC_VER
#include<Windows.h>
#include<direct.h>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#endif
template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET> using downsampler = dlib::relu<dlib::bn_con<con5d<32, dlib::relu<dlib::bn_con<con5d<32, dlib::relu<dlib::bn_con<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = dlib::relu<dlib::bn_con<con5<55, SUBNET>>>;
using net_type = dlib::loss_mmod<dlib::con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

#endif //DLIB_SPRINGEDITION_DLIBSE_MODEL_H
