//
// Created by kimbomm on 2018-11-20.
//
#ifndef DLIB_SPRINGEDITION_DLIBSE_TRAIN_H
#define DLIB_SPRINGEDITION_DLIBSE_TRAIN_H

#include"MMODCNNSE_model.h"


std::vector<std::string> FileList(std::string dir_path, std::string ext, bool recursive = false);
struct ImgPathWithBBox {
	std::string path;
	std::vector<dlib::mmod_rect> rects;
};
std::vector<ImgPathWithBBox> ParseImageListFromXML(std::string xml);
#endif //DLIB_SPRINGEDITION_DLIBSE_TRAIN_H
