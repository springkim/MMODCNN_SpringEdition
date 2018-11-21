/*
*  MMODCNN.hpp
*  MMODCNN_SpringEdition
*
*  Created by kimbomm on 2018. 11. 20...
*  Copyright 2018 kimbomm. All rights reserved.
*
*/
#if !defined(MMODCNN_SPRINGEDITION_7E2_B_14_DLIBSE_HPP_INCLUDED)
#define MMODCNN_SPRINGEDITION_7E2_B_14_DLIBSE_HPP_INCLUDED
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<opencv2/opencv.hpp>
#ifdef _MSC_VER
#	ifndef NOMINMAX
#	define NOMINMAX
#	endif
#include<Windows.h>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
#pragma warning(disable:4305)
#pragma warning(disable:4290)
#else
#include<dlfcn.h>
#endif

#pragma region spring_edition_box
#ifndef  SPRING_EDITION_BOX
#define SPRING_EDITION_BOX
/**
*	@brief 이 클래스는 cv::Rect를 확장한 것으로 클래스값과 스코어값이 추가되었습니다.
*	@author kimbomm
*	@date 2017-10-05
*/
class BoxSE : public cv::Rect {
public:
	int m_class = -1;
	float m_score = 0.0F;
	std::string m_class_name;
	BoxSE() {
		m_class_name = "Unknown";
	}
	BoxSE(int c, float s, int _x, int _y, int _w, int _h, std::string name = "")
		:m_class(c), m_score(s) {
		this->x = _x;
		this->y = _y;
		this->width = _w;
		this->height = _h;
		char* lb[5] = { "th","st","nd","rd","th" };
		if (name.length() == 0) {
			m_class_name = std::to_string(m_class) + lb[m_class < 4 ? m_class : 4] + " class";
		}
	}
};
#endif
#pragma endregion

class MMODCNN {
protected:
	using MMODCNNLoadType = void*(*)(char* dat);
	using MMODCNNDetectFromFileType=int(*)(char* img_path, void* _net, float threshold, float* result, char** label, int result_sz);
	using MMODCNNDetectFromCvMatType= int(*)(void* cvmat, void* _net, float threshold, float* result, char** label, int result_sz);
	using MMODCNNReleaseType = void(*)(void* _net);


	MMODCNNLoadType MMODCNNLoad = nullptr;
	MMODCNNDetectFromFileType MMODCNNDetectFromFile = nullptr;
	MMODCNNDetectFromCvMatType MMODCNNDetectFromCvMat = nullptr;
	MMODCNNReleaseType MMODCNNRelease = nullptr;

	void* m_network = nullptr;
#ifdef _WIN32
	HMODULE m_hmod = nullptr;
#else
	void* m_hmod = nullptr;
#endif
	std::vector<std::string> m_names;
	char** m_cppnames;

public:

	MMODCNN() {
#ifdef _WIN32
		std::string dll = "libMMODCNNSE.dll";
		m_hmod = LoadLibraryA(dll.c_str());
		if (m_hmod == nullptr) {
			::MessageBoxA(NULL, (dll + " not found. or can't load dependency dlls(cudnn)").c_str(), "Fatal", MB_OK);
			exit(1);
		}
		MMODCNNLoad = (MMODCNNLoadType)GetProcAddress(m_hmod, "MMODCNNLoad");
		MMODCNNDetectFromFile = (MMODCNNDetectFromFileType)GetProcAddress(m_hmod, "MMODCNNDetectFromFile");
		MMODCNNDetectFromCvMat = (MMODCNNDetectFromCvMatType)GetProcAddress(m_hmod, "MMODCNNDetectFromCvMat");
		MMODCNNRelease = (MMODCNNReleaseType)GetProcAddress(m_hmod, "MMODCNNRelease");
#else
		m_hmod = dlopen("libYOLOv3SE.so", RTLD_LAZY);
		if (m_hmod == nullptr) {
			std::cerr << dlerror() << std::endl;
			std::cerr << "libYOLOv3SE.so not found. or can't load dependency dlls" << std::endl;
			exit(1);
		}
		YoloLoad = (YoloLoadType)dlsym(m_hmod, "YoloLoad");
		YoloTrain = (YoloTrainType)dlsym(m_hmod, "YoloTrain");
		YoloDetectFromFile = (YoloDetectFromFileType)dlsym(m_hmod, "YoloDetectFromFile");
		YoloDetectFromImage = (YoloDetectFromImageType)dlsym(m_hmod, "YoloDetectFromImage");
		YoloDetectFromImage = (YoloDetectFromImageType)dlsym(m_hmod, "YoloDetectFromImage");
		YoloRelease = (YoloReleaseType)dlsym(m_hmod, "YoloRelease");
#endif
	}
	void Release() {
		if (this->m_hmod != nullptr) {
			MMODCNNRelease(m_network);
#ifdef _WIN32
			FreeLibrary(this->m_hmod);
#else
			dlclose(this->m_hmod);
#endif
			m_hmod = nullptr;
		}
	}
	~MMODCNN() {
		this->Release();
	}
	void Create(std::string dat, std::string names) {
		this->m_network = MMODCNNLoad(const_cast<char*>(dat.c_str()));
		if (names.length() > 0) {
			std::fstream fin(names, std::ios::in);
			if (fin.is_open() == true) {
				this->m_names.clear();
				while (fin.eof() == false) {
					std::string str;
					std::getline(fin, str);
					if (str.length() > 0) {
						this->m_names.push_back(str);
					}
				}
				fin.close();
			}
			m_cppnames = new char*[this->m_names.size()+1];
			for (size_t i = 0; i < this->m_names.size(); i++) {
				m_cppnames[i] = new char[this->m_names[i].length() + 1];
				strcpy(m_cppnames[i], m_names[i].c_str());
			}
			m_cppnames[this->m_names.size()] = nullptr;
		}
	}
	std::vector<BoxSE> Detect(std::string file, float threshold) {
		float result[6000] = { 0 };
		int n = MMODCNNDetectFromFile(const_cast<char*>(file.c_str()), this->m_network, threshold, result, m_cppnames,6000);
		std::vector<BoxSE> boxes;
		for (int i = 0; i < n; i++) {
			BoxSE box;
			box.m_class = static_cast<int>(result[i * 6 + 0]);
			box.m_score = result[i * 6 + 1];
			box.x = static_cast<int>(result[i * 6 + 2]);
			box.y = static_cast<int>(result[i * 6 + 3]);
			box.width = static_cast<int>(result[i * 6 + 4]);
			box.height = static_cast<int>(result[i * 6 + 5]);
			if (this->m_names.size() > 0) {
				box.m_class_name = box.m_class != -1 ? this->m_names[box.m_class] : "Unknown";
			}
			if (box.m_score > threshold) {
				boxes.push_back(box);
			}
		}
		std::sort(boxes.begin(), boxes.end(), [](BoxSE a, BoxSE b)->bool { return a.m_score > b.m_score; });
		return boxes;
	}
	std::vector<BoxSE> Detect(cv::Mat img, float threshold) {
		float result[6000] = { 0 };
		int n = MMODCNNDetectFromCvMat(static_cast<void*>(&img), this->m_network, threshold, result, m_cppnames, 6000);
		std::vector<BoxSE> boxes;
		for (int i = 0; i < n; i++) {
			BoxSE box;
			box.m_class = static_cast<int>(result[i * 6 + 0]);
			box.m_score = result[i * 6 + 1];
			box.x = static_cast<int>(result[i * 6 + 2]);
			box.y = static_cast<int>(result[i * 6 + 3]);
			box.width = static_cast<int>(result[i * 6 + 4]);
			box.height = static_cast<int>(result[i * 6 + 5]);
			if (this->m_names.size() > 0) {
				box.m_class_name = box.m_class != -1 ? this->m_names[box.m_class] : "Unknown";
			}
			if (box.m_score > threshold) {
				boxes.push_back(box);
			}
		}
		std::sort(boxes.begin(), boxes.end(), [](BoxSE a, BoxSE b)->bool { return a.m_score > b.m_score; });
		return boxes;
	}
};

#endif  //MMODCNN_SPRINGEDITION_7E2_B_14_DLIBSE_HPP_INCLUDED