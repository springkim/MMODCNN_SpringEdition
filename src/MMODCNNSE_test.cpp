//
// Created by VIRNECT on 2018-11-20.
//
#include"MMODCNNSE_test.h"
DLL_MACRO void* MMODCNNLoad(char* dat){
	net_type* net=new net_type;
	dlib::shape_predictor* sp=new dlib::shape_predictor;
	dlib::deserialize(dat) >> *net >> *sp;
	void** ret=new void*[2];
	ret[0]=static_cast<void*>(net);
	ret[1]=static_cast<void*>(sp);
	return static_cast<void*>(ret);
}
DLL_MACRO void MMODCNNRelease(void* _net){
	void** param=static_cast<void**>(_net);
	net_type* net=static_cast<net_type*>(param[0]);
	dlib::shape_predictor* sp=static_cast<dlib::shape_predictor*>(param[1]);
	delete net;
	delete sp;
}
DLL_MACRO int MMODCNNDetectFromDLibImage(dlib::matrix<dlib::rgb_pixel>& img,void* _net,float threshold,float* result,char** label,int result_sz){
	void** param=static_cast<void**>(_net);
	net_type* net=static_cast<net_type*>(param[0]);
	dlib::shape_predictor* sp=static_cast<dlib::shape_predictor*>(param[1]);
	auto rects=(*net)(img);
	if(rects.size()*6>=result_sz)return -1;
	for(size_t i=0;i<rects.size();i++){
		auto& d=rects[i];
		auto fd=(*sp)(img,d);
		dlib::rectangle rect;
		for (unsigned long j = 0; j < fd.num_parts(); ++j) rect += fd.part(j);

		result[i*6+1]=d.detection_confidence;
		result[i*6+2]=rect.left();
		result[i*6+3]=rect.top();
		result[i*6+4]=rect.width();
		result[i*6+5]=rect.height();

		result[i*6+0]=0;
		//버그가 있나.....
		char** p=NULL;
		for(p=label;*p!=NULL;p++){
			if(strcmp(*p,d.label.c_str())==0)break;
			result[i*6+0]++;
		}
		if(p==NULL || *p==NULL){
			result[i*6+0]=-1;
		}
	}
	return static_cast<int>(rects.size());
}
DLL_MACRO int MMODCNNDetectFromCvMat(void* cvmat,void* _net,float threshold,float* result,char** label,int result_sz){
	cv::Mat mat=*static_cast<cv::Mat*>(cvmat);
	dlib::cv_image<dlib::bgr_pixel> image(mat);
	dlib::matrix<dlib::rgb_pixel> matrix;
	dlib::assign_image(matrix, image);
	return MMODCNNDetectFromDLibImage(matrix,_net,threshold,result,label,result_sz);
}
DLL_MACRO int MMODCNNDetectFromFile(char* img_path,void* _net,float threshold,float* result,char** label,int result_sz){
	dlib::matrix<dlib::rgb_pixel> img;
	dlib::load_image(img,img_path);
	return MMODCNNDetectFromDLibImage(img,_net,threshold,result,label,result_sz);
}

