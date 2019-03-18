//
// Created by kimbomm on 2018-11-20.
//
#include"MMODCNNSE_train.h"
#include<dlib/threads.h>
#include<chrono>
dlib::mutex count_mutex;
dlib::signaler count_signaler(count_mutex);
#ifdef _MSC_VER
std::vector<std::string> FileList(std::string dir_path, std::string ext, bool recursive/* = false*/) {
	std::vector<std::string> paths; //return value
	if (dir_path.back() != '/' && dir_path.back() != '\\') {
		dir_path.push_back('/');
	}
	std::string str_exp = dir_path + "*.*";
	std::vector<std::string> allow_ext;
	std::string::size_type offset = 0;
	while (offset < ext.length()) {
		std::string str = ext.substr(offset, ext.find(';', offset) - offset);
		std::transform(str.begin(), str.end(), str.begin(), toupper);
		offset += str.length() + 1;
		std::string::size_type pos = str.find_last_of('.');
		pos = pos == std::string::npos ? 0 : pos + 1;
		allow_ext.push_back(str.substr(pos, str.length()));
	}
	WIN32_FIND_DATAA fd;
	HANDLE hFind = ::FindFirstFileA(str_exp.c_str(), &fd);
	if (hFind == INVALID_HANDLE_VALUE) {
		return paths;
	}
	do {
		std::string path = fd.cFileName;
		if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) { //if this is file
			std::string path_ext = path.substr(path.find_last_of('.') + 1, path.length());  //파일의 확장자 추출
			std::transform(path_ext.begin(), path_ext.end(), path_ext.begin(), toupper);
			int i = -1;
			while (++i < (int)allow_ext.size() && allow_ext[i] != path_ext);
			if (i < (int)allow_ext.size() || allow_ext.front() == "*") {    //allow_ext에 포함되어있으면
				paths.push_back(dir_path + path);
			}
		} else if (recursive == true && path != "." && path != "..") {
			std::vector<std::string> temps = FileList(dir_path + path, ext, recursive);
			for (auto&temp : temps) {
				paths.push_back(temp);
			}
		}
	} while (::FindNextFileA(hFind, &fd));
	::FindClose(hFind);
	return paths;   //RVO
}
#endif
#ifdef __linux__
std::vector<std::string> FileList(std::string dir_path, std::string ext, bool recursive/* = false*/) {
	std::vector<std::string> paths; //return value
	if (dir_path.back() != '/' && dir_path.back() != '\\') {
		dir_path.push_back('/');
	}
	std::string str_exp = dir_path + "*.*";
	std::vector<std::string> allow_ext;
	std::string::size_type offset = 0;
	while (offset < ext.length()) {
		std::string str = ext.substr(offset, ext.find(';', offset) - offset);
		std::transform(str.begin(), str.end(), str.begin(), toupper);
		offset += str.length() + 1;
		std::string::size_type pos = str.find_last_of('.');
		pos = pos == std::string::npos ? 0 : pos + 1;
		allow_ext.push_back(str.substr(pos, str.length()));
	}
	DIR* fd = opendir(dir_path.c_str());
	if (fd == NULL) {
		return paths;
	}
	struct dirent* hFind = NULL;
	while (hFind = readdir(fd)) {
		std::string path = hFind->d_name;
		if (hFind->d_type == DT_REG) {    //is File?
			std::string path_ext = path.substr(path.find_last_of('.') + 1, path.length());  //파일의 확장자 추출
			std::transform(path_ext.begin(), path_ext.end(), path_ext.begin(), toupper);
			int i = -1;
			while (++i < (int)allow_ext.size() && allow_ext[i] != path_ext);
			if (i < (int)allow_ext.size() || allow_ext.front() == "*") {    //allow_ext에 포함되어있으면
				paths.push_back(dir_path + path);
			}
		} else if (recursive == true && path != "." && path != "..") {	//is Directory?
			std::vector<std::string> temps = FileList(dir_path + path, ext, recursive);
			for (auto&temp : temps) {
				paths.push_back(temp);
			}
		}
	}
	closedir(fd);
	return paths;   //RVO
}
#endif
std::vector<ImgPathWithBBox> ParseImageListFromXML(std::string xml) {
	std::vector<ImgPathWithBBox> ret;
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml.c_str());
	tinyxml2::XMLNode* node_root = doc.FirstChild()->NextSiblingElement("dataset");
	tinyxml2::XMLElement* node_images = node_root->FirstChildElement("images");

	for (auto it = node_images->FirstChildElement("image"); it != NULL; it=it->NextSiblingElement("image")) {
		ImgPathWithBBox e;
		e.path = it->Attribute("file");
		//std::cout << it->Attribute("file") << std::endl;

		dlib::mmod_rect rect;
		for (auto jt = it->FirstChildElement("box"); jt != NULL; jt = jt->NextSiblingElement("box")) {
			//std::cout << jt->Attribute("top") << ", " << jt->Attribute("left") << ", " << jt->Attribute("width") << ", " << jt->Attribute("height") << std::endl;
			rect.rect.top() = std::atoi(jt->Attribute("top"));
			rect.rect.left() = std::atoi(jt->Attribute("left"));
			rect.rect.right() = rect.rect.left()+std::atoi(jt->Attribute("width"));
			rect.rect.bottom() = rect.rect.top()+std::atoi(jt->Attribute("height"));
			if (jt->Attribute("ignore") != NULL) {
				rect.ignore = true;
			}
			e.rects.push_back(rect);
		}
		ret.push_back(e);
	}
	return ret;
}
std::vector<dlib::matrix<dlib::rgb_pixel>> images_train, images_test;
std::vector<std::vector<dlib::mmod_rect>> boxes_train, boxes_test;
std::vector<dlib::matrix<dlib::rgb_pixel>> impl_mini_batch_samples[2], *mini_batch_samples;
std::vector<std::vector<dlib::mmod_rect>> impl_mini_batch_labels[2], *mini_batch_labels;
dlib::random_cropper cropper;
int mini_batch_size;
std::vector<ImgPathWithBBox> train_data;
dlib::rand rnd;
bool init = true;
int toggle = 0;
std::string solver = "sgd";
void thread_load_images(void* param) {
#pragma omp parallel for
	for (int i = 0; i < mini_batch_size; i++) {
		int idx = rand() % images_train.size();
		load_image(images_train[i], train_data[idx].path);
		boxes_train[i] = train_data[idx].rects;
	}
	cropper(mini_batch_size, images_train, boxes_train, *mini_batch_samples, *mini_batch_labels);
	for (auto&& img : *mini_batch_samples) disturb_colors(img, rnd);

	dlib::auto_mutex locker(count_mutex);
	count_signaler.signal();
}
void thread_run_trainer(void* param) {
	if (init == false) {
		if (solver == "sgd") {
			dlib::dnn_trainer<net_type,dlib::sgd>* trainer_ptr = reinterpret_cast<dlib::dnn_trainer<net_type, dlib::sgd>*>(param);
			trainer_ptr->train_one_step(impl_mini_batch_samples[!toggle], impl_mini_batch_labels[!toggle]);
			std::cerr << "iteration : " << trainer_ptr->get_train_one_step_calls() << ", \tloss : " << trainer_ptr->get_average_loss() << ", \tlr : " << trainer_ptr->get_learning_rate() << std::endl;
		} else if (solver == "adam") {
			dlib::dnn_trainer<net_type, dlib::adam>* trainer_ptr = reinterpret_cast<dlib::dnn_trainer<net_type, dlib::adam>*>(param);
			trainer_ptr->train_one_step(impl_mini_batch_samples[!toggle], impl_mini_batch_labels[!toggle]);
			std::cerr << "iteration : " << trainer_ptr->get_train_one_step_calls() << ", \tloss : " << trainer_ptr->get_average_loss() << ", \tlr : " << trainer_ptr->get_learning_rate() << std::endl;
		}
	}
	dlib::auto_mutex locker(count_mutex);
	count_signaler.signal();
}
int main(int argc,const char* argv[]) try{
	
	if (argc < 9) {
		std::cerr << "Arguments : [directory] [xml] [network name] [mini batch size] [min detection width=40] [min detection height=40] [solver=sgd] [cropper=350] [lr=0.0001]" << std::endl;
		return 1;
	}
	const std::string data_directory = argv[1];
	const std::string xml = argv[2];
	const std::string name = argv[3];
    mini_batch_size=std::atoi(argv[4]);

	int min_detection_width = 40;
	if (argc >= 6) {
		min_detection_width = std::atoi(argv[5]);
	}
	int min_detection_height = 40;
	if (argc >= 7) {
		min_detection_height = std::atoi(argv[6]);
	}
	int min_obj_sz = std::min(min_detection_width, min_detection_height);
	if (argc >= 8) {
		std::string solver = argv[7];
		if (solver != "sgd" || solver != "adam") {
			std::cerr << "@warning!! Support sgd and adam solver only" << std::endl;
			solver = "sgd";
		}
	}
	int cropper_size = 350;
	if (argc >= 9) {
		cropper_size = std::atoi(argv[8]);
	}
	float lr = 0.0001;
    if(argc>=10){
        lr=std::atof(argv[9]);
    }
	
	train_data = ParseImageListFromXML(data_directory + "/" + xml);
	//auto test_data  = ParseImageListFromXML(data_directory + "/testing.xml");
	for (auto&e : train_data) {
		e.path = data_directory + "/" + e.path;
		boxes_train.push_back(e.rects);
	}
	//load_image_dataset(images_train, boxes_train, data_directory + "/training.xml");
	//load_image_dataset(images_test, boxes_test, data_directory + "/testing.xml");


	dlib::mmod_options options(boxes_train, min_detection_height, min_detection_width);



	options.overlaps_ignore = dlib::test_box_overlap(0.5, 0.95);

	net_type net(options);


	net.subnet().layer_details().set_num_filters(options.detector_windows.size());
	const std::string sync_filename = name + "_sync";
	cropper.set_seed(time(0));
	cropper.set_chip_dims(cropper_size, cropper_size);
	// Usually you want to give the cropper whatever min sizes you passed to the
	// mmod_options constructor, or very slightly smaller sizes, which is what we do here.
	cropper.set_min_object_size(min_obj_sz, min_obj_sz);
	cropper.set_max_rotation_degrees(2);

	if (solver == "sgd") {
		dlib::dnn_trainer<net_type,dlib::sgd> trainer(net, dlib::sgd());
		trainer.set_learning_rate(0.1);
		trainer.be_verbose();
		trainer.set_iterations_without_progress_threshold(10000);
		trainer.set_test_iterations_without_progress_threshold(1000);
		trainer.set_synchronization_file(sync_filename, std::chrono::minutes(5));

		// Log the training parameters to the console
		std::cout << trainer << cropper << std::endl;

		int iter = 1;
		// Run the trainer until the learning rate gets small.
		std::cout.setstate(std::ios_base::failbit);
		images_train.assign(mini_batch_size, dlib::matrix<dlib::rgb_pixel>());
		boxes_train.assign(mini_batch_size, std::vector<dlib::mmod_rect>());
		while (trainer.get_learning_rate() >= lr) {
			toggle = !toggle;
			mini_batch_samples = &impl_mini_batch_samples[toggle];
			mini_batch_labels = &impl_mini_batch_labels[toggle];

			dlib::create_new_thread(thread_load_images, 0);
			dlib::create_new_thread(thread_run_trainer, &trainer);
			dlib::auto_mutex mtx(count_mutex);
			count_signaler.wait();
			count_signaler.wait();

			init = false;
			++iter;
		}
		// wait for training threads to stop
		trainer.get_net();
		std::cout << "done training" << std::endl;
	} else if (solver == "adam") {
		dlib::dnn_trainer<net_type,dlib::adam> trainer(net, dlib::adam());
		trainer.set_learning_rate(0.1);
		trainer.be_verbose();
		trainer.set_iterations_without_progress_threshold(10000);
		trainer.set_test_iterations_without_progress_threshold(1000);
		trainer.set_synchronization_file(sync_filename, std::chrono::minutes(5));

		// Log the training parameters to the console
		std::cout << trainer << cropper << std::endl;

		int iter = 1;
		// Run the trainer until the learning rate gets small.
		std::cout.setstate(std::ios_base::failbit);
		images_train.assign(mini_batch_size, dlib::matrix<dlib::rgb_pixel>());
		boxes_train.assign(mini_batch_size, std::vector<dlib::mmod_rect>());
		while (trainer.get_learning_rate() >= lr) {
			toggle = !toggle;
			mini_batch_samples = &impl_mini_batch_samples[toggle];
			mini_batch_labels = &impl_mini_batch_labels[toggle];

			dlib::create_new_thread(thread_load_images, 0);
			dlib::create_new_thread(thread_run_trainer, &trainer);
			dlib::auto_mutex mtx(count_mutex);
			count_signaler.wait();
			count_signaler.wait();

			init = false;
			++iter;
		}
		// wait for training threads to stop
		trainer.get_net();
		std::cout << "done training" << std::endl;
	}

	

	// Save the network to disk
	net.clean();
	dlib::serialize(name+"(final).dat") << net;
	return 0;
}catch(std::exception& e){
	std::cerr << e.what() << std::endl;
}
