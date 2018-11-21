//
// Created by kimbomm on 2018-11-20.
//
#include"MMODCNNSE_train.h"
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
int main(int argc,const char* argv[]) try{
	if (argc != 3) {
		std::cerr << "Arguments : [directory] [xml]" << std::endl;
		return 1;
	}
	const std::string data_directory = argv[1];
	const std::string xml = argv[2];

	std::vector<dlib::matrix<dlib::rgb_pixel>> images_train, images_test;

	std::vector<std::vector<dlib::mmod_rect>> boxes_train, boxes_test;
	auto train_data = ParseImageListFromXML(data_directory + "/" + xml);
	//auto test_data  = ParseImageListFromXML(data_directory + "/testing.xml");
	for (auto&e : train_data) {
		e.path = data_directory + "/" + e.path;
		boxes_train.push_back(e.rects);
	}
	//load_image_dataset(images_train, boxes_train, data_directory + "/training.xml");
	//load_image_dataset(images_test, boxes_test, data_directory + "/testing.xml");


	dlib::mmod_options options(boxes_train, 70, 30);



	options.overlaps_ignore = dlib::test_box_overlap(0.5, 0.95);

	net_type net(options);


	net.subnet().layer_details().set_num_filters(options.detector_windows.size());

	dlib::dnn_trainer<net_type> trainer(net, dlib::sgd(0.0001, 0.9));
	trainer.set_learning_rate(0.1);
	trainer.be_verbose();

	trainer.set_iterations_without_progress_threshold(10000);
	trainer.set_test_iterations_without_progress_threshold(1000);

	const std::string sync_filename = "mmod_cars_sync";
	trainer.set_synchronization_file(sync_filename, std::chrono::minutes(5));


	std::vector<dlib::matrix<dlib::rgb_pixel>> mini_batch_samples;
	std::vector<std::vector<dlib::mmod_rect>> mini_batch_labels;
	dlib::random_cropper cropper;
	cropper.set_seed(time(0));
	cropper.set_chip_dims(350, 350);
	// Usually you want to give the cropper whatever min sizes you passed to the
	// mmod_options constructor, or very slightly smaller sizes, which is what we do here.
	cropper.set_min_object_size(32, 32);
	cropper.set_max_rotation_degrees(2);
	dlib::rand rnd;

	// Log the training parameters to the console
	std::cout << trainer << cropper << std::endl;
	const int mini_batch_size = 32;

	int iter = 1;
	// Run the trainer until the learning rate gets small.
	while (trainer.get_learning_rate() >= 1e-4) {

		std::cout << "iteration : " << trainer.get_train_one_step_calls() << ", \tloss : "  << trainer.get_average_loss() << ", \tlr : " << trainer.get_learning_rate() << std::endl;

		std::random_shuffle(train_data.begin(), train_data.end());
		images_train.assign(mini_batch_size, dlib::matrix<dlib::rgb_pixel>());
		boxes_train.assign(mini_batch_size, std::vector<dlib::mmod_rect>());
		for (int i = 0; i < mini_batch_size; i++) {
			load_image(images_train[i], train_data[i].path);
			boxes_train[i] = train_data[i].rects;
		}
		cropper(mini_batch_size, images_train, boxes_train, mini_batch_samples, mini_batch_labels);
		for (auto&& img : mini_batch_samples)
			disturb_colors(img, rnd);
		trainer.train_one_step(mini_batch_samples, mini_batch_labels);
		++iter;
	}
	// wait for training threads to stop
	trainer.get_net();
	std::cout << "done training" << std::endl;

	// Save the network to disk
	net.clean();
	dlib::serialize("mmod_rear_end_vehicle_detector.dat") << net;
	return 0;
}catch(std::exception& e){
	std::cerr << e.what() << std::endl;
}
