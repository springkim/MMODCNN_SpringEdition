
#ifdef _DEBUG
#pragma comment(lib,"dlib19.16.0_debug_64bit_msvc1900.lib")
#else
#pragma comment(lib,"dlib19.16.0_release_64bit_msvc1900.lib")
#endif

#pragma comment(lib,"cuda.lib")
#pragma comment(lib,"curand.lib")
#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cudnn.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"cusolver.lib")
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
This example shows how to train a CNN based object detector using dlib's
loss_mmod loss layer.  This loss layer implements the Max-Margin Object
Detection loss as described in the paper:
Max-Margin Object Detection by Davis E. King (http://arxiv.org/abs/1502.00046).
This is the same loss used by the popular SVM+HOG object detector in dlib
(see fhog_object_detector_ex.cpp) except here we replace the HOG features
with a CNN and train the entire detector end-to-end.  This allows us to make
much more powerful detectors.

It would be a good idea to become familiar with dlib's DNN tooling before reading this
example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
before reading this example program.  You should also read the introductory DNN+MMOD
example dnn_mmod_ex.cpp as well before proceeding.


This example is essentially a more complex version of dnn_mmod_ex.cpp.  In it we train
a detector that finds the rear ends of motor vehicles.  I will also discuss some
aspects of data preparation useful when training this kind of detector.

*/


#include <iostream>
#include<algorithm>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include"filelist.h"
#include"tinyxml2.h"
using namespace std;
using namespace dlib;



template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<bn_con<con5<55, SUBNET>>>;
using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


// ----------------------------------------------------------------------------------------

int ignore_overlapped_boxes(std::vector<mmod_rect>& boxes,const test_box_overlap& overlaps){
	int num_ignored = 0;
	for (size_t i = 0; i < boxes.size(); ++i) {
		if (boxes[i].ignore)
			continue;
		for (size_t j = i + 1; j < boxes.size(); ++j) {
			if (boxes[j].ignore)
				continue;
			if (overlaps(boxes[i], boxes[j])) {
				++num_ignored;
				if (boxes[i].rect.area() < boxes[j].rect.area())
					boxes[i].ignore = true;
				else
					boxes[j].ignore = true;
			}
		}
	}
	return num_ignored;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try {
	if (argc != 3) {
		std::cerr << "This trainer require base directory and xml file." << std::endl;
		return 1;
	}
	const std::string data_directory = argv[1];
	const std::string xml = argv[2];

	std::vector<matrix<rgb_pixel>> images_train, images_test;
	
	std::vector<std::vector<mmod_rect>> boxes_train, boxes_test;
	auto train_data = ParseImageListFromXML(data_directory + "/" + xml);
	//auto test_data  = ParseImageListFromXML(data_directory + "/testing.xml");
	for (auto&e : train_data) {
		e.path = data_directory + "/" + e.path;
		boxes_train.push_back(e.rects);
	}
	//load_image_dataset(images_train, boxes_train, data_directory + "/training.xml");
	//load_image_dataset(images_test, boxes_test, data_directory + "/testing.xml");
	

	mmod_options options(boxes_train, 70, 30);

	

	
	options.overlaps_ignore = test_box_overlap(0.5, 0.95);

	net_type net(options);


	net.subnet().layer_details().set_num_filters(options.detector_windows.size());

	dnn_trainer<net_type> trainer(net, sgd(0.0001, 0.9));
	trainer.set_learning_rate(0.1);
	trainer.be_verbose();

	trainer.set_iterations_without_progress_threshold(10000);
	trainer.set_test_iterations_without_progress_threshold(1000);

	const string sync_filename = "mmod_cars_sync";
	trainer.set_synchronization_file(sync_filename, std::chrono::minutes(5));


	std::vector<matrix<rgb_pixel>> mini_batch_samples;
	std::vector<std::vector<mmod_rect>> mini_batch_labels;
	random_cropper cropper;
	cropper.set_seed(time(0));
	cropper.set_chip_dims(350, 350);
	// Usually you want to give the cropper whatever min sizes you passed to the
	// mmod_options constructor, or very slightly smaller sizes, which is what we do here.
	cropper.set_min_object_size(32, 32);
	cropper.set_max_rotation_degrees(2);
	dlib::rand rnd;

	// Log the training parameters to the console
	cout << trainer << cropper << endl;
	const int mini_batch_size = 32;

	int iter = 1;
	// Run the trainer until the learning rate gets small.  
	while (trainer.get_learning_rate() >= 1e-4) {
		
		std::cout << "iteration : " << trainer.get_train_one_step_calls() << ", \tloss : "  << trainer.get_average_loss() << ", \tlr : " << trainer.get_learning_rate() << std::endl;

		std::random_shuffle(train_data.begin(), train_data.end());
		images_train.assign(mini_batch_size, matrix<rgb_pixel>());
		boxes_train.assign(mini_batch_size, std::vector<mmod_rect>());
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
	cout << "done training" << endl;

	// Save the network to disk
	net.clean();
	serialize("mmod_rear_end_vehicle_detector.dat") << net;

	return 0;

} catch (std::exception& e) {
	cout << e.what() << endl;
}





