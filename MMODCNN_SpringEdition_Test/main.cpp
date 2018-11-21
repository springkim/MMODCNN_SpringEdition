#include"MMODCNNSE.h"
int main() {
	MMODCNN detector;
	detector.Create("vehicle.dat", "names.txt");

	cv::VideoCapture vc;
	vc.open("traffic.mp4");
	for (cv::Mat frame; vc.read(frame);) {
		double t1 = (double)cv::getTickCount();
		auto boxes = detector.Detect(frame, 0.0F);
		double t2 = (double)cv::getTickCount();
		std::cout << "time:" << (t2 - t1) * 1000 / cv::getTickFrequency() << std::endl;
		for (auto box : boxes) {
			if (box.m_class == 0) {
				cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1, CV_AA);
			} else {
				cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 1, CV_AA);
			}
		}
		cv::imshow("img", frame);
		if (cv::waitKey(10) == 27)break;
	}


	cv::destroyAllWindows();
	return 0;
}