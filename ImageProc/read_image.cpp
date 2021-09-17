
#include <stdio.h> 
#include <iostream> 
#include <cmath>
#include<opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

struct moments_data {
	double theta;
	int px;
	int py;

}mom;

struct mass_center {
	int x;
	int y;
};

void show_image(string img_path, string window_name) {
	Mat image = imread(img_path);
	imshow(window_name, image);
	waitKey(3000);
}

double reduced_central_moment(int p, int q, mass_center center, Mat img) {
	double moment = 0;
	for (double i = 0; i < img.cols; i++) {
		for (double j = 0; j < img.rows; j++) {
			if (img.at<uchar>(j, i) != 0){
			moment += pow((i - center.x), p)*pow((j - center.y), q);
			}
		}
	}
	return moment;
}

void principal_axis(Mat img, mass_center center) {
	double moment_0_0 = reduced_central_moment(0, 0, center, img);
	cout << moment_0_0 << endl;

	double moment_2_0 = reduced_central_moment(2, 0, center, img)/moment_0_0;
	double moment_0_2 = reduced_central_moment(0, 2, center, img) / moment_0_0;
	double moment_1_1 = reduced_central_moment(1, 1, center, img) / moment_0_0;
	Moments m = moments(img, true);

	//mom.theta = atan((2*m.m11) / (m.m20 -m.m02))/2;
	mom.theta = atan((2 * moment_1_1) / (moment_2_0 - moment_0_2)) / 2;
	cout << "Principal Angle: " << mom.theta * (180.0 / 3.141592653589793238463) << endl;

	mom.px = center.x + img.cols / 4 * cos(mom.theta);
	mom.py = center.y + img.rows / 4 * sin(mom.theta);


}


Mat image_threshold(Mat img, int thresh) {
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			if (img.at<uchar>(j, i) > thresh) {
				img.at<uchar>(j, i) = 0;
			}
			else {
				img.at<uchar>(j, i) = 255;
			}

		}
	}
	return img;
}


mass_center get_center_of_Mass(Mat img) {
	int sumx = 0, sumy= 0, black_pix = 0;

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			if (img.at<uchar>(j,i) == 0) {
				sumx += i;
				sumy += j;
				black_pix += 1;
			}

		}
	}
	sumx /= black_pix;
	sumy /= black_pix;
	mass_center result = { sumx,sumy };
	return result;
}



void test_center_of_mass() {
	Mat img = imread("Resource/PEN.pgm");

	rotate(img, img, ROTATE_90_CLOCKWISE);
	cvtColor(img, img, COLOR_BGR2GRAY);
	img = image_threshold(img,100);
	mass_center result = get_center_of_Mass(img);
	principal_axis(img, result);
	cvtColor(img, img, COLOR_GRAY2BGR);
	line(img, Point(mom.px,mom.py), Point(result.x,result.y), Scalar(0, 0, 255),
		2, LINE_4);
	drawMarker(img, Point(result.x, result.y), Scalar(0, 0, 255), MARKER_CROSS, 20, 2);
	imshow("original", img);
	waitKey(0);
}


Mat create_hist(Mat image, bool greyscale) {
	while (true) {
		//https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
		if (greyscale==false) {
			vector<Mat> bgr_planes;
			split(image, bgr_planes);
			int histSize = 256;
			float range[] = { 0, 256 }; //the upper boundary is exclusive
			const float* histRange[] = { range };
			bool uniform = true, accumulate = false;
			Mat b_hist, g_hist, r_hist;
			calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
			calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
			calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
			int hist_w = 512, hist_h = 400;
			int bin_w = cvRound((double)hist_w / histSize);
			Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
			normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			for (int i = 1; i < histSize; i++)
			{
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
					Scalar(255, 0, 0), 2, 8, 0);
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
					Scalar(0, 255, 0), 2, 8, 0);
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
					Scalar(0, 0, 255), 2, 8, 0);
			}
			return histImage;
		}
		else {
			Mat g_plane;
			cvtColor(image, g_plane, COLOR_BGR2GRAY);
			//split(image, bgr_planes);
			int histSize = 256;
			float range[] = { 0, 256 }; //the upper boundary is exclusive
			const float* histRange[] = { range };
			bool uniform = true, accumulate = false;
			Mat g_hist;
			calcHist(&g_plane, 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
			
			int hist_w = 512, hist_h = 400;
			int bin_w = cvRound((double)hist_w / histSize);
			Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
			normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			for (int i = 1; i < histSize; i++)
			{
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
					Scalar(255, 0, 0), 2, 8, 0);
			}
			return histImage;
		}
	}
}



Mat take_photo(bool greyscale, bool canny=false, int thresh=0) {
	cv::VideoCapture camera(0);
	if (!camera.isOpened()) {
		std::cerr << "ERROR: Could not open camera" << std::endl;
	}

	namedWindow("Webcam", 512);
	cout << "Taking photo with webcam";
	Mat frame, frame_canny, histImage;
	namedWindow("Video Stream");
	createTrackbar("Binary Threshold", "Video Stream", &thresh, 255);
	while (true) {
		camera >> frame;
		histImage = create_hist(frame, greyscale);

		if (canny){
			Canny(frame, frame_canny, 25, 75);
			imshow("Canny Edge Detector", frame_canny);
		}
		if (greyscale) {
			cvtColor(frame, frame, COLOR_BGR2GRAY);
			if (thresh > 0) {
				frame = image_threshold(frame, thresh);
				mass_center result = get_center_of_Mass(frame);
				cvtColor(frame, frame, COLOR_GRAY2BGR);
				drawMarker(frame, Point(result.x, result.y), Scalar(0, 0, 255), MARKER_CROSS, 20, 2);

			}

		}

		
		imshow("Video Stream", frame);
		imshow("Histogram of Video Capture", histImage);
		if (waitKey(1) >= 0) {
			destroyAllWindows();
			break;
		}
	}
	if (frame.empty())
	{
		cerr << "Something is wrong with the webcam, could not get frame." << endl;
	}
	return frame;
}

void save_image(Mat image, string img_path, string image_save_format) {
	string image_path_wo_extention = img_path.substr(0,img_path.find("."));
	imwrite(image_path_wo_extention + image_save_format, image);
	cout << "Image saved to: " << image_path_wo_extention << " as: " << image_save_format << endl;
}

void print_image_attributes(string img_path, string window_name, string file_type) {
	Mat image = imread(img_path);
	cout << "file name: " << window_name << endl;
	cout << "file size: " << image.size << endl;
	cout << "image type. " << file_type << endl;
	if (image.channels() == 3) {
		cout << "Color format is BGR" << endl;
	}
}

int main()
{
	const string Resource_path = "Resource/";
	string image_save_format = ".png";
	string img_path = Resource_path + "webcam_test.jpg";
	string window_name = "test";
	string delimiter = ".";
	string file_type = img_path.substr(img_path.find(delimiter) + 1, img_path.length());

	//show_image(img_path, window_name);
	//print_image_attributes(img_path, window_name, file_type);
	//Mat image = take_photo(true, false, 100);
	//save_image(image, img_path, image_save_format);
	test_center_of_mass();
	return 0;
}

