
#include <stdio.h> 
#include <iostream> 
#include <cmath>
#include<opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;


struct moments_data {
	double theta;
	double px;
	double py;

}mom;


struct mass_center {
	int x;
	int y;
};


Mat openingMorphological(Mat img)
{

	int morph_size = 2;
	Mat element = getStructuringElement(
		MORPH_RECT,
		Size(morph_size + 1,
			morph_size + 1),
		Point(morph_size,
			morph_size));
	Mat output;

	// Opening
	morphologyEx(img, img,
		MORPH_OPEN, element,
		Point(-1, -1), 2);
	return img;
}


void show_image(string img_path, string window_name) {
	Mat image = imread(img_path);
	imshow(window_name, image);
	waitKey(3000);
}


double reduced_central_moment(int p, int q, mass_center center, Mat img) {
	double moment = 0;
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			int pix_val = img.at<uchar>(j,i);
			if (pix_val == 255){
			moment += pow((i - center.x), p)*pow((j - center.y), q);
			}
		}
	}
	return moment;
}


double higher_moment(Mat img, mass_center center, int p, int q) {
	double moment_0_0 = reduced_central_moment(0, 0, center, img);
	double moment_p_q = reduced_central_moment(p, q, center, img);
	double higher_moment = moment_p_q / pow(moment_0_0, 1 + (p + q) / 2);
	return higher_moment;
}


double hue_moment(Mat img, mass_center center, int p, int q) {
	double eta_2_0 = higher_moment(img, center, p, q);
	double eta_0_2 = higher_moment(img, center, q, p);
	double hue_first = eta_0_2 + eta_2_0;
	return hue_first;
}


Mat pad_image(Mat img, int pad_width = 1, int pad_val = 0) {
	Mat img_padded = Mat(img.rows + pad_width * 2, img.cols + pad_width * 2,
		img.type(), Scalar(pad_val));

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			img_padded.at<uchar>(j + pad_width, i + pad_width) = img.at<uchar>(j, i);
		}
	}
	return img_padded;
}


void principal_axis(Mat img, mass_center center) {
	double moment_0_0 = reduced_central_moment(0, 0, center, img);

	double moment_2_0 = reduced_central_moment(2, 0, center, img)/moment_0_0;
	double moment_0_2 = reduced_central_moment(0, 2, center, img) / moment_0_0;
	double moment_1_1 = reduced_central_moment(1, 1, center, img) / moment_0_0;

	mom.theta =  atan2((2 * moment_1_1), (moment_2_0 - moment_0_2)) / 2;
	cout << "Principal Angle: " << mom.theta * (180.0 / 3.1415926535) << endl;

	mom.px = center.x + img.rows / 2 * cos(mom.theta);
	mom.py = center.y + img.rows / 2 * sin(mom.theta);

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
	int sumx = 0, sumy= 0, white_pix = 0;

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			if (img.at<uchar>(j,i) == 255) {
				sumx += i;
				sumy += j;
				white_pix += 1;
			}

		}
	}
	if (white_pix != 0) {
		sumx /= white_pix;
		sumy /= white_pix;
	}
	mass_center result = { sumx,sumy };
	return result;
}


void test_center_of_mass() {
	Mat img = imread("Resource/PEN.pgm");

	//Rect myROI(0, 0, 200, 200);
	//Mat img = img_new(myROI);
	Mat img_rotated;
	//img = pad_image(img, 1, 0);

	cvtColor(img, img, COLOR_BGR2GRAY);
	img = image_threshold(img,90);
	int k = 0;
	while (k < 1) {
		img = openingMorphological(img);
		k++;
	}

	mass_center result = get_center_of_Mass(img);
	principal_axis(img, result);
	rotate(img, img_rotated, ROTATE_90_CLOCKWISE);
	double hue_moment_1 = hue_moment(img, result, 2, 0);
	double hue_moment_rot = hue_moment(img_rotated, result, 2, 0);
	Moments mome = moments(img, false);
	double cv_hue_moment[7];
	HuMoments(mome, cv_hue_moment);

	cout << "Hue Moment: " << hue_moment_1 << endl << "Hue Moment rotated: " << hue_moment_rot << endl << "Opencv Hue Moments: "<< cv_hue_moment[0];
	cvtColor(img, img, COLOR_GRAY2BGR);
	line(img, Point(mom.px,mom.py), Point(result.x,result.y), Scalar(0, 0, 255),
		2, LINE_4);
	drawMarker(img, Point(result.x, result.y), Scalar(0, 0, 255), MARKER_CROSS, 20, 2);
	imshow("thresh", img);
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


Mat filter_func(Mat img, int n, int m, Mat filter) {
	// n: size of filter in horizontal direction
	// m: size of filter in vertical direction
	Mat img_filtered = img.clone();
	float scaling = 1 / pow(n + m + 1, 2);

	for (int i = n; i < img.cols-n; i++) {
		//cout << "i: " << i << endl;
		for (int j = m; j < img.rows-m; j++) {
			//cout << "j: " << j << endl;
			int values = 0;
			for (int p = i - n; p < i + n + 1; p++) {
				//cout << "p: " << p << endl;
				for (int k = j - m; k < j + m + 1; k++) {
					//cout << "k: " << k << endl;
					values += img.at<uchar>(k, p);
				}
			}
			img_filtered.at<uchar>(j, i) = scaling * values;
		}
	}
	return img_filtered;
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
	//test_center_of_mass();

	Mat img = imread(img_path);
	cvtColor(img, img, COLOR_BGR2GRAY);
	int n = 1;
	int m = 1;
	//float filter[] = { 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9 };
	Mat img_filtered = filter_func(img, n, m, filter);

	window_name = "original";
	imshow(window_name, img);
	window_name = "filtered";
	imshow(window_name, img_filtered);
	waitKey(0);

	return 0;
}

