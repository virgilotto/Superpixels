#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <string>

using namespace std;
using namespace cv;

int max(int x, int y) {
	if (x > y) {
		return x;
	}
	else {
		return y;
	}
}

int min(int x, int y) {
	if (x > y) {
		return y;
	}
	else {
		return x;
	}
}

float dist(Point p1, Point p2, Mat img_lab, float m, float S) {

	Vec3f p1_lab = img_lab.at<Vec3f>(p1);
	Vec3f p2_lab = img_lab.at<Vec3f>(p2);

	float dl = p1_lab[0] - p2_lab[0];
	float da = p1_lab[1] - p2_lab[1];
	float db = p1_lab[2] - p2_lab[2];

	float d_lab = sqrt(dl*dl + da*da + db*db);

	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;

	float d_xy = sqrt(dx*dx + dy*dy);

	return d_lab + m / S * d_xy;
}

void viewSuperpixels(Mat img, Mat labels) {

	Mat boundaries(img.rows, img.cols, CV_8UC1);
	boundaries = cv::Scalar(255, 255, 255);

	for (int i = 0; i < img.rows - 1; i++) {
		for (int j = 0; j < img.cols - 1; j++) {
			if (labels.at<int>(i, j) != labels.at<int>(i + 1, j + 1)) {
				img.at<Vec3b>(i, j) = 0;
				boundaries.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("Final", img);
	imshow("Boundaries", boundaries);
	imwrite("final.png", img);
	waitKey(0);
}

void calculateSuperpixels(Mat img, float m, int nx, int ny, float dx, float dy, float S) {

	Mat img_f;
	normalize(img, img_f, 0, 1, NORM_MINMAX, CV_32F);
	Mat img_lab;
	cvtColor(img_f, img_lab, CV_BGR2Lab);

	vector<Point> centers;
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			centers.push_back(Point2f(j*dx + dx / 2, i*dy + dy / 2));
		}
	}

	int n = nx * ny;
	int w = img.cols;
	int h = img.rows;

	vector<int> label_vec(n);
	for (int i = 0; i < n; i++) {
		label_vec[i] = i;
	}

	Mat labels = Mat::ones(img_lab.size(), CV_32S);
	Mat dists = Mat::ones(img_lab.size(), CV_32F);
	Mat centerStep;
	Point2i p1, p2;
	Vec3f p1_lab, p2_lab;

	for (int i = 0; i < 10; i++) {
		for (int c = 0; c < n; c++){
			int label = label_vec[c];
			p1 = centers[c];
 
			int xmin = max(p1.x - S, 0);
			int ymin = max(p1.y - S, 0);
			int xmax = min(p1.x + S, w);
			int ymax = min(p1.y + S, h);

			centerStep = img_lab(Range(ymin, ymax), Range(xmin, xmax));

			for (int i = 0; i < centerStep.rows; i++) {
				for (int j = 0; j < centerStep.cols; j++) {
					p2 = Point2i(xmin + j, ymin + i);
					float d = dist(p1, p2, img_lab, m, S);
					float last_d = dists.at<float>(p2);
					if (d < last_d || last_d == 1) {
						dists.at<float>(p2) = d;
						labels.at<int>(p2) = label;
					}
				}
			}
		}
	}

	viewSuperpixels(img, labels);
	
	/*labels.convertTo(labels, CV_32F);

	// ?
	const Mat sobel = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat show;
	Mat gx, gy, grad;

	filter2D(labels, gx, -1, sobel);
	filter2D(labels, gy, -1, sobel.t());
	magnitude(gx, gy, grad);
	show = 1 - grad;

	imshow("Boundaries", show);
	viewSuperpixels(img_f, show);*/
}

void superpixel(Mat img) {

	float m = 40.0;
	int nx = 8;
	int ny = 8;
	float dx = img.cols / float(nx);
	float dy = img.rows / float(ny);
	float S = sqrt(dx * dy);

	calculateSuperpixels(img, m, nx, ny, dx, dy, S);
}

int main() {

	Mat img = imread("Images/brain.png");
	imshow("Original", img);
	superpixel(img);
	return 0;
}