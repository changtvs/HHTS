// Copyright (c) Technische Hochschule Nürnberg, Game Tech Lab.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <opencv2/core.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

#include "hhts.h"

using namespace cv;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
using namespace std;

void testSingleLevel()
{
    string imagePath = "247012.jpg";
    int spCount = 500;
    int minDetailSize = 64;

    Mat image = imread(imagePath, IMREAD_COLOR);
    Mat labels;
    InputArray mask = noArray();

    boost::timer::cpu_timer timer;

    int labelCount = HHTS::hhts(image, labels, spCount, 0.0, 32, minDetailSize, HHTS::ColorChannel::RGB | HHTS::ColorChannel::LAB | HHTS::ColorChannel::HSV, false, noArray());

    boost::chrono::duration<double> secondsWall = boost::chrono::nanoseconds(timer.elapsed().wall);
    double elapsedWall = secondsWall.count();
    cout << elapsedWall << endl;

    imshow("mean labels " + to_string(labelCount), getColoredLabels(labels, image));
    imshow("random labels " + to_string(labelCount), getColoredLabels(labels));
}

void testMultiLevel()
{
    string imagePath = "247012.jpg";
    vector<int> spCounts{100, 250, 500};
    int minDetailSize = 64;

    Mat image = imread(imagePath, IMREAD_COLOR);
    vector<Mat> labels;
    InputArray mask = noArray();

    boost::timer::cpu_timer timer;

    vector<int> labelCounts = HHTS::hhts(image, labels, spCounts, 0.0, 32, minDetailSize, HHTS::ColorChannel::RGB | HHTS::ColorChannel::LAB | HHTS::ColorChannel::HSV, false, noArray());

    boost::chrono::duration<double> secondsWall = boost::chrono::nanoseconds(timer.elapsed().wall);
    double elapsedWall = secondsWall.count();
    cout << elapsedWall << endl;

    for (int i = 0; i < labels.size(); ++i)
    {
        imshow("mean labels " + to_string(labelCounts[i]), getColoredLabels(labels[i], image));
        imshow("random labels " + to_string(labelCounts[i]), getColoredLabels(labels[i]));
    }
}

void testAutotermination()
{
    string imagePath = "247012.jpg";
    vector<int> spCounts{500, -1};
    int minDetailSize = 64;

    Mat image = imread(imagePath, IMREAD_COLOR);
    vector<Mat> labels;
    InputArray mask = noArray();

    boost::timer::cpu_timer timer;

    vector<int> labelCounts = HHTS::hhts(image, labels, spCounts, 0.0, 32, minDetailSize, HHTS::ColorChannel::RGB | HHTS::ColorChannel::LAB | HHTS::ColorChannel::HSV, false, noArray());

    boost::chrono::duration<double> secondsWall = boost::chrono::nanoseconds(timer.elapsed().wall);
    double elapsedWall = secondsWall.count();
    cout << elapsedWall << endl;

    for (int i = 0; i < labels.size(); ++i)
    {
        imshow("mean labels " + to_string(labelCounts[i]), getColoredLabels(labels[i], image));
        imshow("random labels " + to_string(labelCounts[i]), getColoredLabels(labels[i]));
    }
}

int main(int argc, char *argv[])
{
    testSingleLevel();
    // testMultiLevel();
    // testAutotermination();
    waitKey();
}