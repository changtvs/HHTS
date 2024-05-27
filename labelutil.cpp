// Copyright (c) Technische Hochschule Nürnberg, Game Tech Lab.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "labelutil.h"

Vec3b getRandomColor()
{
    int b = theRNG().uniform(0, 256);
    int g = theRNG().uniform(0, 256);
    int r = theRNG().uniform(0, 256);

    return Vec3b((uchar)b, (uchar)g, (uchar)r);
}

// zero label always black
Mat getColoredLabels(InputArray inputLabels)
{
    Mat labels = inputLabels.getMat();

    double maxLabel;
    minMaxLoc(labels, nullptr, &maxLabel, nullptr, nullptr);

    // build label images
    Mat labelImage = Mat::zeros(labels.size(), CV_8UC3);
    for (int label = 1; label <= maxLabel; ++label)
    {
        labelImage.setTo(getRandomColor(), (labels == label));
    }

    return labelImage;
}

Mat getColoredLabels(InputArray inputLabels, InputArray inputImage)
{
    Mat labels = inputLabels.getMat();
    Mat image = inputImage.getMat();

    double maxLabel;
    minMaxLoc(labels, nullptr, &maxLabel, nullptr, nullptr);

    // build label images
    Mat labelImage = Mat::zeros(labels.size(), CV_8UC3);
    for (int label = 1; label <= maxLabel; ++label)
    {
        Mat labelMask = (labels == label);
        Scalar meanColor = mean(image, labelMask);
        labelImage.setTo(meanColor, labelMask);
    }

    return labelImage;
}