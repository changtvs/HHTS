// Copyright (c) Technische Hochschule Nürnberg, Game Tech Lab.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _LABELUTIL_
#define _LABELUTIL_

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

Vec3b getRandomColor();
Mat getColoredLabels(InputArray inputLabels);
Mat getColoredLabels(InputArray inputLabels, InputArray inputImage);

#endif /* _LABELUTIL_ */
