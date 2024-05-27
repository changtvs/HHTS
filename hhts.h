// Copyright (c) Technische Hochschule Nürnberg, Game Tech Lab.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _HHTS_
#define _HHTS_

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "labelutil.h"

using namespace cv;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
using namespace std;

namespace HHTS
{
    enum ColorChannel
    {
        RGB = 1,
        HSV = 2,
        LAB = 4
    };

    struct SplitParams
    {
        vector<int> superpixels;
        double splitThreshold;
        int histogramBins;
        int minSegmentSize;

    public:
        SplitParams(const vector<int> &superpixels = {}, const double splitThreshold = 0.0, const int histogramBins = 16, const int minSegmentSize = 64) : superpixels(superpixels), splitThreshold(splitThreshold), histogramBins(histogramBins), minSegmentSize(minSegmentSize) {}
    };

    struct ChannelInfo
    {
        int min, max, width;
        double splitCriteria;
        ChannelInfo(const InputArray channel, const InputArray mask, const int size);
    };

    struct Label
    {
        int id;

        Mat mask;

        vector<ChannelInfo> channelInfos;
        double labelSplitCriteria;
        int labelSplitChannel;

        int labelSize;
        int childMinSize;

    public:
        Label(const InputArrayOfArrays channels, const InputArray mask, const int labelSize, const int id, const SplitParams &splitParams);
        bool isSizeSplittable() const { return labelSize / 2 >= childMinSize; }
        bool isSplittable(const double splitThreshold) const { return labelSplitCriteria > splitThreshold && isSizeSplittable(); }
        void split(const InputArrayOfArrays channels, const InputOutputArray inputOutputLabels,
                   const Size &size,
                   int &nextLabel, vector<Label> &splittableLabels,
                   const SplitParams &splitParams);
        static bool compare(const Label label0, const Label label1) { return label0.labelSplitCriteria > label1.labelSplitCriteria; }

    private:
        void interruptSplit(const InputOutputArray inputOutputLabels, vector<Label> &splittableLabels, const SplitParams &splitParams);
    };

    // returns label count
    int hhts(const InputArray image, const OutputArray outputLabels, const int superpixels, const double splitThreshold = 0.0, const int histogramBins = 16, const int minSegmentSize = 64, const int colorChannels = RGB | HSV | LAB,
                    const bool applyBlur = false, const InputArray preLabels = noArray());

    vector<int> hhts(const InputArray image, const OutputArrayOfArrays outputLabels,
                    const vector<int> &superpixels = {}, const double splitThreshold = 0.0, const int histogramBins = 16, const int minSegmentSize = 64, const int colorChannels = RGB | HSV | LAB,
                    const bool applyBlur = false, const InputArray preLabels = noArray());
}

#endif /* _HHTS_ */
