// Copyright (c) Technische Hochschule Nürnberg, Game Tech Lab.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "hhts.h"

namespace HHTS
{
    void getChannels(InputArray image, int colorChannels, OutputArrayOfArrays outputChannels, bool applyBlur)
    {
        vector<Mat> channels;
        const int blurSize = 3;

        if ((colorChannels & RGB) > 0)
        {
            Mat img;
            Mat chs[3];
            image.copyTo(img);
            if (applyBlur)
            {
                GaussianBlur(img, img, Size(blurSize, blurSize), 0, 0);
            }
            split(img, chs);
            channels.push_back(chs[2]);
            channels.push_back(chs[1]);
            channels.push_back(chs[0]);
        }

        if ((colorChannels & HSV) > 0)
        {
            Mat img;
            Mat chs[3];
            cvtColor(image, img, COLOR_BGR2HSV);
            if (applyBlur)
            {
                GaussianBlur(img, img, Size(blurSize, blurSize), 0, 0);
            }
            split(img, chs);
            channels.push_back(chs[0]);
            channels.push_back(chs[1]);
            channels.push_back(chs[2]);
        }

        if ((colorChannels & LAB) > 0)
        {
            Mat img;
            Mat chs[3];
            cvtColor(image, img, COLOR_BGR2Lab);
            if (applyBlur)
            {
                GaussianBlur(img, img, Size(blurSize, blurSize), 0, 0);
            }
            split(img, chs);
            channels.push_back(chs[0]);
            channels.push_back(chs[1]);
            channels.push_back(chs[2]);
        }

        outputChannels.create(Size(channels.size(), 1), CV_8UC1);
        for (int iChannel = 0; iChannel < channels.size(); iChannel++)
        {
            channels[iChannel].copyTo(outputChannels.getMatRef(iChannel));
        }
    }

    ChannelInfo::ChannelInfo(InputArray channel, InputArray mask, int size)
    {
        // min, max, width
        double minVal, maxVal;
        minMaxLoc(channel, &minVal, &maxVal, nullptr, nullptr, mask);
        min = minVal;
        max = maxVal;
        width = max - min;

        // prevent small channels -> auto-termination
        if (width < 2)
        {
            splitCriteria = -1.0;
            return;
        }

        // mean, stdDev
        Scalar meanVal, stdDevVal;
        meanStdDev(channel, meanVal, stdDevVal, mask);
        const double mean = meanVal.val[0];
        const double stdDev = stdDevVal.val[0];

        splitCriteria = stdDev * size * size;
    }

    Label::Label(const InputArrayOfArrays channels, const InputArray inputMask, const int labelSize, const int id, const SplitParams &splitParams) : id(id), labelSize(labelSize)
    {
        mask = inputMask.getMat().clone();

        childMinSize = splitParams.minSegmentSize;

        if (!isSizeSplittable())
        {
            return;
        }

        labelSplitCriteria = -1.0;
        labelSplitChannel = -1;
        for (int iChannel = 0; iChannel < channels.total(); iChannel++)
        {
            const ChannelInfo channelInfo(channels.getMat(iChannel), mask, labelSize);
            channelInfos.push_back(channelInfo);
            if (channelInfo.splitCriteria > labelSplitCriteria)
            {
                labelSplitCriteria = channelInfo.splitCriteria;
                labelSplitChannel = iChannel;
            }
        }
    }

    int histogramBinToThreshold(const double bin, const double channelBins, const double minVal, const double maxVal)
    {
        const int threshold = minVal + 0.5 * (((maxVal - minVal + 1) * (2 * bin + 1) / channelBins) - 1);
        return threshold;
    }

    void getChannelThreshold(const InputArray inputChannel, const ChannelInfo &channelInfo, const InputArray mask, const SplitParams &splitParams, int &thresholdValue)
    {
        // calc hist
        const int channelBins = min(splitParams.histogramBins, channelInfo.width);
        const float range[] = {((float)channelInfo.min), ((float)channelInfo.max) + 1};
        const float *histRange[] = {range};

        Mat hist;
        Mat channel = inputChannel.getMat();
        calcHist(&channel, 1, 0, mask, hist, 1, &channelBins, histRange);
        channel.release();
        hist = hist.t();

        // get responses
        // --calculate high pass hist
        Mat responseHist;
        const Mat kernel = (Mat_<float>(1, 3) << 1, -2, 1); // 1D Laplacian kernel
        filter2D(hist, responseHist, -1, kernel, Point(-1, -1), 0.0, BORDER_REPLICATE);

        // --get weights to enforce balanced partitions (~50/50)
        Mat balancedPartitionWeights = Mat::zeros(hist.size(), CV_32FC1);
        balancedPartitionWeights.at<float>(0, 0) = hist.at<float>(0, 0);
        for (int b = 1; b < balancedPartitionWeights.size().width; ++b)
        {
            balancedPartitionWeights.at<float>(0, b) = balancedPartitionWeights.at<float>(0, b - 1) + hist.at<float>(0, b);
        }
        const int max = balancedPartitionWeights.at<float>(0, balancedPartitionWeights.size().width - 1);
        const int mean = max / 2;
        for (int b = 0; b < balancedPartitionWeights.size().width; ++b)
        {
            const float value = balancedPartitionWeights.at<float>(0, b);
            const float d = 2.0f;
            const float weight = 1 / (pow((mean - value) / mean * d, 4) + 1);
            balancedPartitionWeights.at<float>(0, b) = weight;
        }

        // --apply weights to response (weakens responses on histogram border)
        multiply(responseHist, balancedPartitionWeights, responseHist);

        // get threshold
        Point maxLoc;
        minMaxLoc(responseHist, nullptr, nullptr, nullptr, &maxLoc);
        const int thresholdBin = maxLoc.x;
        thresholdValue = histogramBinToThreshold(thresholdBin, channelBins, channelInfo.min, channelInfo.max);
    }

    void Label::interruptSplit(const InputOutputArray inputOutputLabels, vector<Label> &splittableLabels, const SplitParams &splitParams)
    {
        // invalidate current label split channel
        channelInfos[labelSplitChannel].splitCriteria = -1.0;

        // find new label split channel
        labelSplitCriteria = -1.0;
        labelSplitChannel = -1;
        for (int iChannelInfo = 0; iChannelInfo < channelInfos.size(); iChannelInfo++)
        {
            if (channelInfos[iChannelInfo].splitCriteria > labelSplitCriteria)
            {
                labelSplitCriteria = channelInfos[iChannelInfo].splitCriteria;
                labelSplitChannel = iChannelInfo;
            }
        }

        // remember label as splittable if label split criteria is valid
        if (isSplittable(splitParams.splitThreshold))
        {
            splittableLabels.insert(std::lower_bound(splittableLabels.begin(), splittableLabels.end(), *this, Label::compare), *this);
        }
        else
        {
            mask.release();
        }
    }

    void Label::split(const InputArrayOfArrays channels, const InputOutputArray inputOutputLabels,
                      const Size &size,
                      int &nextLabel, vector<Label> &splittableLabels,
                      const SplitParams &splitParams)
    {
        // get channel to split
        const Mat channel = channels.getMat(labelSplitChannel);

        int thresholdValue;
        getChannelThreshold(channel, channelInfos[labelSplitChannel], mask, splitParams, thresholdValue);

        // threshold and spacial split for low and high
        Mat rawFloodAreas = Mat::zeros(size, CV_8UC1);
        vector<Point> floodSeeds{};
        Mat ccLabels, ccStats, ccCentroids;
        int ccLabelCount;

        // --low
        // get threshold masks
        Mat lowMask;
        threshold(channel, lowMask, thresholdValue, 1, THRESH_BINARY_INV);
        bitwise_and(lowMask, mask, lowMask);

        const int CCL_Type = CCL_DEFAULT;
        // spacial low high split
        bool hasLowLabels = false;
        ccLabelCount = connectedComponentsWithStats(lowMask, ccLabels, ccStats, ccCentroids, 4, CV_32SC1, CCL_Type);
        lowMask.release();
        for (int ccLabel = 1; ccLabel < ccLabelCount; ++ccLabel)
        {
            const int area = ccStats.at<int>(ccLabel, CC_STAT_AREA);
            if (area < childMinSize)
            {
                rawFloodAreas.setTo(3, ccLabels == ccLabel);
                continue;
            }
            const int left = ccStats.at<int>(ccLabel, CC_STAT_LEFT);
            const int top = ccStats.at<int>(ccLabel, CC_STAT_TOP);
            const int width = ccStats.at<int>(ccLabel, CC_STAT_WIDTH);
            const int height = ccStats.at<int>(ccLabel, CC_STAT_HEIGHT);
            Point floodSeed(-1, -1);
            for (int dx = 0; dx < width && floodSeed.x < 0; ++dx)
            {
                for (int dy = 0; dy < height && floodSeed.y < 0; ++dy)
                {
                    if (ccLabels.at<int>(top + dy, left + dx) == ccLabel)
                    {
                        floodSeed = Point(left + dx, top + dy);
                    }
                }
            }
            rawFloodAreas.setTo(2, ccLabels == ccLabel);
            floodSeeds.push_back(floodSeed);
            hasLowLabels = true;
        }
        if (!hasLowLabels)
        {
            return interruptSplit(inputOutputLabels, splittableLabels, splitParams);
        }

        // --high
        // get threshold masks
        Mat highMask;
        threshold(channel, highMask, thresholdValue, 1, THRESH_BINARY);
        bitwise_and(highMask, mask, highMask);

        // spacial low high split
        bool hasHighLabels = false;
        ccLabelCount = connectedComponentsWithStats(highMask, ccLabels, ccStats, ccCentroids, 4, CV_32SC1, CCL_Type);
        highMask.release();
        for (int ccLabel = 1; ccLabel < ccLabelCount; ++ccLabel)
        {
            const int area = ccStats.at<int>(ccLabel, CC_STAT_AREA);
            if (area < childMinSize)
            {
                rawFloodAreas.setTo(3, ccLabels == ccLabel);
                continue;
            }
            const int left = ccStats.at<int>(ccLabel, CC_STAT_LEFT);
            const int top = ccStats.at<int>(ccLabel, CC_STAT_TOP);
            const int width = ccStats.at<int>(ccLabel, CC_STAT_WIDTH);
            const int height = ccStats.at<int>(ccLabel, CC_STAT_HEIGHT);
            Point floodSeed(-1, -1);
            for (int dx = 0; dx < width && floodSeed.x < 0; ++dx)
            {
                for (int dy = 0; dy < height && floodSeed.y < 0; ++dy)
                {
                    if (ccLabels.at<int>(top + dy, left + dx) == ccLabel)
                    {
                        floodSeed = Point(left + dx, top + dy);
                    }
                }
            }
            rawFloodAreas.setTo(4, ccLabels == ccLabel);
            floodSeeds.push_back(floodSeed);
            hasHighLabels = true;
        }
        if (!hasHighLabels)
        {
            return interruptSplit(inputOutputLabels, splittableLabels, splitParams);
        }
        mask.release();

        // final flooding
        const double iterationBorderConfidence = -(nextLabel + 1);
        Mat labels = inputOutputLabels.getMatRef();
        bool floodedFirst = false;
        for (const Point &floodSeed : floodSeeds)
        {
            if (rawFloodAreas.at<uchar>(floodSeed) == 0)
            {
                // already merged
                continue;
            }
            const int floodSize = floodFill(rawFloodAreas, floodSeed, 6, nullptr, 1, 1, 4 | FLOODFILL_FIXED_RANGE);
            const Mat floodMask = (rawFloodAreas == 6);
            rawFloodAreas.setTo(0, floodMask);

            // create child label
            const int childLabelId = floodedFirst ? nextLabel++ : id; // first child takes id of parent
            const Label childLabel(channels, floodMask, floodSize, childLabelId, splitParams);

            // remember child label if still splittable
            if (floodedFirst)
            {
                labels.setTo(childLabelId, floodMask);
            }

            if (childLabel.isSplittable(splitParams.splitThreshold))
            {
                splittableLabels.insert(std::lower_bound(splittableLabels.begin(), splittableLabels.end(), childLabel, Label::compare), childLabel);
            }
            floodedFirst = true;
        }
    }

    int hhts(const InputArray image, const OutputArray outputLabels, const int superpixels, const double splitThreshold, const int histogramBins, const int minSegmentSize, const int colorChannels, const bool applyBlur, const InputArray inputPreLabels)
    {
        vector<Mat> labels;
        const vector<int> superpixelss = {superpixels};
        const vector<int> labelCounts = hhts(image, labels, superpixelss, splitThreshold, histogramBins, minSegmentSize, colorChannels, applyBlur, inputPreLabels);
        labels[0].copyTo(outputLabels.getMatRef());
        return labelCounts[0];
    }

    vector<int> hhts(const InputArray image, const OutputArrayOfArrays outputLabels, const vector<int> &superpixels, const double splitThreshold, const int histogramBins, const int minSegmentSize, const int colorChannels, const bool applyBlur, const InputArray inputPreLabels)
    {
        const Size size = image.size();

        vector<Mat> channels;
        getChannels(image, colorChannels, channels, applyBlur);

        SplitParams splitParams(superpixels, splitThreshold, histogramBins, minSegmentSize);

        Mat labels = Mat::zeros(size, CV_32SC1);
        vector<Label> splittableLabels{}; // sorted

        int nextLabel = 1;

        Mat preLabels = inputPreLabels.getMat();
        if (preLabels.empty()) // empty pre-labels => segment everything from scratch
        {
            preLabels = Mat::ones(size, CV_32SC1);
        }
        // init pre labels
        while (true)
        {
            double maxVal;
            minMaxLoc(preLabels, nullptr, &maxVal, nullptr, nullptr);
            const int preLabelPreId = maxVal;
            if (preLabelPreId == 0)
            {
                // processed all pre-labels
                break;
            }

            // process pre-label
            const int preLabelId = nextLabel++;
            const Mat preLabelMask = preLabels == preLabelPreId;
            preLabels.setTo(0, preLabelMask);
            const int preLabelSize = SparseMat(preLabelMask).nzcount();
            const Label preLabel(channels, preLabelMask, preLabelSize, preLabelId, splitParams);

            labels.setTo(preLabelId, preLabelMask);

            if (preLabel.isSplittable(splitParams.splitThreshold))
            {
                splittableLabels.insert(std::lower_bound(splittableLabels.begin(), splittableLabels.end(), preLabel, Label::compare), preLabel);
            }
        }

        outputLabels.create(Size(splitParams.superpixels.size(), 1), CV_32SC1);
        vector<int> labelCounts{};

        while ((splitParams.superpixels.size() > 0 || splitParams.superpixels[0] < 0) && splittableLabels.size() > 0)
        {
            Label worstLabel = splittableLabels[0];
            splittableLabels.erase(splittableLabels.begin());

            worstLabel.split(channels, labels, size, nextLabel, splittableLabels, splitParams);

            // check for label output
            while (splitParams.superpixels.size() > 0 && splitParams.superpixels[0] >= 0 && nextLabel > splitParams.superpixels[0])
            {
                splitParams.superpixels.erase(splitParams.superpixels.begin());

                labels.copyTo(outputLabels.getMatRef(labelCounts.size()));
                labelCounts.push_back(nextLabel);
            }
        }

        // add remaining segmentation levels
        while (splitParams.superpixels.size() > 0)
        {
            splitParams.superpixels.erase(splitParams.superpixels.begin());

            labels.copyTo(outputLabels.getMatRef(labelCounts.size()));
            labelCounts.push_back(nextLabel);
        }

        return labelCounts;
    }
}