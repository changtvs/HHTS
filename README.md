# Hierarchical Histogram Threshold Segmentation - Auto-terminating High-detail Oversegmentation

**[Technische Hochschule Nuernberg, Game Tech Lab](https://www.th-nuernberg.de/fakultaeten/in/forschung/game-tech-labor/)**
[Thomas V. Chang](https://www.th-nuernberg.de/person/chang-thomas/), Simon Seibt, Bartosz von Rymon Lipinski

[[`Paper`]()] [[`Project page`](https://changtvs.github.io/hierarchical-histogram-threshold-segmentation/)] [[`BibTeX`](#citing-hhts)]

![HHTS samples](assets/HHTS-teaser.png?raw=true)

**Hierarchical Histogram Threshold Segmentation (HHTS)** generates high-detail oversegmentation labels with minimal parameter fine-tuning.

## Usage

Refer to `main.cpp` for usage. Sample image `247012.jpg` is part of the [BSDS500 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).

### Single-level segmentation
```
Mat labels;
int labelCount = HHTS::hhts(image, labels, 500, 0.0, 32, 64, HHTS::ColorChannel::RGB | HHTS::ColorChannel::LAB | HHTS::ColorChannel::HSV, false, noArray());
```

### Multi-level segmentation
```
vector<Mat> labels;
vector<int> spCounts{100, 250, 500};
vector<int> labelCounts = HHTS::hhts(image, labels, spCounts, 0.0, 32, 64, HHTS::ColorChannel::RGB | HHTS::ColorChannel::LAB | HHTS::ColorChannel::HSV, false, noArray());
```

### Auto-terminating segmentation
```
vector<Mat> labels;
vector<int> spCounts{500, -1};
vector<int> labelCounts = HHTS::hhts(image, labels, spCounts, 0.0, 32, 64, HHTS::ColorChannel::RGB | HHTS::ColorChannel::LAB | HHTS::ColorChannel::HSV, false, noArray());
```

## Abstract

Superpixels play a crucial role in image processing by partitioning an image into clusters of pixels with similar visual attributes. This facilitates subsequent image processing tasks, offering computational advantages over the manipulation of individual pixels. While numerous oversegmentation techniques have emerged in recent years, many rely on predefined initialization and termination criteria. In this paper, a novel top-down superpixel segmentation algorithm called Hierarchical Histogram Threshold Segmentation (HHTS) is introduced. It eliminates the need for initialization and implements auto-termination, outperforming state-of-the-art methods w.r.t boundary recall. This is achieved by iteratively partitioning individual pixel segments into foreground and background and applying intensity thresholding across multiple color channels. The underlying iterative process constructs a superpixel hierarchy that adapts to local detail distributions until color information exhaustion. Experimental results demonstrate the superiority of the proposed approach in terms of boundary adherence, while maintaining competitive runtime performance on the BSDS500 and NYUV2 datasets. Furthermore, an application of HHTS in refining machine learning-based semantic segmentation masks produced by the Segment Anything Foundation Model (SAM) is presented.

![HHTS samples](assets/hhts-mean-random-more.svg?raw=true)

![HHTS samples](assets/hhts-mean-random-detail.svg?raw=true)

## Evaluation

Our [Evaluation Repository](https://github.com/changtvs/hhts-evaluation) contains an implementation for the [Superpixel Benchmark](https://github.com/davidstutz/superpixel-benchmark) by Stutz et al. The current implementation may yield slightly different results as stated in our paper. The metrics used for Figure 4 can be found in the file [output/hhts/data-from-paper-figure4.csv](https://github.com/changtvs/hhts-evaluation/blob/master/output/hhts/data-from-paper-figure4.csv).

## License

The code is licensed under the [Apache 2.0 license](LICENSE).

## Citing HHTS

If you use HHTS in your research, please use the following BibTeX entry.

```
@InProceedings{Chang_2024_CVPR,
  title     = {Hierarchical Histogram Threshold Segmentation – Auto-terminating High-detail Oversegmentation},
  author    = {Chang, Thomas V. and Seibt, Simon and von Rymon Lipinski, Bartosz},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024}
}
```
