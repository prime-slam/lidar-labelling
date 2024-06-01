# lidar-labelling
[![tests](https://github.com/prime-slam/lidar-labelling/actions/workflows/ci.yml/badge.svg)](https://github.com/prime-slam/lidar-labelling/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

lidar-labelling is a tool for automatic segmentation of raw lidar clouds based on image segmentation.

Our labelling algorithm consists of two stages.

The first stage is a preliminary segmentation - a naive transfer of segmentation from images (performed by the [SAM](https://github.com/facebookresearch/segment-anything) algorithm) to the cloud. Then multi-stage processing of the cloud is performed, which allows us to make the cloud more compact before the final labelling without losing points that are significant for labelling. The preprocessing stages include removing points whose corresponding pixels were not marked on a sequence of images, selection of points close to the sensors, removal of noise, cloud voxelization.

The next stage is segmentation itself. The segmentation criterion in this work is the distance between points, which is calculated through the physical distance and the degree of similarity of the labelling of points on several images. Based on the distance matrix, the prepared cloud is segmented using the [GraphCut](https://ieeexplore.ieee.org/abstract/document/937505) algorithm.

## Datasets
This tool currently supports processing of [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset clouds. Support for other datasets involves writing an implementation of an abstract dataset class.

## Usage
Please check `example.ipynb` with a example of cloud segmentation from the [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset.

## Experiments
Files for reproducing experiments are in folder `experiment`. You will need the [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset with its folder structure and pre-performed image segmentation using the [SAM](https://github.com/facebookresearch/segment-anything) algorithm in npz format.

First run `main_kitti_processing.py` to generate segmentation with our algorithm. Then run `main_kitti_processing_metrics.py` to calculate the segmentation metrics of each run and write them to a csv file. Then run `main_calc_metrics_by_csv.py` to calculate the average values ​​of the metrics in the csv file.

## License
This project is licensed under the Apache License — 
see the [LICENSE](https://github.com/prime-slam/lidar-labelling/blob/main/LICENSE) file for details.
