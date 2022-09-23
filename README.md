# EagerMOT: 3D Multi-Object Tracking via Sensor Fusion

This repo is the mindspore implementation of "EagerMOT: 3D Multi-Object Tracking via Sensor Fusion"(https://arxiv.org/abs/2104.14682)

## Requirements for conda
$ conda create --name <env> --file <this file>
    platform: linux-64


## Requirements for pip

## Benchmark results

Our current standings on **KITTI** for 2D MOT on [the official leaderboard](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). For 2D MOTS, see [this page](http://www.cvlibs.net/datasets/kitti/eval_mots_detail.php?result=714550ab34eca8356b2163f8c18c246ec18fbf0b). 
Our current standings on **NuScenes** for 3D MOT on [the official leaderboard](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any).

## How to set up

Download official NuScenes and KITTI data if you plan on running tracking on them. Change the paths to that data in `configs/local_variables.py`. 

Also set a path to a working directory for each dataset - all files produced by EagerMOT will be saved in that directory, e.g. fused instances, tracking results. A subfolder will be created for each dataset for each split, for example, if the working directory is `/workspace/kitti`, then `/workspace/kitti/training` and `/workspace/kitti/testing` will be used for each data split. 

The data split to be run is specified in `local_variables.py`. For example, for KITTI, the `SPLIT` variable with be either `training` or `testing`. For NuScenes, in addition to changing the name of the `SPLIT` (train/val/test/mini-train/...), the version of the dataset (`VERSION = "v1.0-trainval"`) also has to be modified in `run_tracking.py` when switching between train/test. 

If running on KITTI, download `ego_motion.zip` from the [drive](https://drive.google.com/drive/folders/1MpAa9YErhAZNEJjIrC4Ky21YfNj2jatM?usp=sharing) and unzip it into the KITTI working directory specified above (either training or testing). NuScenes data is already in world coordinates, so no need to ego motion estimates.

### Download 3D and 2D detections, which ones to download depends on what you want to run:
We thank other researchers for opening their code and data to the community and therefore provide links to their data directly in hopes that you will also check out their great work! 
Data that was not available directly but was generated by us using their open source code is given through a cloud download link.

* KITTI 2D MOTSFusion detections/segmentations from https://github.com/tobiasfshr/MOTSFusion. Under the "Results" section, they provide link to download their detections.
* KITTI 2D TrackRCNN detections/segmentations from https://www.vision.rwth-aachen.de/page/mots. Under the "Downloads" section they provide a link to download detections.
* KITTI 3D [PointGNN](https://github.com/WeijingShi/Point-GNN), NuScenes 3D [CenterPoint](https://github.com/tianweiy/CenterPoint), NuScenes 2D detections using an [MMDetection](https://github.com/open-mmlab/mmdetection) model from the [drive](https://drive.google.com/drive/folders/1MpAa9YErhAZNEJjIrC4Ky21YfNj2jatM?usp=sharing).  
* NuScenes 3D CenterPoint detections can also be downloaded directly from [the author's page](https://github.com/tianweiy/CenterPoint/blob/master/configs/nusc/README.md) for the `centerpoint_voxel_1440_dcn(flip)` config.
NuScenes 2D MMDetection output was obtained using the `cascade_mask_rcnn | X-101_32x4d` model given in [the model zoo](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/nuimages). See issue #9 for more details.
* KITTI 3D AB3DMOT detections can be downloaded from the original source https://github.com/xinshuoweng/AB3DMOT, but their format has recently changed and is no longer compatible with this repo. The detections provided in the above cloud link are simpy a copy downloaded at an earlier time when this the parsing code in this repo was written.

Our best benchmark results were achieved using detections from 3D PointGNN + (2D MOTSFusion+RRC) for KITTI and 3D CenterPoint + 2D MMDetectionCascade for NuScenes.

Unzip detections anywhere you want and provide the path to the root method folder in the `inputs/utils.py` file. 

**NOTE**: If using MOTSFusion input, also run the `adapt_kitti_motsfusion_input.py` script to copy the necessary detection information to its segmentation file.  

### Set up a virtual environment
* if using conda: 
```
conda create --name <env> --file requirements_conda.txt
```
* if using pip: 
```
python3 -m venv env
source env/bin/activate
pip install -r requirements_pip.txt
```



## How to run
See `run_tracking.py` for the code that launches tracking. Modify which function that file calls, depending on which dataset you want to run. See nearby comments for instructions.
```py
if __name__ == "__main__":
    # choose which one to run, comment out the other one
    run_on_nuscenes()  
    run_on_kitti()
```
Start the script with `$python run_tracking.py`. Check the code itself to see what is being called. 












