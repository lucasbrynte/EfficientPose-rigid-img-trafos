# EfficientPose with modifications for Natural Image Transformations
This code is a modification of the EfficientPose repository found [on GitHub](https://github.com/ybkscht/EfficientPose).
This README is a modification of the original EfficientPose README, which is reproduced in its entirety further down.

To be able to run the experiments for the Natural Image Transformations paper, use the original [installation instructions](#installation) from the EfficientPose, and use the same [datasets and pretrained weights](#dataset-and-pretrained-weights).

## Preprocessing
In order to train the PY-equivariant networks, you first need to preprocess the images, transforming them to the PY-domain, by running the Matlab script `preprocessing/warp_all_imgs.m`.
Before running, make sure to configure the `data_path` variable (set internally), such that it points to the `Linemod_preprocessed` directory (which you should have retrieved when following the dataset download instructions).
The result of running this preprocessing script is that next to the sub-directories `rgb`, `mask`, and `merged_masks` of the datasets, you should now have gotten corresponding directories next to them, with a `_arctan_warped` suffix added to their corresponding names, containing the images warped to the PY-domain.

## Training, evaluation, and modified options
Training and evaluation is carried out in essentially the same way as in the original repository, as described [here](#training) and [here](#evaluating).
There are however a number of alterations to the available options.
As the training script also carries out evaluation, the evaluation script is optional.

### Additional options shared between training and evaluation
- `--radial-arctan-prewarped-images` : Indicates that input images have been transformed to the PY-domain. This is not carried out online: prewarping is necessary, as described [here](#preprocessing).
- `--image-width` : Image width. Required by, and to be used along with `--radial-arctan-prewarped-images`.
- `--image-height` : Image height. Required by, and to be used along with `--radial-arctan-prewarped-images`.
- `--depth-regression-mode` : Controls what to be regressed in order to perform object depth estimation. Choose from: `zcoord` and `cam2obj_dist`. Default: `zcoord`.
- `--rot-target-frame-of-ref` : Controls what frame of reference to use for the target orientation to be estimated. Choose from: `cam` and `cam_aligned_towards_obj`. Default: `cam`.

### Additional options for training
Geometric data augmentations: (including our proposed "tilt" augmentations, referred to as RHaug in the paper)
- `--scale-6dof-augmentation` : Range from which to uniformly sample scales for 6DoF augmentation. Default: `--scale-6dof-augmentation 0.7 1.3`. Default: `(0.7, 1.3)`.
- `--inplane-angle-6dof-augmentation` : Range from which to uniformly sample angles for in-plane rotation for 6DoF augmentation. Provide 2 arguments, space separated, e.g. `--inplane-angle-6dof-augmentation 0 360`. Default: `(0, 360)`.
- `--tilt-angle-6dof-augmentation` : Range from which to uniformly sample angles for tilt rotation for 6DoF augmentation. Provide 2 arguments, space separated, e.g. `--tilt-angle-6dof-augmentation 0 360`. Default: `(0, 0)`.

Checkpoints and validation:
- `--validation-interval` : Validation interval (\#epochs). Originally, this was achieved by duplicating the dataset 10 times in each epoch, but now there is instead a dedicated option. Default: `10`.
- `--snapshot-interval` : Snapshot interval (\#epochs). If provided, not only the best snapshot is saved, but also snapshots with the given interval. In order to avoid cherry-picking the best-performing model across all training epochs for evaluation, such an interval should be provided. Optional.

Logging:
- `--csv-log-path` : Path to stream results as CSV. Optional.
- `--history-dump-path` : Path to dump final history dictionary as JSON. Optional.

Finally, note that the original `--steps` option is removed.
Instead, the number of batches per epoch is automatically inferred from the datasets.
Furthermore, reshuffling is now carried out at every epoch.



# \[THE ORIGINAL README:\] EfficientPose
This is the official implementation of [EfficientPose](https://arxiv.org/abs/2011.04307). 
We based our work on the Keras EfficientDet implementation [xuannianz/EfficientDet](https://github.com/xuannianz/EfficientDet) which again builds up on the great Keras RetinaNet implementation [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet), the official EfficientDet implementation [google/automl](https://github.com/google/automl) and [qubvel/efficientnet](https://github.com/qubvel/efficientnet).

![image1](figures/title_figure_repo.png)

## Installation

1) Clone this repository
2) Create a new environment with ```conda create -n EfficientPose python==3.7```
3) Activate that environment with ```conda activate EfficientPose```
4) Install Tensorflow 1.15.0 with ```conda install tensorflow-gpu==1.15.0```
5) Go to the repo dir and install the other dependencys using ```pip install -r requirements.txt```
6) Compile cython modules with ```python setup.py build_ext --inplace```

## Dataset and pretrained weights

You can download the Linemod and Occlusion datasets and the pretrained weights from [here](https://drive.google.com/drive/folders/1VcBLcIBhuT5MmXfE9NMrFdAk2xzF3mP5?usp=sharing).
Just unzip the Linemod_and_Occlusion.zip file and you can train or evaluate using these datasets as described below.

The dataset is originally downloaded from [j96w/DenseFusion](https://github.com/j96w/DenseFusion) as well as [chensong1995/HybridPose](https://github.com/chensong1995/HybridPose) and were preprocessed using the ```generate_masks.py``` script.
The EfficientDet COCO pretrained weights are from [xuannianz/EfficientDet](https://github.com/xuannianz/EfficientDet).

## Training

### Linemod
To train a phi = 0 EfficientPose model on object 8 of Linemod (driller) using COCO pretrained weights:
```
python train.py --phi 0 --weights /path_to_weights/file.h5 linemod /path_to_dataset/Linemod_preprocessed/ --object-id 8
```

### Occlusion
To train a phi = 0 EfficientPose model on Occlusion using COCO pretrained weights:
```
python train.py --phi 0 --weights /path_to_weights/file.h5 occlusion /path_to_dataset/Linemod_preprocessed/
```

See train.py for more arguments.

## Evaluating

### Linemod
To evaluate a trained phi = 0 EfficientPose model on object 8 of Linemod (driller) and (optionally) save the predicted images:
```
python evaluate.py --phi 0 --weights /path_to_weights/file.h5 --validation-image-save-path /where_to_save_predicted_images/ linemod /path_to_dataset/Linemod_preprocessed/ --object-id 8
```

### Occlusion
To evaluate a trained phi = 0 EfficientPose model on Occlusion and (optionally) save the predicted images:
```
python evaluate.py --phi 0 --weights /path_to_weights/file.h5 --validation-image-save-path /where_to_save_predicted_images/ occlusion /path_to_dataset/Linemod_preprocessed/
```

If you don`t want to save the predicted images just skip the --validation-image-save-path argument.

## Inferencing

We also provide two basic scripts demonstrating the exemplary use of a trained EfficientPose model for inferencing.
With ```python inference.py``` you can run EfficientPose on all images in a directory. The needed parameters, e.g. the path to the images and the model can be modified in the ```inference.py``` script.

With ```python inference_webcam.py``` you can run EfficientPose live with your webcam. Please note that you have to replace the intrinsic camera parameters used in this script (Linemod) with your webcam parameters.
Since the Linemod and Occlusion datasets are too small to expect a reasonable 6D pose estimation performance in the real world and a lot of people probably do not have the exact same objects used in Linemod (like me), you can try to display a Linemod image on your screen and film it with your webcam.

## Benchmark

To measure the runtime of EfficientPose on your machine you can use ```python benchmark_runtime.py```.
The needed parameters, e.g. the path to the model can be modified in the ```benchmark_runtime.py``` script.
Similarly, you can also measure the vanilla EfficientDet runtime on your machine with the ```benchmark_runtime_vanilla_effdet.py``` script.

## Debugging Dataset and Generator

If you want to modify the generators or build a new custom dataset, it can be very helpful to display the dataset annotations loaded from your generator to make sure everything works as expected.
With 
```
python debug.py --phi 0 --annotations linemod /path_to_dataset/Linemod_preprocessed/ --object-id 8
```
 you can display the loaded and augmented image as well as annotations prepared for a phi = 0 model from object 8 of the Linemod dataset.
Please see debug.py for more arguments.

## Citation

Please cite [EfficientPose](https://arxiv.org/abs/2011.04307) if you use it in your research
```
@misc{bukschat2020efficientpose,
      title={EfficientPose: An efficient, accurate and scalable end-to-end 6D multi object pose estimation approach}, 
      author={Yannick Bukschat and Marcus Vetter},
      year={2020},
      eprint={2011.04307},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

EfficientPose is licensed under the Creative Commons Attribution-NonCommercial 4.0 International license and is freely available for non-commercial use. Please see the LICENSE for further details. If you are interested in commercial use, please contact us under yannick.bukschat@stw.de or marcus.vetter@stw.de.
