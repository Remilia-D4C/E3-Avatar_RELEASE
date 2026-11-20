# E3-Avatar_RELEASE
Official implementation of "E3-Avatar: Efficient, Expressive and Easy-to-use Monocular Gaussian Avatars Reenactment", ACM MM 2026.

Paper Link: https://dl.acm.org/doi/10.1145/3767308.3835180

❗Due to some force majeure, parts of density control are missing, the region labels are hands, head and torso. We will complete then in following updates.

## Install
* Clone and install LGM: Follow the installation instructions provided in the LGM repository.
* Then set up the environment:
    conda env create -f environment.yml
    conda activate e3avatar

## Directory
```
${ROOT}
|-- main
|-- common
|-- |-- utils/human_model_files
|-- |-- |-- smplx/SMPLX_FEMALE.npz
|-- |-- |-- smplx/SMPLX_MALE.npz
|-- |-- |-- smplx/SMPLX_NEUTRAL.npz
|-- |-- |-- smplx/MANO_SMPLX_vertex_ids.pkl
|-- |-- |-- smplx/SMPL-X__FLAME_vertex_ids.npy
|-- |-- |-- smplx/smplx_flip_correspondences.npz
|-- |-- |-- flame/flame_dynamic_embedding.npy
|-- |-- |-- flame/FLAME_FEMALE.pkl
|-- |-- |-- flame/FLAME_MALE.pkl
|-- |-- |-- flame/FLAME_NEUTRAL.pkl
|-- |-- |-- flame/flame_static_embedding.pkl
|-- |-- |-- flame/FLAME_texture.npz
|-- data
|-- |-- XHumans
|-- |-- |-- data/00028
|-- |-- |-- data/00034
|-- |-- |-- data/00087
|-- LGM
|-- output
```
* Download SMPL-X 1.1 version from [here](https://smpl-x.is.tue.mpg.de/download.php). 
* Download FLAME 2020 version from [here](https://flame.is.tue.mpg.de/download.php).

## XHumans videos
* You can download original XHumans data from [here](https://skype-line.github.io/projects/X-Avatar/).
* First, use the official pre-processed code (`_X_Humans_preprocess_XHumans.py`). As the original code has a bug, which does not clamp UV coordinates in [0,1], we added a single line to handle this. Please use our `tools/preprocess_XHumans.py` instead of the original one.
* Then use the same pre-processing stage of the ExAvatar.[fitting](https://github.com/mks0601/ExAvatar_RELEASE/tree/main/fitting).

## Train
* Run `LGM` folder to generate initial coarse Gaussians and place the results in:`data/XHumans/$SUBJECT_ID`.
* Go to `main` folder and run `python train.py --subject_id $SUBJECT_ID`. The checkpoints are saved in `output/model/$SUBJECT_ID`.
* You can use one of `00028`, `00034`, and `00087` for `$SUBJECT_ID`.

## Animation
* Go to `main` folder and run `python animation.py --subject_id $SUBJECT_ID --test_epoch 1 --motion_path $PATH` if you want to use an avatar in `output/model_dump/$SUBJECT_ID`. `$PATH` should point to SMPL-X parameter files used to animate the avatar.These can be prepared using ExAvatar's fitting tool.

## Test and evaluation
* You can see test results on the testing frames by running `python test.py --subject_id $SUBJECT_ID --test_epoch 1`. The results are saved to `output/result/$SUBJECT_ID`.
* For the evaluation of the X-Humans dataset, run `python eval_xhumans.py --output_path ../output/result/$SUBJECT_ID --subject_id $SUBJECT_ID`.
*If you want to eval the X-Humans dataset with hand or head masks, run `python eval_masks.py --output_path ../output/result/$SUBJECT_ID --subject_id $SUBJECT_ID`.
