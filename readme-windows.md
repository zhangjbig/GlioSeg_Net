# GlioSeg-Net

This manual introduces the training and inference process of the GlioSeg-Net.

The code was developed on top of the excellent [nnU-Net library](https://github.com/MIC-DKFZ/nnUNet). This readme only provides a targeted introduction around our network. You can refer to the original files of nnU-Net for more detailed instructions on installation, usage, and common Q&A.


# Installation
GlioSeg-Net has been tested on Linux (Ubuntu 16, 18 and 20; centOS, RHEL) and windows(windows10, windows11). We do not provide support for other operating systems.

We very strongly recommend you install GlioSeg-Net in a virtual environment. 

If you choose to compile pytorch from source, you will need to use conda instead of pip. In that case, please set the environment variable OMP_NUM_THREADS=1 (preferably in your bashrc using `export OMP_NUM_THREADS=1`). This is important!

Python 2 is deprecated and not supported. Please make sure you are using Python 3.

1. Install [PyTorch](https://pytorch.org/get-started/locally/). You need at least version 1.6.

2. Install GlioSeg Net based on your system  (this will create a copy of the GlioSeg-Net code on your computer so that you can modify it as needed):

    ```bash
    git clone https://github.com/MIC-DKFZ/nnUNet.git(待修改)
    cd nnUNet
    pip install -e .
    ```

3. GlioSeg-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to set a few of environment variables.

4. You can download the model we provide for inferencing, or choose to train a model yourself.

    \- folder_0 : GlioSeg-Net;

    \- folder_1 : nnU-Net;

    \- folder_2 : the network with only the added Transformer module in nnUNet, which is the smaller GlioSeg-Net;

    \- folder_3 : the network with only modified nnU-Net to a larger network, which is the larger nnU-Net.

5. (OPTIONAL) Install hiddenlayer. Hiddenlayer enables GlioSeg-Net to generate plots of the network topologies it generates. To install hiddenlayer, run the following commands:
   ```bash
   pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
   ```

# Dataset conversion
GlioSeg-Net expects datasets in a structured format. This format closely (but not entirely) follows the data structure of 
the [Medical Segmentation Decthlon](http://medicaldecathlon.com/). Please read [nnUNet library](https://github.com/MIC-DKFZ/nnUNet) for information on how to convert datasets to be compatible with our network.

# Experiment planning and preprocessing
Provided that the requested raw dataset is located in the correct folder (`nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK`), you can run this step with the following command:

```bash
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```

`XXX` is the integer identifier associated with your Task name `TaskXXX_MYTASK`. You can pass several task IDs at once.

`--verify_dataset_integrity` should be run at least for the first time the command is run on a given dataset. This will execute some checks on the dataset to ensure that it is compatible with GlioSeg-Net. If this check has passed once, it can be omitted in future runs. If you adhere to the dataset conversion guide (see above) then this should pass without issues :-)

After `nnUNet_plan_and_preprocess` is completed, the U-Net configurations have been created and a preprocessed copy of the data will be located at nnUNet_preprocessed/TaskXXX_MYTASK.

# Model training

**Before training, please ensure that the network you have chosen is the one you want to use. To meet the format requirements of nnUNet, we suggest making the following modifications.**

1. **Change the file name of "./nnunet/network_architecture/GlioSeg_Net.py" to "generic_UNet.py".**
2. **Change the file name of "./nnunet/training/network_training/GlioSegNet_Trainer.py" to "nnUNetTrainerV2.py".**

Training models is done with the `nnUNet_train` command. For FOLD in [0, 1, 2, 3, 4], the structure of the command is:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 TASK_NAME_OR_ID FOLD  --npz (additional options)
```

`3d_fullres` is a string that identifies the requested U-Net configuration. Since GlioSeg-Net is built based on 3D full resolution U-Net, it should be 3d_fullres here.

`nnUNetTrainerV2` is the name of the model trainer. If you implement custom trainers (GlioSeg-Net as a framework) you can specify your custom trainer here.
`TASK_NAME_OR_ID` specifies what dataset should be trained on and `FOLD` specifies which fold of the 5-fold-cross-validaton is trained.

IMPORTANT: `--npz` makes the models save the softmax outputs during the final validation. It should only be used for trainings where you plan to run `nnUNet_find_best_configuration` afterwards (this is nnU-Nets automated selection of the best performing (ensemble of) configuration(s), see below).

GlioSeg-Net stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `-c` to the  training command.

The trained models will we written to the RESULTS_FOLDER/nnUNet folder. Each training obtains an automatically generated output folder name:

nnUNet_preprocessed/CONFIGURATION/TaskXXX_MYTASKNAME/TRAINER_CLASS_NAME__PLANS_FILE_NAME/FOLD For Task001_BrainTumour (from the MSD), for example, this looks like this:

    RESULTS_FOLDER/nnUNet/
    ├── 3d_fullres
    │   └── Task01_BrainTumour
    │       └── nnUNetTrainerV2__nnUNetPlansv2.1
    │           ├── fold_0
    │           │   ├── debug.json
    │           │   ├── model_best.model
    │           │   ├── model_best.model.pkl
    │           │   ├── model_final_checkpoint.model
    │           │   ├── model_final_checkpoint.model.pkl
    │           │   ├── network_architecture.pdf
    │           │   ├── progress.png
    │           │   └── validation_raw
    │           │       ├── BraTS_001.nii.gz
    │           │       ├── BraTS_001.pkl
    │           │       ├── BraTS_025.nii.gz
    │           │       ├── BraTS_025.pkl
    │           │       ├── BraTS_066.nii.gz
    │           │       ├── BraTS_066.pkl
    │           │       ├── BraTS_084.nii.gz
    │           │       ├── BraTS_084.pkl
    │           │       ├── summary.json
    │           │       └── validation_args.json
    │           ├── fold_1
    │           ├── fold_2
    │           ├── fold_3
    │           └── fold_4

In each model training output folder, the following files will be created (only shown for one folder above for brevity):

- **debug.json**: Contains a summary of blueprint and inferred parameters used for training this model. Not easy to read, 
but very useful for debugging ;-)
- **model_best.model / model_best.model.pkl**: checkpoint files of the best model identified during training. Not used right now.
- **model_final_checkpoint.model / model_final_checkpoint.model.pkl**: checkpoint files of the final model (after training 
has ended). This is what is used for both validation and inference.
- **network_architecture.pdf** (only if hiddenlayer is installed!): a pdf document with a figure of the network architecture in it.
- **progress.png**: A plot of the training (blue) and validation (red) loss during training. Also shows an approximation of 
the evlauation metric (green). This approximation is the average Dice score of the foreground classes. It should, 
however, only to be taken with a grain of salt because it is computed on randomly drawn patches from the validation 
data at the end of each epoch, and the aggregation of TP, FP and FN for the Dice computation treats the patches as if 
they all originate from the same volume ('global Dice'; we do not compute a Dice for each validation case and then 
average over all cases but pretend that there is only one validation case from which we sample patches). The reason for 
this is that the 'global Dice' is easy to compute during training and is still quite useful to evaluate whether a model 
is training at all or not. A proper validation is run at the end of the training.
- **validation_raw**: in this folder are the predicted validation cases after the training has finished. The summary.json 
contains the validation metrics (a mean over all cases is provided at the end of the file).

During training it is often useful to watch the progress. We therefore recommend that you have a look at the generated 
progress.png when running the first training. It will be updated after each epoch.

Training times largely depend on the GPU. The smallest GPU we recommend for training is the Nvidia RTX 2080ti. With 
this GPU (and pytorch compiled with cuDNN 8.0.2), all network trainings take less than 2 days.

# Identifying the best U-Net configuration
Once all 5 models are trained, use the following command to automatically determine what U-Net configuration(s) to use for test set prediction:

```bash
nnUNet_find_best_configuration -m 3d_fullres -t XXX --strict
```

(all 5 folds need to be completed for all specified configurations!)

# Run inference
Remember that the data located in the input folder must adhere to the nnU-Net format.

`nnUNet_find_best_configuration` will print a string to the terminal with the inference commands you need to use. 
The easiest way to run inference is to simply use these commands. 

If you wish to manually specify the configuration(s) used for inference, use the following commands:

For each of the desired configurations, run:
```
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz
```

Only specify `--save_npz` if you intend to use ensembling. `--save_npz` will make the command save the softmax 
probabilities alongside of the predicted segmentation masks requiring a lot of disk space.

Please select a separate `OUTPUT_FOLDER` for each configuration!

If you wish to run ensembling, you can ensemble the predictions from several configurations with the following command:
```bash
nnUNet_ensemble -f FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -pp POSTPROCESSING_FILE
```

You can specify an arbitrary number of folders, but remember that each folder needs to contain npz files that were generated by `nnUNet_predict`. 

For ensembling you can also specify a file that tells the command how to postprocess. These files are created when running `nnUNet_find_best_configuration` and are located in the respective trained model directory (RESULTS_FOLDER/nnUNet/CONFIGURATION/TaskXXX_MYTASK/TRAINER_CLASS_NAME__PLANS_FILE_IDENTIFIER/postprocessing.json or RESULTS_FOLDER/nnUNet/ensembles/TaskXXX_MYTASK/ensemble_X__Y__Z--X__Y__Z/postprocessing.json). 

You can also choose to not provide a file (simply omit -pp) and GlioSeg-Net will not run postprocessing.