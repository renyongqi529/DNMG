# TSDG:An target-specific deep learning model by fusion of 3D information for de novo drug generation

![pipeline](images/mod2.png)

### Requirements

Model training is written in `pytorch==1.11.0` and uses `keras==2.4.0` for data loaders. `RDKit==2020.09.5` is needed for molecule manipulation.


### Creat a new environment in conda 

 `conda env create -f env.yaml `


## Pre-training

### Data Preparation
For the training a npy file is needed. We used subset of the Zinc dataset, using only the drug-like.The same clear target specific datasets were obtained from DUD-E database (http://dude.docking.org/targets).

In the `data/zinc` folder there will be the `zinc.smi` file that is required for the preparing data step.

`python prepare_data.py     --input ./data/zinc/zinc.csv 
                            --output ./data/zinc/zinc.npy`

## Training TSDG

`python wmain02.py --input ./data/zinc/zinc.npy
                   --output_dir ./savemodel/`

## Generation
Load the `./savemodel/discriminator-100000.pkl` ,`./savemodel/generator-100000.pkl`,`./savemodel/encoder-100000.pkl` and `./savemodel/decoder-100000.pkl` generative model, and typing the following codes:

`python GAN_generation.py `

## Transfer Learning 

The process of transfer learning is the same as it is in zinc data sets, using wmain02_trs1.py files when training models.


nohup python prepare_data.py -i ./data/zinc/small_data.csv -o ./data/zinc/small_data.npy  >data_test 2>&1 &
nohup python wmain02.py -i ./data/zinc/small_data.npy  >test 2>&1 &
