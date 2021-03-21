# American Sign Language Translation in Real Time

## About

This project converts [American Sign Language](https://en.wikipedia.org/wiki/American_Sign_Language) to English language in Real time video using Convolutional Neural Networks (CNNs).

<img alt="American Sign Language" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/American_Sign_Language_ASL.svg/1200px-American_Sign_Language_ASL.svg.png">

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

<img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

<img alt="Pandas" src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" />

<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />
</p>

## Files

```
train.py
main.py

src/
 | -augments.py
 |
 | -dataset.py
 |
 | -models.py
 |
 | -trainer.py

requirements.txt
image-data-file.csv
lb.pkl
```

- `main.py` contains the code to access webcam and make predictions
- `train.py` contains the code to train the model
- `src/augments.py` contains the code to define Train and Valid Augmentations
- `src/dataset.py` contains the code to create Train and Valid Dataset class
- `src/models.py` contains the code to create a custom neural network model
- `src/trainer.py` contains the code to train one epoch of the model
- `image-data-file.csv` has 2 columns - image_path and corresponding targets (labels encoded using scikit-learn)
- `lb.pkl` is a saved scikit-learn model to encode labels

## Training the model

### 1. Getting the data

You can download the data from [here](https://www.kaggle.com/grassknoted/asl-alphabet/data)

Now, take the downloaded `.zip` file and extract it into a new folder: `dataset/` and delete the `space/` folder from the `dataset/`.

Make sure the `dataset/` folder is at the same directory level as the train.py file.

### 2. Setting up Environment

_Note: If you already have the requirements (libraries and packages) included in the `requirements.txt` skip this step._

`$ conda create --name <env> --file requirements.txt`

### 3. Training the model

If you have done the above steps right, then just running the train.py script should not produce any errors.

To run training, open the terminal and change your working directory to the same level as the train.py file.

`$ python train.py`

This should start a training in a few seconds and you should see a progress bar.

## Making Predictions

If you don't want to train the model yourself and just want to make predictions then just install the required packages by following this [step](###3-training-the-model).

Then run the following command

`$ python main.py`

## Contact

If you have any query regarding the code or anything else, please open an Issue and I'll be happy to help!
