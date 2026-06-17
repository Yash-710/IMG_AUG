Image Augmentation Pipeline
This project provides a powerful and highly customizable image augmentation pipeline designed to increase the number of images for ML training. It uses the Albumentations library to apply a wide range of transformations.

✨ Key Features
37 Diverse Augmentations: A comprehensive set of transformations covering color, geometry, noise, weather, and lighting.
Bounding Box Safe: All transformations are compatible with YOLO format bounding boxes, ensuring label integrity is maintained throughout the pipeline.
Ready for Training: The script processes an entire directory of images and saves the augmented versions, ready for model training.
🚀 Getting Started
Prerequisites
Python 3.8+
A virtual environment (recommended)
Installation
Clone the repository:

git clone https://github.com/{YOUR_USERNAME}/imageaug-feature_yash.git
cd imageaug-feature_yash
Create and activate a virtual environment:

python -m venv env
.\env\Scripts\activate
Install the required dependencies: A requirements.txt file is recommended. If you don't have one, you can create it with:

pip freeze > requirements.txt
Then install with:

pip install -r requirements.txt
Usage
Place your source images and corresponding YOLO annotation files (.txt) in the images directory.
Run the main script:
python main.py
The augmented images and their annotations will be saved in the output directory.
🎨 Available Augmentations
The pipeline applies the following 37 augmentations:

Geometric & Crop
RandomSizedBBoxSafeCrop
HorizontalFlip
VerticalFlip
ShiftScaleRotate
RandomRotate90
Transpose
OpticalDistortion
GridDistortion
ElasticTransform
PadIfNeeded
Color & Contrast
RandomBrightnessContrast
HueSaturationValue
RGBShift
RandomGamma
CLAHE
ChannelShuffle
InvertImg
ToGray
ToSepia
ColorJitter
Blur & Noise
Blur
MotionBlur
MedianBlur
GaussianBlur
GaussNoise
ISONoise
Weather & Environment (Adaptive)
RandomRain (Adaptive)
RandomSnow
RandomFog
RandomSunFlare (Adaptive & Multi-Location)
RandomShadow
Other
Cutout
Posterize
Solarize
Equalize
Downscale
ImageCompression