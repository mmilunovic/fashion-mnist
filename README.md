# Fashion MNIST On Steroids

This was the second project for the Machine Learning course on  [Faculty Of Computer Science](https://raf.edu.rs/).

## Problem description

The first part of the project was to train a convolutional neural network using Keras framework to classify images from [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist) dataset with at least **85%** accuracy on the test set. Trained models are in directory *models*.

**fashion_full.h5** is a CNN classifier trained on all 60k train images and tested on 10k images. It has 4 convolution layers, 2 max-pooling layers, 3 dense layers, 5 dropout layers, and 6 batch normalization layers. Achieved **93.52%** accuracy. But it's not great for the second part. 

**fashion_1_64.h5** is trained on 10k train and 10k test images with data augmentation. Achieved **90.22%** accuracy. Solid on the second problem.

**fashion_dataaug_1.h5** Is same architecture as **fashion_full.h5** but with different data augmentation method and only 10k images. Achieved **90.98%** accuracy. Best results on the second problem than previous solutions. 


<img src="https://cdn-images-1.medium.com/max/1080/1*HkW94w1erHUYMguiDWWHJQ.png" width="850"/>

The second part of the project was to use the previously trained model to classify multiple clothing items in a single image and draw their bounding boxes. They are always rotated properly but they can be scaled in any way, also there is no overlap between items. Example of test image and solution is given below.

<img src="https://github.com/mmilunovic/fashion-mnist/blob/master/tests/0.png" width="425"/> <img src="https://github.com/mmilunovic/fashion-mnist/blob/master/tests/0_sol.png" width="425"/> 


A detailed explanation is in [ML D2.pdf](https://github.com/mmilunovic/fashion-mnist/blob/master/ML%20D2.pdf)

## Image preprocessing - OpenCV

In order to extract individual items from a noisy image, I used several computer vision methods from [OpenCV](https://opencv.org/) library.

- Non-local Means Denoising 
- Inverted Binary Thresholding
- Contour extraction

Since images are very low quality there were many useless contours so I ignored those that were inside other contours (problem with sandal images) and ignored very small contours. Then, to get centered clothing items I calculated the center of mass of each *valid* contour and shifted bounding box coordinates accordingly. The last part was to use *cv2.bitwise_not* function to invert pixel values because training images have a black background and whitish items. 

## Prerequisites

- Keras
- OpenCV
- Tensorflow
- SciPy
- pandas

## Usage

To try just run following commands. 

```bash
git clone https://github.com/mmilunovic/fashion-mnist.git
cd fashion-mnist
python main.py <image_name>
```
This will produce an image named <image_name>_out.png which should contain bounding boxes and labels for each item.
