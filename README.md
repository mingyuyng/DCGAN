# DCGAN based Semantic Image Inpainting

## Prerequisites

- Python 3.3+
- [Tensorflow](https://www.tensorflow.org/install/pip) 
- [Opencv](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html) 
- [SciPy](http://www.scipy.org/install.html)
- [Pillow](https://pillow.readthedocs.io/en/5.3.x/installation.html)
- [Requests](http://docs.python-requests.org/en/v2.7.0/user/install/)
- [Tqdm](https://stackoverflow.com/questions/47529792/no-module-named-tqdm)
 
## Usage

First, download dataset celebA and MNIST:

    $ python download.py mnist celebA
  
Then, to test image inpainting with celebA, run:

    $ python complete.py --outDir outputImages  --num 16 --batch_size 16 --nIter 1000 --imgs './data/celebA/*' --dataset celebA --maskType 'center' 

The number of images, the size of batch, the number of iterations and and type of mask can be changed. 

This code supports 5 masks: random, center, eye, left, crop

The completed images without Poisson Blending are stored in ./outputImages/completed/

The completed images with Poisson Blending are stored in ./outputImages/completed_blend/

The images sampled from DCGAN are stored in ./outputImages/hats_imgs/

To test image inpainting with MNIST, run:

    $ python complete.py --outDir outputImages  --num 16 --batch_size 16 --nIter 1000 --imgs './data/mnist/*' --dataset mnist --maskType 'center' 
    
For MNIST, Poisson Blending is not applied. 
