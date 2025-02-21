# Generation of handwritten digits using GANs

This repository demonstrates how to train a generator using a Generative Adversarial Network (GAN) to generate handwritten digits.

The dataset used is the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits.
The generator is trained to produce images that are indistinguishable from real images in the dataset.

## Example

The following image shows the initial state of the generator, which generates random noise.
At this stage, the discriminator's predictions are also random.
| ![initial generation](resources/initial.png) |
|:---:|
| *Initial state of the generator. The upper half of the images corresponds to real images, while the lower half corresponds to generated images. Colored pixels represent the discriminator's predicted values, with green indicating real images and red indicating generated images.* |

After one epoch of training, the generator is still unable to produce images resembling real digits.
However, the discriminator has improved and can distinguish between real and fake images to some extent.
| ![first epoch](resources/first_epoch.png) |
|:---:|
| *Generator after the first epoch of training. The upper half of the images corresponds to real images, while the lower half corresponds to generated images. Colored pixels represent the discriminator's predictions, with green indicating real images and red indicating generated images.* |


After two epochs of training, the generator begins to create patterns that slightly resemble handwritten digits, though they are not yet recognizable.
The discriminator remains quite accurate in its predictions.
| ![second epoch](resources/second_epoch.png) |
|:---:|
| *Generator after the second epoch of training. The upper half of the images corresponds to real images, while the lower half corresponds to generated images. Colored pixels represent the discriminator's predictions, with green indicating real images and red indicating generated images.* |

At a later stage of training, the generator successfully produces images that closely resemble real handwritten digits.
Conversely, the discriminator struggles to distinguish between real and fake images.
| ![final generation](resources/final.png) |
|:---:|
| *Final state of the generator. The upper half of the images corresponds to real images, while the lower half corresponds to generated images. Colored pixels represent the discriminator's predictions, with green indicating real images and red indicating generated images.* |


## Usage

1. Install dependencies.
    ```shell
    pip install -r requirements.txt
    ```

2. Run the training script: `train.py`.
