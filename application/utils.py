from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image 
from torch import nn
import numpy as np
import torch
import math

def show_tensor_images(image_tensor, num_images = 25, nrow = 5, 
                       image_name = 'application/images/tensor_images.png'):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow = nrow)
    plt.imsave(image_name, image_grid.permute(1, 2, 0).squeeze().detach().numpy())

def get_input_dimensions(z_dim, mnist_shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''

    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    
    return generator_input_dim, discriminator_im_chan

def create_gif(images_pytorch_tensor):

    images_numpy = list()
    for index in range(0, len(images_pytorch_tensor)):
        convert_image = Image.fromarray(np.uint8(images_pytorch_tensor[index, 0, :, :].detach().numpy()*255))
        images_numpy.append(convert_image)
    
    images_numpy[0].save('application/images/tensor_images.gif', save_all = True, 
                         append_images = images_numpy, duration = 50, loop = 0)

def get_noise(n_samples, input_dim, device = 'cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    
    return torch.randn(n_samples, input_dim, device=device)

def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes),
    Parameters:
    labels: tensor of labels from the dataloader, size (?)
    n_classes: the total number of classes in the dataset, an integer scalar
    '''

    return F.one_hot(labels, n_classes)

def combine_vectors(x, y):
    '''
    Function for combining two vectors with shape (n_samples, ?) and (n_samples, ?).
    Parameters:
    x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
    y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''

    return torch.cat(tensors = (x.float(), y.float()), dim = 1)

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, input_dim = 10, im_chan = 1, hidden_dim = 64):
        
        super(Generator, self).__init__()
        self.input_dim = input_dim
        
        self.gen = nn.Sequential(
            self.make_gen_block(input_channels = input_dim, output_channels = hidden_dim * 4),
            self.make_gen_block(input_channels = hidden_dim * 4,  output_channels = hidden_dim * 2, 
                                kernel_size = 4, stride = 1),
            self.make_gen_block(input_channels = hidden_dim * 2,  output_channels = hidden_dim),
            self.make_gen_block(input_channels = hidden_dim,  output_channels = im_chan, 
                                kernel_size = 4, final_layer = True))

    def make_gen_block(self, input_channels, output_channels, kernel_size = 3, stride = 2, 
                       final_layer = False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, 
                                   kernel_size = kernel_size, stride = stride),
                nn.BatchNorm2d(num_features = output_channels),
                nn.ReLU(inplace = True))
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels = input_channels, out_channels = output_channels, 
                                   kernel_size = kernel_size, stride = stride),
                nn.Tanh())

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def interpolate_class(first_number, second_number, z_dim, n_classes, device, gen, n_interpolation = 30, 
                      show_tensor = True):

    interpolation_noise = get_noise(1, z_dim, device = device).repeat(n_interpolation, 1)

    first_label = get_one_hot_labels(torch.Tensor([first_number]).long(), n_classes)
    second_label = get_one_hot_labels(torch.Tensor([second_number]).long(), n_classes)

    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label

    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)

    if show_tensor:
        show_tensor_images(fake, num_images = n_interpolation, nrow = int(math.sqrt(n_interpolation)))
    
    return fake

def interpolate_noise(first_noise, second_noise, n_interpolation, interpolation_label, n_noise, gen, 
                      device):
    # This time you're interpolating between the noise instead of the labels
    percent_first_noise = torch.linspace(0, 1, n_interpolation)[:, None].to(device)
    interpolation_noise = first_noise * percent_first_noise + second_noise * (1 - percent_first_noise)

    # Combine the noise and the labels again
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_label.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), 
                       image_name = f'application/images/{n_noise}-generated.png')

def interpolate_noise_gif(n_noise, z_dim, n_interpolation, interpolation_label, gen, device):
    
    images_numpy = list()
    plot_noises = [get_noise(1, z_dim, device = device) for i in range(10)]
    for i, first_plot_noise in enumerate(plot_noises):
        for j, second_plot_noise in enumerate(plot_noises):
            interpolate_noise(first_plot_noise, second_plot_noise, n_interpolation, interpolation_label, 
                              n_noise, gen, device)
            images_numpy.append(Image.open(f'application/images/{n_noise}-generated.png').convert("RGB"))

    images_numpy[0].save('application/images/interpolate-noise.gif', save_all = True, 
                         append_images = images_numpy, duration = 100, loop = 0)