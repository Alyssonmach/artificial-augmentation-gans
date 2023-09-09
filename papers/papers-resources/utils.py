import matplotlib.pyplot as plt
import torch

def grid_images(image_array, cmap, suptitle, name_fig):
    '''
    Function that creates a 3 x 3 image grid from an array tensor with 9 images

    Parameters:
        image_array (tensor): tensor with the array of 9 images.
        cmap (string): Type of image color mapping.
        suptitle (string): general title of the image grid.
        name_fig (string): Name of the picture grid to be saved in .png format.
    '''

    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (20, 20))


    axes[0, 0].imshow(image_array[0], cmap = cmap)
    axes[0, 1].imshow(image_array[1], cmap = cmap)
    axes[0, 2].imshow(image_array[2], cmap = cmap)
    axes[1, 0].imshow(image_array[3], cmap = cmap)
    axes[1, 1].imshow(image_array[4], cmap = cmap)
    axes[1, 2].imshow(image_array[5], cmap = cmap)
    axes[2, 0].imshow(image_array[6], cmap = cmap)
    axes[2, 1].imshow(image_array[7], cmap = cmap)
    axes[2, 2].imshow(image_array[8], cmap = cmap)

    fig.suptitle(suptitle, fontsize = 35, y = 0.92)
 
    fig.subplots_adjust(wspace = 0.1, hspace = 0.1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(f'{name_fig}.png', bbox_inches = 'tight')

def get_input_dimensions(z_dim, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.

    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        n_classes: the total number of classes in the dataset, an integer scalar
    
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                             which takes the noise and class vectors
    '''

    generator_input_dim = z_dim + n_classes
    
    return generator_input_dim

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
