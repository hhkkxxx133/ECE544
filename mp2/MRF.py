import pylab as pl
import numpy as np
import argparse
import time


def MRF(I, J, eta, zeta, beta):
    """ 
    Perform Inference to determine the label of every pixel.
    1. Go through the image in random order.
    2. Evaluate the energy for every pixel being being set to either 1 or -1.
    3. Assign the label resulting in the least energy to the pixel in question.

    Inputs: 
        I: The original noisy image.
        J: The denoised image from the previous iteration.
        eta: Weight of the observed-latent pairwise ptential.
        zeta: Weight of the latent-latent pairwise ptential.
        beta: Weight of unary term.   
    Output:
        denoised_image: The denoised image after one MRF iteration.   
    """  
    start = time.time()
    height = np.arange(J.shape[0])
    width = np.arange(J.shape[1])
    np.random.shuffle(height)
    np.random.shuffle(width)
    denoised_image = J.copy()
    for y in height:
        for x in width:
            if energy_evaluation(I, denoised_image, x, y, 1, eta, zeta, beta) < energy_evaluation(I, denoised_image, x, y, -1, eta, zeta, beta):
                denoised_image[y,x] = 1
            else:
                denoised_image[y,x] = -1
    # denoised_image = J 
    # print("MRF")
    # print("time: "+str(time.time()-start))              
    return denoised_image
 

def energy_evaluation(I, J, pixel_x_coordinate, pixel_y_coordinate, 
    pixel_value, eta, zeta, beta):
    """
    Evaluate the energy of the image of a particular pixel set to either 1or-1.
    1. Set pixel(pixel_x_coordinate,pixel_y_coordinate) to pixel_value
    2. Compute the unary, and pairwise potentials.
    3. Compute the energy

    Inputs: 
        I: The original noisy image.
        J: The denoised image from the previous iteration.
        pixel_x_coordinate: the x coordinate of the pixel to be evaluated.
        pixel_y_coordinate: the y coordinate of the pixel to be evaluated.
        pixel_value: could be 1 or -1.
        eta: Weight of the observed-latent pairwise ptential.
        zeta: Weight of the latent-latent pairwise ptential.
        beta: Weight of unary term.   
    Output:
        energy: Energy value.

    """
    J[pixel_y_coordinate,pixel_x_coordinate] = pixel_value
    energy = 0
    height, width = J.shape

    if pixel_y_coordinate==0:
        if pixel_x_coordinate==0:
            energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate+1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate+1])
        elif pixel_x_coordinate==width-1:
            energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate+1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate-1])
        else:
            energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate+1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate-1]+J[pixel_y_coordinate,pixel_x_coordinate+1])
    elif pixel_y_coordinate==height-1:
        if pixel_x_coordinate==0:
            energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate-1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate+1])
        elif pixel_x_coordinate==width-1:
            energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate-1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate-1])
        else:
            energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate-1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate-1]+J[pixel_y_coordinate,pixel_x_coordinate+1])
    elif pixel_x_coordinate==0:
        energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate-1,pixel_x_coordinate]+J[pixel_y_coordinate+1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate+1]) 
    elif pixel_x_coordinate==width-1:
        energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate-1,pixel_x_coordinate]+J[pixel_y_coordinate+1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate-1])
    else:
        energy -= eta*J[pixel_y_coordinate,pixel_x_coordinate]*(J[pixel_y_coordinate-1,pixel_x_coordinate]+J[pixel_y_coordinate+1,pixel_x_coordinate]+J[pixel_y_coordinate,pixel_x_coordinate-1]+J[pixel_y_coordinate,pixel_x_coordinate+1])

    energy -= zeta*np.sum(np.multiply(J, I))
    energy -= beta*np.sum(J)

    return energy


def greedy_search(noisy_image, eta, zeta, beta, conv_margin):
    """
    While convergence is not achieved (this verified by calling 
    the function 'not_converged'),
    1. iteratively call the MRF function to perform inference.
    2. If the number of iterations is above 10, stop and return 
    the image that you have at the 10th iteration.

    Inputs: 
        noisy_image: the noisy image.
        eta: Weight of the pairwise observed-latent potential.
        zeta: Weight of the pairwise latent-latent potential.
        beta: Weight of unary term.    
        conv_margin: Convergence margin
    Output:
        denoised_image: The denoised image.   
    """
    
    # Noisy Image.
    I = noisy_image.copy()

    itr = 0
    denoised_image = I.copy()
    while(True):
        denoised_image = MRF(noisy_image,denoised_image,eta,zeta,beta)
        itr += 1
        if not_converged(I, denoised_image, itr, conv_margin):
            # print("yeah")
            break
        I = denoised_image.copy()

    return denoised_image


def not_converged(image_old, image_new, itr, conv_margin):
    """
    Check for convergence. Convergence is achieved if the denoised image 
    does not change between two consequtive iterations by a certain 
    margin 'conv_margin'.
    1. Compute the percentage of pixels that changed between two
     consecutive iterations.
    2. Convergence is achieved if the computed percentage is below 
    the convergence margin.

    Inputs:
        image_old: Denoised image from the previous iteration.
        image_new: Denoised image from the current iteration.
        itr: The number of iteration.
        conv_margin: Convergence margin.
    Output:  
        has_converged: a boolean being true in case of convergence
    """ 
    has_converged = False
    num = np.size( image_new[np.where(image_new!=image_old)] )

    if float(num/np.size(image_new)) < conv_margin or itr>=10:
        # print(num)
        # print(np.size(image_new))
        has_converged = True

    return has_converged


def load_image(input_file_path, binarization_threshold):
    """
    Load image and binarize it by:
    0. Read the image.
    1. Consider the first channel in the image
    2. Binarize the pixel values to {-1,1} by setting the values 
    below the binarization_threshold to -1 and above to 1.
    Inputs: 
        input_file_path.
        binarization_threshold.
    Output: 
        I: binarized image.
    """
    K = pl.imread(input_file_path)
    I = K[:,:,0]
    I[I>binarization_threshold] = 1
    I[I<=binarization_threshold] = -1
    return I


def inject_noise(image):
    """
    Inject noise by flipping the value of some randomly chosen pixels.
    1. Generate a matrix of probabilities of every pixel 
    to keep its original value .
    2. Flip the pixels if its corresponding probability in 
    the matrix is below 0.1.

    Input:
        image: original image
    Output:
        noisy_image: Noisy image
    """
    noisy_image = image.copy()

    height,width = noisy_image.shape
    prob = np.random.rand(height, width)
    noisy_image = np.where(prob<0.1, -noisy_image, noisy_image)
    
    return noisy_image


def f_reconstruction_error(original_image, reconstructed_image):
    """
    Compute the reconstruction error (L2 loss)
    inputs:
        original_image.
        reconstructed_image.
    output: 
        reconstruction_error: MSE of reconstruction.
    """
    reconstruction_error = np.mean(np.square(original_image - reconstructed_image))
    return reconstruction_error


def plot_image(image, title, path):
    pl.figure()
    pl.imshow(image)
    pl.title(title)
    pl.savefig(path)


def parse_arguments(parser):
    """
    Parse arguments from the command line
    Inputs: 
        parser object
    Output:
        Parsed arguments
    """
    parser.add_argument('--input_file_path',
        type=str,
        default="img/seven.png",
        metavar='<input_file_path>', 
        help='Path to the input file.')

    parser.add_argument('--weight_pairwise_observed_unobserved',
        type=float,
        default=2,
        metavar= '<weight_pairwise_observed_unobserved>',
        help='Weight of observed-unobserved pairwise potential.')

    parser.add_argument('--weight_pairwise_unobserved_unobserved',
        type=float, 
        default=1.5,
        metavar='<weight_pairwise_unobserved_unobserved>',
        help='Weight of unobserved-unobserved pairwise potential.')

    parser.add_argument('--weight_unary', 
        type=float, 
        default=0.1, 
        metavar='<weight_unary>', 
        help='Weight of the unary term.')

    parser.add_argument('--convergence_margin', 
        type=float, 
        default=0.001,#0.999, 
        metavar='<convergence_margin>', 
        help='Convergence margin.')

    parser.add_argument('--binarization_threshold', 
        type=float,\
        default=0.05, \
        metavar='<convergence_margin>',
        help='Perc of different btw the images between two iter.')
    
    args = parser.parse_args()
    return args


def main(): 
    # Read the input arguments
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    # Parse the MRF hyperparameters
    eta = args.weight_pairwise_observed_unobserved
    zeta = args.weight_pairwise_unobserved_unobserved
    beta = args.weight_unary

    # Parse the convergence margin
    conv_margin = args.convergence_margin  

    # Parse the input file path
    input_file_path = args.input_file_path
     
    # Load the image.  
    I = load_image(input_file_path, args.binarization_threshold)

    # Create a noisy version of the image.
    J = inject_noise(I)
    
    # Call the greedy search function to perform MRF inference
    newJ = greedy_search(J, eta, zeta, beta, conv_margin)

    # Plot the Original Image
    plot_image(I, 'Original Image', 'img/Original_Image')

    # Plot the Denoised Image
    plot_image(newJ, 'Denoised version', 'img/Denoised_Image')
    
    # Compute the reconstruction error 
    reconstruction_error = f_reconstruction_error(I, newJ)
    print('Reconstruction Error: ', reconstruction_error)

    
if __name__ == "__main__":
    main()