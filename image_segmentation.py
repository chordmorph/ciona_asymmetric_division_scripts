from skimage.morphology import watershed
import numpy as np
from skimage.io import imread, imsave
from mahotas import label
from scipy.ndimage import binary_erosion



def volume_outputs(labelmatrix, file_name, voxel_volume = 0.027):

    '''Writes a .txt file of the volume of each label in the given label
    matrix.'''

    x = np.unique(labelmatrix, return_counts = True)
    labels = list(x[0])
    volumes = list(x[1] * voxel_volume)
    file = open('{}.txt'.format(file_name), 'w')
    old_stdout = sys.stdout
    sys.stdout = file
    for cells in range(len(labels)):
        print('{}     {}'.format(labels[cells], volumes[cells]))
        print()
    sys.stdout = old_stdout
    file.close



def voxel_outputs(labelmatrix, file_name):

    '''Writes a .txt file of the number of voxels contained in each
    label in the given label matrix.'''

    x = np.unique(labelmatrix, return_counts = True)
    labels = list(x[0])
    voxels = list(x[1])
    file = open('{}.txt'.format(file_name), 'w')
    old_stdout = sys.stdout
    sys.stdout = file
    for cells in range(len(labels)):
        print('{}     {}'.format(labels[cells], voxels[cells]))
        print()
    sys.stdout = old_stdout
    file.close



def double_segment(membrane_name, mask_name, label_matrix_name,
                   dilations = 4, output_type = 'volumes'):

    '''Uses the seeds from the image at the "mask_name" file path
    to perform a seeded watershed segmentation on the image at the
    "membrane_name" file path. This segmentation is performed twice
    with a binary erosion step in between for improved segmentation.'''

    #read images from file
    memb = np.swapaxes(imread(membrane_name), 0, 2)
    mask = np.swapaxes(imread(mask_name), 0, 2)

    #create seeds from mask image
    seeds = label(mask, Bc = 8)[0]

    #first watershed segmentation
    label_matrix = np.uint16(watershed(memb, seeds, watershed_line = True))

    #binary erosion at the membrane between cells
    membranes = (label_matrix != 0).astype(np.uint16)
    eroded_memb = binary_erosion(
        membranes, iterations = dilations).astype(np.uint16)

    #creating new seeds and second watershed segmentation
    new_seeds = label(eroded_memb, Bc = 8)[0]
    new_label_matrix = np.uint16(
        watershed(memb, new_seeds, watershed_line = True))

    #saving image
    imsave('{} labelmatrix.tif'.format(label_matrix_name),
           np.swapaxes(new_label_matrix, 0, 2))

    #output of cell volumes or cell voxel counts depending on input image
    if output_type == 'volumes':
        volume_outputs(new_label_matrix, '{} volumes'.format(label_matrix_name))
    elif output_type == 'voxels':
        voxel_outputs(new_label_matrix, '{} voxels'.format(label_matrix_name))
    else:
        print('Please specify an output format of either "voxels" or "volumes"')
