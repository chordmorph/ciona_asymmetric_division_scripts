from vigra import impex, blockwise, sampling
import numpy as np
from veeman_lab_scripts.image_input.czifile import CziFile


def channel_slicer(czi_file, channel):

    '''Returns the channel of interest from a multi-channel 3-dimensional
    .czi (Zeiss) file as a 3D numpy array.'''

    axes_list = [czi_file.axes[i] for i in range(len(czi_file.axes))]
    channel_axis = axes_list.index(67)
    z_axis = axes_list.index(90)
    x_axis = axes_list.index(88)
    y_axis = axes_list.index(89)

    full_image = czi_file.asarray()
    slices = full_image.shape

    slicer = []
    for i in range(len(slices)):
        if i == channel_axis:
            slicer.append(channel - 1)
        elif i == z_axis:
            slicer.append(slice(None,None))
        elif i == x_axis:
            slicer.append(slice(None,None))
        elif i == y_axis:
            slicer.append(slice(None,None))
        else:
            slicer.append(0)

    single_channel_stack = full_image[slicer]
    single_channel_stack = np.swapaxes(single_channel_stack, 0, 2)

    return single_channel_stack



def image_filter(filepath, image_name, scale_image, sigma = 3, channel = 3):

    '''Reads a .czi (Zeiss) file, and rescales the z-dimension to get
    isotropic voxels if desired, then performs a VIGRA second-derivate filter on
    the channel of interest with a given pre-filter gaussian blur.'''

    #reading image and extracting the channel of choice
    czi_image = CziFile(filepath)
    single_channel_stack = channel_slicer(czi_image, channel)

    if scale_image == True:
        #get voxel size for scaling
        image_metadata = czi_image.metadata
        size_element = image_metadata[0][3][0][0]
        if size_element.attrib.get(size_element.attrib.keys()[0]) == 'X':
            pixel_size = float(size_element[0].text)
        else:
            print('Pixel size not properly located in metadata')
        scaling_factor = pixel_size/0.0000003

        #set new dimensions of output image with isotropic voxels and rescale
        new_dimensions = [
          int(round(single_channel_stack.shape[0]*scaling_factor)),
          int(round(single_channel_stack.shape[1]*scaling_factor)),
          single_channel_stack.shape[2]]
        single_channel_stack = sampling.resize(single_channel_stack, shape = new_dimensions)

    else:
        continue

    #set pre-filter gaussian blur radius
    options = blockwise.BlockwiseConvolutionOptions3D()
    options.stdDev = (sigma, )*3

    #run VIGRA filter and save image
    filtered = blockwise.hessianOfGaussianLastEigenvalue(
      single_channel_stack.astype(np.float32), options)
    filtered = np.max(filtered) - filtered
    scale = ((2 ** 16) * .9)/ filtered.max()
    filtered = filtered * scale
    impex.writeVolume(
      filtered.astype(np.uint16), '{}.tif'.format(image_name), '')
