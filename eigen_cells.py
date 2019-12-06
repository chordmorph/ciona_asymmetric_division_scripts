import numpy as np
from skimage.io import imread, imsave
from scipy.ndimage import (
    zoom, binary_dilation, binary_opening, binary_closing,
    rotate, binary_erosion)
from mahotas import border
from sympy import Plane, Point3D, Line3D, N, solve, Point

from spindle_analysis_functions import (
    apical_surface_extractor, centroid_finder, apical_belt_extractor_planes,
    coordinate_creator, pca_analysis, pc_apical_relationship,
    )


def second_last_eigen_cell_creator(
    x_scale, y_scale, z_scale, cell_label, labelmatrix_path, outside_label,
    ant_pole_x, ant_pole_y, ant_pole_z,post_pole_x, post_pole_y, post_pole_z,
    image_name, cell_name
    ):


    x_scale = x_scale
    y_scale = y_scale
    z_scale = z_scale
    cell_label = cell_label

    #import labelmatrix
    labelmatrix = np.swapaxes(imread(labelmatrix_path), 0, 2)

    #create ndarrays for different entities
    apical_surface = apical_surface_extractor(labelmatrix, outside_label, cell_label)
    anterior_pole = np.zeros(labelmatrix.shape)
    posterior_pole = np.zeros(labelmatrix.shape)
    anterior_pole[ant_pole_x, ant_pole_y, ant_pole_z] = 1
    posterior_pole[post_pole_x, post_pole_y, post_pole_z] = 1
    anterior_pole = binary_dilation(anterior_pole, structure=np.ones((3,3,3)).astype(np.int), iterations = 3)
    posterior_pole = binary_dilation(posterior_pole, structure=np.ones((3,3,3)).astype(np.int), iterations = 3)
    membrane = border(labelmatrix, 0, cell_label).astype(np.uint16)
    original_cell_centroid = (centroid_finder(labelmatrix, x_scale, z_scale, cell_label))
    apical_belt = apical_belt_extractor_planes(
                labelmatrix, outside_lab, cell_label, 3,
                x_scale, y_scale, z_scale, original_cell_centroid,
                return_image = False, apical_belt_name = None)
    embryo = np.array(labelmatrix != outside_lab).astype(np.uint16)

    #rescale to get isotropic voxels
    scaling_factor = x_scale/z_scale
    resized_labelmatrix = zoom(labelmatrix, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_apical = zoom(apical_surface, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_anterior_pole = zoom(anterior_pole, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_posterior_pole = zoom(posterior_pole, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_membrane = zoom(membrane, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_apical_belt = zoom(apical_belt, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_embryo = zoom(embryo, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')

    #now that voxels are isotropic, we'll just call the voxel size 1 for simplicity
    new_x_scale = 1
    new_y_scale = 1
    new_z_scale = 1

    #make everything not the entity of interest zero and the entity label 1
    resized_cell_only_labelmatrix = np.array(resized_labelmatrix == cell_label).astype(np.uint16)
    resized_apical_labelmatrix = np.array(resized_apical == 1).astype(np.uint16)
    resized_anterior_pole_labelmatrix = np.array(resized_anterior_pole == 1).astype(np.uint16)
    resized_posterior_pole_labelmatrix = np.array(resized_posterior_pole == 1).astype(np.uint16)
    resized_membrane_labelmatrix = np.array(resized_membrane == 1).astype(np.uint16)
    resized_apical_belt_labelmatrix = np.array(resized_apical_belt == 1).astype(np.uint16)
    resized_embryo_labelmatrix = np.array(resized_embryo == 1).astype(np.uint16)

    #remove any extra labels (this occurs often for some reason)
    opened = binary_opening(resized_cell_only_labelmatrix)

    #find the position the centroid of the cell
    offset = (centroid_finder(opened, new_x_scale, new_z_scale, 1))

    #find the coordinates and PCs of the cell
    coordinates_opened = coordinate_creator(opened, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_cell_coordinates = coordinates_opened - offset
    apical_coordinates = coordinate_creator(resized_apical_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_apical_coordinates = apical_coordinates - offset
    anterior_pole_coordinates = coordinate_creator(resized_anterior_pole_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_anterior_pole_coordinates = anterior_pole_coordinates - offset
    posterior_pole_coordinates = coordinate_creator(resized_posterior_pole_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_posterior_pole_coordinates = posterior_pole_coordinates - offset
    membrane_coordinates = coordinate_creator(resized_membrane_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_membrane_coordinates = membrane_coordinates - offset
    apical_belt_coordinates = coordinate_creator(resized_apical_belt_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_apical_belt_coordinates = apical_belt_coordinates - offset
    embryo_coordinates = coordinate_creator(resized_embryo_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_embryo_coordinates = embryo_coordinates - offset

    pc1, pc2, pc3, cell_midpoint, pc1_loading, pc2_loading, pc3_loading = pca_analysis(shifted_cell_coordinates)
    embryo_pc1, embryo_pc2, embryo_pc3, embryo_midpoint, embryo_loading1, embryo_loading2, embryo_loading3 = pca_analysis(shifted_embryo_coordinates)


    #defining basis vectors for rotation
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    #find rotation matrix to transform C1-C3 to the x, y and z axes
    apical_pc1, apical_pc2, apical_pc3, apical_midpoint, apical_pc1_loading, apical_pc2_loading, apical_pc3_loading = pca_analysis(shifted_apical_coordinates)
    apical_plane = Plane(Point3D(apical_midpoint),
                         Point3D(apical_midpoint + apical_pc1),
                         Point3D(apical_midpoint + apical_pc2))
    cell_pc1_line = Line3D(Point3D(cell_midpoint), Point3D(cell_midpoint + pc1))
    cell_pc2_line = Line3D(Point3D(cell_midpoint), Point3D(cell_midpoint + pc2))
    cell_pc3_line = Line3D(Point3D(cell_midpoint), Point3D(cell_midpoint + pc3))
    ortho_pc, largest_apical_plane_pc, smallest_apical_plane_pc = pc_apical_relationship(
        cell_pc1_line, cell_pc2_line, cell_pc3_line, apical_plane,
        pc1_loading, pc2_loading, pc3_loading)
    smallest_apical_plane_pc_projection = apical_plane.projection_line(smallest_apical_plane_pc)
    largest_apical_plane_pc_projection = apical_plane.projection_line(largest_apical_plane_pc)
    C1_C2_plane = Plane(Point3D(apical_midpoint),
                        Point3D(apical_midpoint + largest_apical_plane_pc_projection.direction_ratio),
                        Point3D(apical_midpoint + apical_plane.normal_vector))
        V0 = np.array([x_axis, y_axis, z_axis])
    V2 = np.array([np.array(largest_apical_plane_pc_projection.direction_ratio).astype(np.float),
                  np.array(apical_plane.normal_vector).astype(np.float),
                  np.array(C1_C2_plane.normal_vector).astype(np.float)])
    M2 = np.linalg.solve(V0,V2).T

    #transform cell labelmatrix onto new coordinate system
    rotated_cell_coordinates2 = np.zeros(shifted_cell_coordinates.shape)
    for i in range(shifted_cell_coordinates.shape[0]):
        rotated_cell_coordinates2[i] = np.matmul(shifted_cell_coordinates[i], M2).round()

    rotated_apical_coordinates2 = np.zeros(shifted_apical_coordinates.shape)
    for i in range(shifted_apical_coordinates.shape[0]):
        rotated_apical_coordinates2[i] = np.matmul(shifted_apical_coordinates[i], M2).round()

    rotated_anterior_pole_coordinates2 = np.zeros(shifted_anterior_pole_coordinates.shape)
    for i in range(shifted_anterior_pole_coordinates.shape[0]):
        rotated_anterior_pole_coordinates2[i] = np.matmul(shifted_anterior_pole_coordinates[i], M2).round()

    rotated_posterior_pole_coordinates2 = np.zeros(shifted_posterior_pole_coordinates.shape)
    for i in range(shifted_posterior_pole_coordinates.shape[0]):
        rotated_posterior_pole_coordinates2[i] = np.matmul(shifted_posterior_pole_coordinates[i], M2).round()

    rotated_membrane_coordinates2 = np.zeros(shifted_membrane_coordinates.shape)
    for i in range(shifted_membrane_coordinates.shape[0]):
        rotated_membrane_coordinates2[i] = np.matmul(shifted_membrane_coordinates[i], M2).round()

    rotated_apical_belt_coordinates2 = np.zeros(shifted_apical_belt_coordinates.shape)
    for i in range(shifted_apical_belt_coordinates.shape[0]):
        rotated_apical_belt_coordinates2[i] = np.matmul(shifted_apical_belt_coordinates[i], M2).round()

    #rotated_embryo_coordinates2 = np.zeros(embryo_midpoint.shape)
    rotated_embryo_coordinates2 = np.zeros([1,3])
    rotated_embryo_coordinates2[0] = np.matmul(embryo_midpoint, M2).round()

    #shift the cell back so that none of the cell is clipped
    x_shift = 250
    y_shift = 250
    z_shift = 250
    shifting_array = np.array([x_shift, y_shift, z_shift])

    rotated_shifted_cell_coordinates2 = rotated_cell_coordinates2 + shifting_array
    rotated_shifted_apical_coordinates2 = rotated_apical_coordinates2 + shifting_array
    rotated_shifted_anterior_pole_coordinates2 = rotated_anterior_pole_coordinates2 + shifting_array
    rotated_shifted_posterior_pole_coordinates2 = rotated_posterior_pole_coordinates2 + shifting_array
    rotated_shifted_membrane_coordinates2 = rotated_membrane_coordinates2 + shifting_array
    rotated_shifted_apical_belt_coordinates2 = rotated_apical_belt_coordinates2 + shifting_array
    rotated_shifted_embryo_coordinates2 = rotated_embryo_coordinates2 + shifting_array

    #transform coordinates to a labelmatrix
    blank_cell2 = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_cell_coordinates2.shape[0]):
        blank_cell2[
            int(rotated_shifted_cell_coordinates2[j,0]),
            int(rotated_shifted_cell_coordinates2[j,1]),
            int(rotated_shifted_cell_coordinates2[j,2])] = 1

    blank_apical2 = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_apical_coordinates2.shape[0]):
        blank_apical2[
            int(rotated_shifted_apical_coordinates2[j,0]),
            int(rotated_shifted_apical_coordinates2[j,1]),
            int(rotated_shifted_apical_coordinates2[j,2])] = 1

    blank_anterior_pole2 = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_anterior_pole_coordinates2.shape[0]):
        blank_anterior_pole2[
            int(rotated_shifted_anterior_pole_coordinates2[j,0]),
            int(rotated_shifted_anterior_pole_coordinates2[j,1]),
            int(rotated_shifted_anterior_pole_coordinates2[j,2])] = 1

    blank_posterior_pole2 = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_posterior_pole_coordinates2.shape[0]):
        blank_posterior_pole2[
            int(rotated_shifted_posterior_pole_coordinates2[j,0]),
            int(rotated_shifted_posterior_pole_coordinates2[j,1]),
            int(rotated_shifted_posterior_pole_coordinates2[j,2])] = 1

    blank_membrane2 = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_membrane_coordinates2.shape[0]):
        blank_membrane2[
            int(rotated_shifted_membrane_coordinates2[j,0]),
            int(rotated_shifted_membrane_coordinates2[j,1]),
            int(rotated_shifted_membrane_coordinates2[j,2])] = 1

    blank_apical_belt_2 = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_apical_belt_coordinates2.shape[0]):
        blank_apical_belt_2[
            int(rotated_shifted_apical_belt_coordinates2[j,0]),
            int(rotated_shifted_apical_belt_coordinates2[j,1]),
            int(rotated_shifted_apical_belt_coordinates2[j,2])] = 1

    blank_embryo_2 = np.zeros([500, 500, 500])
    blank_embryo_2[
        int(rotated_shifted_embryo_coordinates2[0,0]),
        int(rotated_shifted_embryo_coordinates2[0,1]),
        int(rotated_shifted_embryo_coordinates2[0,2])] = 1

    #closing any gaps in the labelmatrix

    dilated_cell2 = binary_dilation(blank_cell2, structure=np.ones((3,3,3)).astype(np.int), iterations = 1)
    dilated_apical2 = binary_dilation(blank_apical2, structure=np.ones((3,3,3)).astype(np.int), iterations = 1)
    dilated_membrane2 = binary_dilation(blank_membrane2, structure=np.ones((3,3,3)).astype(np.int), iterations = 1)
    dilated_apical_belt2 = binary_dilation(blank_apical_belt_2, structure=np.ones((3,3,3)).astype(np.int), iterations = 2)
    dilated_embryo2 = binary_dilation(blank_embryo_2, structure=np.ones((3,3,3)).astype(np.int), iterations = 3)

    closed_cell2 = binary_closing(dilated_cell2, structure=np.ones((2,2,2)).astype(np.int))
    closed_apical2 = binary_closing(dilated_apical2, structure=np.ones((2,2,2)).astype(np.int))
    closed_anterior_pole2 = binary_closing(blank_anterior_pole2, structure=np.ones((2,2,2)).astype(np.int))
    closed_posterior_pole2 = binary_closing(blank_posterior_pole2, structure=np.ones((2,2,2)).astype(np.int))
    closed_membrane2 = binary_closing(dilated_membrane2, structure=np.ones((2,2,2)).astype(np.int))
    closed_apical_belt2 = binary_closing(dilated_apical_belt2, structure=np.ones((2,2,2)).astype(np.int))

    #reflecting volume so that apical surfaces align
    if rotated_shifted_apical_coordinates2[:, 1].mean() >= 250:
        closed_cell2 = np.flip(closed_cell2, axis = 1)
        closed_apical2 = np.flip(closed_apical2, axis = 1)
        closed_anterior_pole2 = np.flip(closed_anterior_pole2, axis = 1)
        closed_posterior_pole2 = np.flip(closed_posterior_pole2, axis = 1)
        closed_membrane2 = np.flip(closed_membrane2, axis = 1)
        closed_apical_belt2 = np.flip(closed_apical_belt2, axis = 1)
        dilated_embryo2 = np.flip(dilated_embryo2, axis = 1)
    else:
        True


    #reflecting volume so that anterior and posterior poles align
    final_anterior_pole_coordinates2 = coordinate_creator(
        closed_anterior_pole2, 1, new_x_scale, new_y_scale, new_z_scale)
    final_posterior_pole_coordinates2 = coordinate_creator(
        closed_posterior_pole2, 1, new_x_scale, new_y_scale, new_z_scale)

    if final_anterior_pole_coordinates2[:, 0].mean() >= final_posterior_pole_coordinates2[:, 0].mean():
        closed_cell2 = np.flip(closed_cell2, axis = 0)
        closed_apical2 = np.flip(closed_apical2, axis = 0)
        closed_anterior_pole2 = np.flip(closed_anterior_pole2, axis = 0)
        closed_posterior_pole2 = np.flip(closed_posterior_pole2, axis = 0)
        closed_membrane2 = np.flip(closed_membrane2, axis = 0)
        closed_apical_belt2 = np.flip(closed_apical_belt2, axis = 0)
        dilated_embryo2 = np.flip(dilated_embryo2, axis = 0)
    else:
        True

    #reflecting volume to make all cells "right sided"
    final_embryo_coordinates2 = coordinate_creator(
        dilated_embryo2, 1, new_x_scale, new_y_scale, new_z_scale)

    if final_embryo_coordinates2[:, 2].mean() >= 250:
        closed_cell2 = np.flip(closed_cell2, axis = 2)
        closed_apical2 = np.flip(closed_apical2, axis = 2)
        closed_anterior_pole2 = np.flip(closed_anterior_pole2, axis = 2)
        closed_posterior_pole2 = np.flip(closed_posterior_pole2, axis = 2)
        closed_membrane2 = np.flip(closed_membrane2, axis = 2)
        closed_apical_belt2 = np.flip(closed_apical_belt2, axis = 2)
    else:
        True

    #saving images
    imsave('{}/{}_{}_registered_cell2.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_cell2, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_apical2.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_apical2, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_anterior_pole2.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_anterior_pole2, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_posterior_pole2.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_posterior_pole2, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_membrane2.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_membrane2, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_apical_belt2.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_apical_belt2, 0, 2).astype(np.uint16))


def final_div_eigen_cell_creator(
    x_scale, y_scale, z_scale, cell_label, labelmatrix_path, outside_label,
    ant_pole_x, ant_pole_y, ant_pole_z, post_pole_x, post_pole_y, post_pole_z,
    ant_point_x, ant_point_y, dorsal, primordium_labelmatrix_path, image_name,
    primordium_label, embryo_half, cell_name):

    x_scale = x_scale
    y_scale = y_scale
    z_scale = z_scale
    cell_label = cell_label

    #import labelmatrix
    labelmatrix = np.swapaxes(imread(labelmatrix_path), 0, 2)

    #create ndarrays for apical surface and spindle poles

    anterior_pole = np.zeros(labelmatrix.shape)
    posterior_pole = np.zeros(labelmatrix.shape)
    anterior_point = np.zeros(labelmatrix.shape)
    anterior_pole[ant_pole_x, ant_pole_y, ant_pole_z] = 1
    posterior_pole[post_pole_x, post_pole_y, post_pole_z] = 1
    anterior_point[ant_point_x, ant_point_y, 20] = 1
    anterior_pole = binary_dilation(anterior_pole, structure=np.ones((3,3,3)).astype(np.int), iterations = 3)
    anterior_point = binary_dilation(anterior_point, structure=np.ones((3,3,3)).astype(np.int), iterations = 3)
    posterior_pole = binary_dilation(posterior_pole, structure=np.ones((3,3,3)).astype(np.int), iterations = 3)
    membrane = border(labelmatrix, 0, cell_label).astype(np.uint16)
    dorsal_point = np.zeros(labelmatrix.shape)
    dorsal_dict = {
        'top': (np.array(labelmatrix.shape) * np.array([.5, .5, 0])),
        'bottom': (np.array(labelmatrix.shape) * np.array([.5, .5, .9])),
        'front': (np.array(labelmatrix.shape) * np.array([.5, 0, .5])),
        'back': (np.array(labelmatrix.shape) * np.array([.5, .9, .5])),
    }

    dorsal_point[
        int(dorsal_dict[dorsal][0]),
        int(dorsal_dict[dorsal][1]),
        int(dorsal_dict[dorsal][2])] = 1
    dorsal_point = binary_dilation(dorsal_point, structure=np.ones((3,3,3)).astype(np.int), iterations = 3)


    #rescale to get isotropic voxels
    scaling_factor = x_scale/z_scale
    resized_labelmatrix = zoom(labelmatrix, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_anterior_pole = zoom(anterior_pole, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_posterior_pole = zoom(posterior_pole, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_membrane = zoom(membrane, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_dorsal_point = zoom(dorsal_point, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')
    resized_anterior_point = zoom(anterior_point, zoom = [scaling_factor, scaling_factor, 1], order = 1, mode = 'nearest')

    #now that voxels are isotropic, we'll just call the voxel size 1 for simplicity
    new_x_scale = 1
    new_y_scale = 1
    new_z_scale = 1

    #make everything not the cell of interest zero and the cell label 1
    resized_cell_only_labelmatrix = np.array(resized_labelmatrix == cell_label).astype(np.uint16)
    resized_anterior_pole_labelmatrix = np.array(resized_anterior_pole == 1).astype(np.uint16)
    resized_posterior_pole_labelmatrix = np.array(resized_posterior_pole == 1).astype(np.uint16)
    resized_membrane_labelmatrix = np.array(resized_membrane == 1).astype(np.uint16)
    resized_dorsal_point = np.array(resized_dorsal_point == 1).astype(np.uint16)
    resized_anterior_point_labelmatrix = np.array(resized_anterior_point == 1).astype(np.uint16)

    #remove any extra labels (this occurs often for some reason)
    opened = binary_opening(resized_cell_only_labelmatrix)

    #find the position the centroid of the cell
    offset = (centroid_finder(opened, new_x_scale, new_z_scale, 1))

    #find the coordinates and PCs of the cell
    coordinates_opened = coordinate_creator(opened, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_cell_coordinates = coordinates_opened - offset
    anterior_pole_coordinates = coordinate_creator(resized_anterior_pole_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_anterior_pole_coordinates = anterior_pole_coordinates - offset
    posterior_pole_coordinates = coordinate_creator(resized_posterior_pole_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_posterior_pole_coordinates = posterior_pole_coordinates - offset
    membrane_coordinates = coordinate_creator(resized_membrane_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_membrane_coordinates = membrane_coordinates - offset
    dorsal_point_coordinates = coordinate_creator(resized_dorsal_point, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_dorsal_coordinates = dorsal_point_coordinates - offset
    anterior_point_coordinates = coordinate_creator(resized_anterior_point_labelmatrix, 1, new_x_scale, new_y_scale, new_z_scale)
    shifted_anterior_point_coordinates = anterior_point_coordinates - offset

    pc1, pc2, pc3, cell_midpoint, pc1_loading, pc2_loading, pc3_loading = pca_analysis(shifted_cell_coordinates)

    apical_labelmatrix = np.swapaxes(
        imread(primordium_labelmatrix_path), 0, 2)

    apical_coordinates = coordinate_creator(
        apical_labelmatrix, primordium_label, x_scale, y_scale, z_scale)

    apical_pc1, apical_pc2, apical_pc3, apical_midpoint, apical_pc1_loading, apical_pc2_loading, apical_pc3_loading = pca_analysis(
        apical_coordinates)

    cell_pc1_line = Line3D(Point3D(cell_midpoint), Point3D(cell_midpoint + pc1))
    cell_pc2_line = Line3D(Point3D(cell_midpoint), Point3D(cell_midpoint + pc2))
    cell_pc3_line = Line3D(Point3D(cell_midpoint), Point3D(cell_midpoint + pc3))

    apical_plane = Plane(Point3D(apical_midpoint),
                         Point3D(apical_midpoint + apical_pc1),
                         Point3D(apical_midpoint + apical_pc2))

    ortho_pc, largest_apical_plane_pc, smallest_apical_plane_pc = pc_apical_relationship(
        cell_pc1_line, cell_pc2_line, cell_pc3_line, apical_plane,
        pc1_loading, pc2_loading, pc3_loading)

    C1 = apical_plane.normal_vector
    C2 = apical_plane.projection_line(largest_apical_plane_pc)
    C3 = apical_plane.projection_line(smallest_apical_plane_pc)

    #defining basis vectors for rotation
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    #find rotation matrix to transform the PCs to the x, y, and z axes
    V0 = np.array([x_axis, y_axis, z_axis])
    V1 = np.array([
        np.array(C2.direction_ratio, dtype = 'float'),
        np.array(C1, dtype = 'float'),
        np.array(C3.direction_ratio, dtype = 'float')])
    M = np.linalg.solve(V0,V1).T

        #transform cell labelmatrix onto new coordinate system
    rotated_cell_coordinates = np.zeros(shifted_cell_coordinates.shape)
    for i in range(shifted_cell_coordinates.shape[0]):
        rotated_cell_coordinates[i] = np.matmul(shifted_cell_coordinates[i], M).round()

    rotated_anterior_pole_coordinates = np.zeros(shifted_anterior_pole_coordinates.shape)
    for i in range(shifted_anterior_pole_coordinates.shape[0]):
        rotated_anterior_pole_coordinates[i] = np.matmul(shifted_anterior_pole_coordinates[i], M).round()

    rotated_posterior_pole_coordinates = np.zeros(shifted_posterior_pole_coordinates.shape)
    for i in range(shifted_posterior_pole_coordinates.shape[0]):
        rotated_posterior_pole_coordinates[i] = np.matmul(shifted_posterior_pole_coordinates[i], M).round()

    rotated_membrane_coordinates = np.zeros(shifted_membrane_coordinates.shape)
    for i in range(shifted_membrane_coordinates.shape[0]):
        rotated_membrane_coordinates[i] = np.matmul(shifted_membrane_coordinates[i], M).round()

    rotated_dorsal_coordinates = np.zeros(shifted_dorsal_coordinates.shape)
    for i in range(shifted_dorsal_coordinates.shape[0]):
        rotated_dorsal_coordinates[i] = np.matmul(shifted_dorsal_coordinates[i], M).round()

    rotated_anterior_point_coordinates = np.zeros(shifted_anterior_point_coordinates.shape)
    for i in range(shifted_anterior_point_coordinates.shape[0]):
        rotated_anterior_point_coordinates[i] = np.matmul(shifted_anterior_point_coordinates[i], M).round()

    #shift the cell back so that none of the cell is clipped
    x_shift = 250
    y_shift = 250
    z_shift = 250
    shifting_array = np.array([x_shift, y_shift, z_shift])

    rotated_shifted_cell_coordinates = rotated_cell_coordinates + shifting_array
    rotated_shifted_anterior_pole_coordinates = rotated_anterior_pole_coordinates + shifting_array
    rotated_shifted_posterior_pole_coordinates = rotated_posterior_pole_coordinates + shifting_array
    rotated_shifted_membrane_coordinates = rotated_membrane_coordinates + shifting_array
    rotated_shifted_dorsal_coordinates = rotated_dorsal_coordinates + shifting_array
    rotated_shifted_anterior_point_coordinates = rotated_anterior_point_coordinates + shifting_array

    #transform coordinates to a labelmatrix
    blank_cell = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_cell_coordinates.shape[0]):
        blank_cell[
            int(rotated_shifted_cell_coordinates[j,0]),
            int(rotated_shifted_cell_coordinates[j,1]),
            int(rotated_shifted_cell_coordinates[j,2])] = 1

    blank_anterior_pole = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_anterior_pole_coordinates.shape[0]):
        blank_anterior_pole[
            int(rotated_shifted_anterior_pole_coordinates[j,0]),
            int(rotated_shifted_anterior_pole_coordinates[j,1]),
            int(rotated_shifted_anterior_pole_coordinates[j,2])] = 1

    blank_posterior_pole = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_posterior_pole_coordinates.shape[0]):
        blank_posterior_pole[
            int(rotated_shifted_posterior_pole_coordinates[j,0]),
            int(rotated_shifted_posterior_pole_coordinates[j,1]),
            int(rotated_shifted_posterior_pole_coordinates[j,2])] = 1

    blank_membrane = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_membrane_coordinates.shape[0]):
        blank_membrane[
            int(rotated_shifted_membrane_coordinates[j,0]),
            int(rotated_shifted_membrane_coordinates[j,1]),
            int(rotated_shifted_membrane_coordinates[j,2])] = 1

    blank_dorsal = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_dorsal_coordinates.shape[0]):
        blank_dorsal[
            int(rotated_shifted_dorsal_coordinates[j,0]),
            int(rotated_shifted_dorsal_coordinates[j,1]),
            int(rotated_shifted_dorsal_coordinates[j,2])] = 1

    blank_anterior_point = np.zeros([500, 500, 500])
    for j in range(rotated_shifted_anterior_point_coordinates.shape[0]):
        blank_anterior_point[
            int(rotated_shifted_anterior_point_coordinates[j,0]),
            int(rotated_shifted_anterior_point_coordinates[j,1]),
            int(rotated_shifted_anterior_point_coordinates[j,2])] = 1


    #closing any gaps in the labelmatrix
    closed_cell = binary_closing(blank_cell)
    closed_anterior_pole = binary_closing(blank_anterior_pole)
    closed_posterior_pole = binary_closing(blank_posterior_pole)
    closed_membrane = binary_closing(blank_membrane)
    closed_dorsal = binary_closing(blank_dorsal)
    closed_anterior_point = binary_closing(blank_anterior_pole)


    #reflecting volumes so that anterior and posterior are oriented properly
    final_anterior_pole_coordinates = coordinate_creator(
        closed_anterior_pole, 1, new_x_scale, new_y_scale, new_z_scale)
    final_posterior_pole_coordinates = coordinate_creator(
        closed_posterior_pole, 1, new_x_scale, new_y_scale, new_z_scale)

    if final_anterior_pole_coordinates[:, 0].mean() >= final_posterior_pole_coordinates[:, 0].mean():
        closed_cell = np.flip(closed_cell, axis = 0)
        closed_anterior_pole = np.flip(closed_anterior_pole,  axis = 0)
        closed_posterior_pole = np.flip(closed_posterior_pole,  axis = 0)
        closed_membrane = np.flip(closed_membrane, axis = 0)
        closed_dorsal = np.flip(closed_dorsal, axis = 0)
    else:
        True

    #reflecting volumes so that dorsal surfaces are oriented properly
    final_dorsal_coordinates = coordinate_creator(
        closed_dorsal, 1, new_x_scale, new_y_scale, new_z_scale)

    if final_dorsal_coordinates[:, 1].mean() < 250:
        closed_cell = np.flip(closed_cell,  axis = 1)
        closed_anterior_pole = np.flip(closed_anterior_pole, axis = 1)
        closed_posterior_pole = np.flip(closed_posterior_pole, axis = 1)
        closed_membrane = np.flip(closed_membrane, axis = 1)
    else:
        True

    #reflecting volumes to make all cells "right sided"
    if embryo_half == 'top':
        if dorsal == 'bottom':
            reflect = 'yes'
        else:
            reflect = 'no'
    else:
        if dorsal == 'top':
            reflect = 'yes'
        else:
            reflect = 'no'

    if reflect == 'yes':
        closed_cell = np.flip(closed_cell, axis = 2)
        closed_anterior_pole = np.flip(closed_anterior_pole, axis = 2)
        closed_posterior_pole = np.flip(closed_posterior_pole, axis = 2)
        closed_membrane = np.flip(closed_membrane, axis = 2)
    else:
        True

    #saving images
    imsave('{}/{}_{}_registered_cell.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_cell, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_anterior_pole.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_anterior_pole, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_posterior_pole.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_posterior_pole, 0, 2).astype(np.uint16))
    imsave('{}/{}_{}_registered_membrane.tif'.format(folder_name, image_name, cell_name),np.swapaxes(closed_membrane, 0, 2).astype(np.uint16))
