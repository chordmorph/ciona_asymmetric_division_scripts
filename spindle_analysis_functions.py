import numpy as np
from sklearn.decomposition import PCA as skPCA
from mahotas import border
from scipy.ndimage import zoom, binary_closing, binary_dilation
from scipy.ndimage.measurements import center_of_mass
from sympy import Plane, Point3D, N, Symbol, lambdify
from tifffile import imsave

def coordinate_creator(labelmatrix, label, x_scale, y_scale, z_scale):
    """Returns an [n,3] array of x, y, and z coordinates in µm where the
    labelmatrix is equal to a particular label.
    """
    cell = labelmatrix == label
    x, y, z = np.where(cell)
    x = x * x_scale
    y = y * y_scale
    z = z * z_scale
    coordinates = np.array([x, y, z]).T

    return coordinates

def apical_surface_extractor(
            labelmatrix, outside_label, cell_label,
            return_image = False, apical_surface_name = None
            ):
    """Finds the apical surface of a cell from a labelmatrix given the
    label of the cell and the label marking exterior of the embryo.
    Returns a matrix the same size as labelmatrix where the apical
    surface is equal to 1, and writes this labelmatrix as a tiff if
    desired.
    """
    memb_cell_border = border(labelmatrix, 0, cell_label).astype(np.uint16)
    memb_outside_border = border(
        labelmatrix, 0, outside_label).astype(np.uint16)
    apical = memb_outside_border + memb_cell_border
    apical[apical < 2] = 0
    apical[apical == 2] = 1
    if return_image == True:
        imsave('{}.tif'.format(apical_surface_name), np.swapaxes(apical, 0, 2))
        return apical
    else:
        return apical

def pca_analysis(coordinates):
    """Preforms a scikit-learn PCA on an array of 3-dimensional
    coordinates and returns the 3 principal component direction vectors.
    """
    pca = skPCA(n_components = 3)
    pca_results = pca.fit(coordinates)
    pc1 = pca_results.components_[0]
    pc2 = pca_results.components_[1]
    pc3 = pca_results.components_[2]
    pca_midpoint = pca_results.mean_
    pc1_loading = pca_results.explained_variance_ratio_[0]
    pc2_loading = pca_results.explained_variance_ratio_[1]
    pc3_loading = pca_results.explained_variance_ratio_[2]

    return pc1, pc2, pc3, pca_midpoint, pc1_loading, pc2_loading, pc3_loading

def apical_ring_extractor(
            labelmatrix, outside_label, cell_label,
            return_image = False, apical_surface_name = None
            ):
    """Finds the apical surface of a cell from a labelmatrix given the
    label of the cell and the label marking exterior of the embryo.
    Returns a matrix the same size as labelmatrix where the apical
    ring is equal to 1, and writes this labelmatrix as a tiff if
    desired.
    """
    memb_cell_border = border(labelmatrix, 0, cell_label).astype(np.uint16)
    memb_outside_border = border(
        labelmatrix, 0, outside_label).astype(np.uint16)
    apical = memb_outside_border + memb_cell_border
    apical[apical < 2] = 0
    apical[apical == 2] = 1
    apical_dil = binary_dilation(apical, iterations = 2)
    cell_memb_minus_apical = memb_cell_border - apical_dil
    cell_memb_minus_apical = cell_memb_minus_apical * 2
    combined = apical_dil + cell_memb_minus_apical
    intersection = border(combined, 1, 2).astype(np.uint16)
    if return_image == True:
        imsave('{}.tif'.format(apical_surface_name), np.swapaxes(
        intersection, 0, 2))
        return intersection
    else:
        return intersection

def apical_belt_extractor_planes(
            labelmatrix, outside_label, cell_label, belt_depth,
            x_scale, y_scale, z_scale, cell_centroid_point,
            return_image = False, apical_belt_name = None
            ):
    apical_ring_labelmatrix = apical_ring_extractor(
        labelmatrix, outside_label, cell_label,
        return_image = return_image, apical_surface_name = apical_belt_name)
    apical_ring_coordinates = coordinate_creator(
          apical_ring_labelmatrix, 1, x_scale, y_scale, z_scale)
    apical_ring_pc1, apical_ring_pc2, apical_ring_pc3, apical_ring_midpoint, apcial_ring_pc1_loading, apcial_ring_pc2_loading, apical_ring_pc3_loading = pca_analysis(
        apical_ring_coordinates)
    apical_ring_plane = Plane(
        Point3D(apical_ring_midpoint),
        Point3D(apical_ring_midpoint + apical_ring_pc1),
        Point3D(apical_ring_midpoint + apical_ring_pc2))
    apical_ring_plane_normal_vector_point = Point3D(apical_ring_midpoint +
        np.array(apical_ring_plane.normal_vector))
    apical_ring_plane_normal_vector_length = N(Point3D(apical_ring_midpoint).distance(apical_ring_plane_normal_vector_point))
    apical_belt_bottom_point = Point3D(apical_ring_midpoint +
        ((np.array(apical_ring_plane.normal_vector)/apical_ring_plane_normal_vector_length) * belt_depth))

    if apical_belt_bottom_point.distance(cell_centroid_point) > Point3D(apical_ring_midpoint).distance(Point3D(cell_centroid_point)):
        apical_belt_bottom_point = (apical_ring_midpoint +
            (np.array(apical_ring_plane.normal_vector)/apical_ring_plane_normal_vector_length) * (belt_depth * -1))
    else:
        True

    apical_belt_bottom_plane = apical_ring_plane.parallel_plane(
        apical_belt_bottom_point)

    memb_cell_border = border(labelmatrix, 0, cell_label).astype(np.uint16)
    membrane_coordinates = coordinate_creator(memb_cell_border, 1, x_scale, y_scale, z_scale)

    x = Symbol('x', real = True)
    y = Symbol('y', real = True)
    z = Symbol('z', real = True)
    top_coordinate_evaluator = lambdify((x, y, z), apical_ring_plane.equation(), "numpy")
    bottom_coordinate_evaluator = lambdify((x, y, z), apical_belt_bottom_plane.equation(), "numpy")
    top_results = []
    bottom_results = []
    for i in range(membrane_coordinates.shape[0]):
        top_results.append(top_coordinate_evaluator(
            membrane_coordinates[i][0], membrane_coordinates[i][1], membrane_coordinates[i][2]))
        bottom_results.append(bottom_coordinate_evaluator(
            membrane_coordinates[i][0], membrane_coordinates[i][1], membrane_coordinates[i][2]))

    top_results_signs = np.sign(top_results)
    bottom_results_signs = np.sign(bottom_results)

    both_signs = top_results_signs + bottom_results_signs

    belt_coordinates = membrane_coordinates[both_signs == 0]
    belt_coordinates_voxels = belt_coordinates / np.array([x_scale, y_scale, z_scale])

    blank_apical_belt_labelmatrix = np.zeros(labelmatrix.shape)
    for j in range(belt_coordinates_voxels.shape[0]):
      blank_apical_belt_labelmatrix[
          int(belt_coordinates_voxels[j,0]),
          int(belt_coordinates_voxels[j,1]),
          int(belt_coordinates_voxels[j,2])] = 1

    apical_belt_labelmatrix = binary_closing(blank_apical_belt_labelmatrix)

    if return_image == True:
        imsave('{}.tif'.format(apical_belt_name), np.swapaxes(
        apical_belt_labelmatrix, 0, 2))
        return apical_belt_labelmatrix
    else:
        return apical_belt_labelmatrix

def sibling_volume_evaluator(equation, cell_coords, spindle_coordinates):
    x = Symbol('x', real = True)
    y = Symbol('y', real = True)
    z = Symbol('z', real = True)
    coordinate_evaluator = lambdify((x, y, z), equation, "numpy")
    results = []
    for i in range(cell_coords.shape[0]):
        results.append(coordinate_evaluator(
            cell_coords[i][0], cell_coords[i][1], cell_coords[i][2]))
    negatives = [j for j in results if j < 0]
    positives = [j for j in results if j > 0]
    if coordinate_evaluator(
        spindle_coordinates[0][0],
        spindle_coordinates[0][1],
        spindle_coordinates[0][2]) < 0:
        anterior_daughter_size = negatives
        posterior_daughter_size = positives
    else:
        anterior_daughter_size = positives
        posterior_daughter_size = negatives
    volume_ratio = len(
        anterior_daughter_size)/(
        len(anterior_daughter_size) + len(posterior_daughter_size))
    return volume_ratio

def centroid_finder(labelmatrix, x_scale, z_scale, cell_label):
    """Rescales the labelmatrix so that voxels are isotropic, then finds the
    centroid of the cell with the given label and returns the position of the
    centroid in um coordinates"""
    scaling_factor = x_scale/z_scale
    resized_labelmatrix = zoom(
        labelmatrix,
        zoom = [scaling_factor, scaling_factor, 1],
        mode = 'nearest')
    cell_only = np.array(resized_labelmatrix == cell_label).astype(np.uint16)
    centroid = center_of_mass(cell_only, index = cell_label)
    um_centroid = np.array(centroid) * [z_scale, z_scale, z_scale]
    return um_centroid

def angle_return(entity1, entity2):
    """Finds and returns the angle in degrees between two sympy 3D entities"""
    angle = np.rad2deg(float(N(entity1.angle_between(entity2))))
    return angle

def pc_apical_relationship(pc1, pc2, pc3, apical_plane, pc1_loading, pc2_loading, pc3_loading):
    axes_list = ['pc1', 'pc2', 'pc3']
    pc_list = [pc1, pc2, pc3]
    pc_angles = []
    for x in pc_list:
        pc_angles.append(abs(angle_return(apical_plane, x)))
    ortho_pc_name = axes_list[pc_angles.index(max(pc_angles))]
    loadings = [pc1_loading, pc2_loading, pc3_loading]
    loadings[axes_list.index(ortho_pc_name)] = 0
    largest_apical_plane_pc_name = axes_list[loadings.index(max(loadings))]
    ortho_pc = pc_list[axes_list.index(ortho_pc_name)]
    largest_apical_plane_pc = pc_list[axes_list.index(largest_apical_plane_pc_name)]
    smallest_apical_plane_pc_name = axes_list[np.where((np.array(axes_list) != ortho_pc_name) & (np.array(axes_list) != largest_apical_plane_pc_name))[0][0]]
    smallest_apical_plane_pc = pc_list[axes_list.index(smallest_apical_plane_pc_name)]

    return ortho_pc, largest_apical_plane_pc, smallest_apical_plane_pc
