from skimage.io import imread
import numpy as np
from sympy import Plane, Point3D, Line3D, N

import spindle_analysis_functions
from czifile import CziFile


def final_div_spindle_analysis(
        labelmatrix_path, cell_label, x_scale, y_scale, z_scale, spindle_coords,
        cell_name, half_name, apical_image_return, date,
        apical_labelmatrix_path, apical_label, ant_point_x, ant_point_y,
        post_point_x, post_point_y, dorsal_direction
        ):
    """Given a given segmented cell in a labelmatrix tiff and the coordinates of
    its spindle poles, along with a few other pieces of information, this
    function measures many aspects of spindle orientation and displacement.
    """

    #import labelmatrix
    labelmatrix = np.swapaxes(imread(labelmatrix_path), 0, 2)

    #find coordinates of cell in um and perform PCA
    cell_coordinates = spindle_analysis_functions.coordinate_creator(
        labelmatrix, cell_label, x_scale, y_scale, z_scale)
    cell_pc1, cell_pc2, cell_pc3, cell_midpoint, cell_pc1_loading, cell_pc2_loading, cell_pc3_loading = spindle_analysis_functions.pca_analysis(cell_coordinates)

    #find apical surface, get coordinates in um, and perform PCA
    apical_labelmatrix = np.swapaxes(imread(apical_labelmatrix_path), 0, 2).astype(np.uint16)
    apical_coordinates = spindle_analysis_functions.coordinate_creator(
        apical_labelmatrix, apical_label, x_scale, y_scale, z_scale)
    apical_pc1, apical_pc2, apical_pc3, apical_midpoint, apical_pc1_loading, apical_pc2_loading, apical_pc3_loading = spindle_analysis_functions.pca_analysis(
        apical_coordinates)

    '''converting spindle voxel coordinates into um coordinates'''
    um_scaling_array = np.array([x_scale, y_scale, z_scale])
    spindle_coords = spindle_coords * um_scaling_array

    centroid_point = Point3D(spindle_analysis_functions.centroid_finder(
        labelmatrix, x_scale, z_scale, cell_label))

    '''creating sympy 3D entities for desired reference lines, points, and planes'''
    apical_plane = Plane(
        Point3D(apical_midpoint),
        Point3D(apical_midpoint + apical_pc1),
        Point3D(apical_midpoint + apical_pc2))

    cell_pc1_line = Line3D(
        Point3D(cell_midpoint),
        direction_ratio = cell_pc1)
    cell_pc2_line = Line3D(
        Point3D(cell_midpoint),
        direction_ratio = cell_pc2)
    cell_pc3_line = Line3D(
        Point3D(cell_midpoint),
        direction_ratio = cell_pc3)

    #finding orientation of cell PCs with respect to the apical plane
    ortho_pc, largest_apical_plane_pc, smallest_apical_plane_pc = spindle_analysis_functions.pc_apical_relationship(
        cell_pc1_line, cell_pc2_line, cell_pc3_line, apical_plane,
        cell_pc1_loading, cell_pc2_loading, cell_pc3_loading)

    #spindle references
    spindle_line = Line3D(
        Point3D(spindle_coords[0]), Point3D(spindle_coords[1]))
    spindle_midpoint_point = Point3D(np.mean(spindle_coords, axis = 0))
    posterior_pole_point = Point3D(spindle_coords[1])
    anterior_pole_point = Point3D(spindle_coords[0])
    spindle_apical_projection = apical_plane.projection_line(spindle_line)

    #needed for measuring spindle displacement
    centroid_spindle_line_projection = spindle_line.projection(centroid_point)

    #define C1-C3 coordinate system
    C1_line = Line3D(Point3D(apical_midpoint), Point3D(apical_midpoint + apical_plane.normal_vector))
    C2_line = apical_plane.projection_line(largest_apical_plane_pc)
    C3_line = apical_plane.projection_line(smallest_apical_plane_pc)

    #needed for measuring apical basal displacement of the spindle
    tiff_dims = labelmatrix.shape
    dorsal_dict = {
        'top': (tiff_dims * np.array([.5, .5, 0]) * um_scaling_array),
        'bottom': (tiff_dims * np.array([.5, .5, 1]) * um_scaling_array),
        'front': (tiff_dims * np.array([.5, 0, .5]) * um_scaling_array),
        'back': (tiff_dims * np.array([.5, 1, .5]) * um_scaling_array),
    }
    dorsal_point = Point3D(dorsal_dict[dorsal_direction])

    centroid_C1_projection = C1_line.projection(centroid_point)
    spindleMidpoint_C1_projection = C1_line.projection(spindle_midpoint_point)
    dorsalPoint_C1_projection = C1_line.projection(dorsal_point)

    #needed for measuring angles with respect to embryos AP axis
    C1_apicalPlane_projection = apical_plane.projection_line(C1_line)
    C2_apicalPlane_projection = apical_plane.projection_line(C2_line)
    C3_apicalPlane_projection = apical_plane.projection_line(C3_line)
    PC1_apicalPlane_projection = apical_plane.projection_line(cell_pc1_line)
    PC2_apicalPlane_projection = apical_plane.projection_line(cell_pc2_line)
    PC3_apicalPlane_projection = apical_plane.projection_line(cell_pc3_line)
    reference_plane = Plane(
        Point3D(0, 0, 0), normal_vector = (0, 0, 1))
    a_p_line = Line3D(
        (ant_point_x, ant_point_y, 0),
        (post_point_x, post_point_y, 0))
    a_p_reference_projection = reference_plane.projection_line(a_p_line)
    midline_plane = reference_plane.perpendicular_plane(a_p_reference_projection.p1,
                                                        a_p_reference_projection.p2)
    ant_post_vector = apical_plane.intersection(midline_plane)[0]


    '''creating planes to artifically divide cell'''
    actualOrientation_actualPosition_divisionPlane = Plane(
        spindle_midpoint_point,
        normal_vector = spindle_midpoint_point - anterior_pole_point)
    actualOrientation_predictedPosition_divisionPlane = Plane(
        centroid_point,
        normal_vector = spindle_midpoint_point - anterior_pole_point)

    '''finding angles between references lines and planes'''
    results_dict = {}

    #spindle vs cell PC or C1-C3 angles
    results_dict['spindle_pc1_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, cell_pc1_line)
    results_dict['spindle_pc2_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, cell_pc2_line)
    results_dict['spindle_pc3_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, cell_pc3_line)
    results_dict['C1_spindle_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, C1_line)
    results_dict['C2_spindle_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, C2_line)
    results_dict['C3_spindle_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, C3_line)

    #apical plane vs cell PC or C1-C3 angles
    results_dict['cellPC1_apicalPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_plane, cell_pc1_line)
    results_dict['cellPC2_apicalPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_plane, cell_pc2_line)
    results_dict['cellPC3_apicalPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_plane, cell_pc3_line)
    results_dict['C1_apicalPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_plane, C1_line)
    results_dict['C2_apicalPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_plane, C2_line)
    results_dict['C3_apicalPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_plane, C3_line)
    results_dict['spindle_apicalPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_plane, spindle_line)

    #angles with respect to the AP embryo axis
    results_dict['cellPC1_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, PC1_apicalPlane_projection)
    results_dict['cellPC2_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, PC2_apicalPlane_projection)
    results_dict['cellPC3_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, PC3_apicalPlane_projection)
    results_dict['C1_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, C1_line)
    results_dict['C2_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, C2_apicalPlane_projection)
    results_dict['C3_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, C3_apicalPlane_projection)
    results_dict['spindle_antPost_angle'] = spindle_analysis_functions.angle_return(
        spindle_apical_projection, ant_post_vector)

    #spindle vs C2 angle (lateral deviation only)
    results_dict['spindleApicalProjection_C2_angle'] = spindle_analysis_functions.angle_return(
        spindle_apical_projection, C2_line)


    '''get volume of cell'''
    czi_image = CziFile('{}.czi'.format(labelmatrix_path[:-16]))
    image_metadata = czi_image.metadata
    size_element = image_metadata[0][3][0][0]
    pixel_size = float(size_element[0].text)
    results_dict['cell_volume'] = (cell_coordinates.shape[0]*(
        pixel_size * pixel_size * 0.0000003)) * 1000000

    '''finding distances between reference points, lines, and planes'''
    #raw distance between spindle midpoint and cell centroid
    results_dict['spindle_displacement_distance_raw'] = N(
        spindle_midpoint_point.distance(centroid_point))

    #apical basal displacement of spindle midpoint from cell centroid
    cellCentroid_dorsalPoint_distance = N(
        dorsalPoint_C1_projection.distance(centroid_C1_projection))
    spindleMidpoint_dorsalPoint_distance = N(
        dorsalPoint_C1_projection.distance(spindleMidpoint_C1_projection))
    results_dict['spindle_apical_basal_displacement_towards_dorsal'] = (
        cellCentroid_dorsalPoint_distance - spindleMidpoint_dorsalPoint_distance)

    #displacement towards anterior daughter on spindle vector
    midpoint_anterior_pole_distance = N(
        spindle_midpoint_point.distance(anterior_pole_point))
    centroid_anterior_pole_distance = N(
        centroid_spindle_line_projection.distance(anterior_pole_point))
    results_dict['displacement_towards_anterior_daughter'] = (
        centroid_anterior_pole_distance - midpoint_anterior_pole_distance)

    '''predicting sibling cell volume ratios'''
    results_dict['actualOrientation_actualPosition_ratio'] = spindle_analysis_functions.sibling_volume_evaluator(
        actualOrientation_actualPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)
    results_dict['actualOrientation_predictedPosition_ratio'] = spindle_analysis_functions.sibling_volume_evaluator(
        actualOrientation_predictedPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)




    measurements = [
        date,
        labelmatrix_path[(labelmatrix_path.find('/') + 1):(labelmatrix_path.find('/') + 7)],
        cell_name,
        half_name,
        results_dict['spindle_pc1_angle'],
        results_dict['spindle_pc2_angle'],
        results_dict['spindle_pc3_angle'],
        results_dict['C1_spindle_angle'],
        results_dict['C2_spindle_angle'],
        results_dict['C3_spindle_angle'],
        results_dict['cellPC1_apicalPlane_angle'],
        results_dict['cellPC2_apicalPlane_angle'],
        results_dict['cellPC3_apicalPlane_angle'],
        results_dict['C1_apicalPlane_angle'],
        results_dict['C2_apicalPlane_angle'],
        results_dict['C3_apicalPlane_angle'],
        results_dict['spindle_apicalPlane_angle'],
        results_dict['cellPC1_antPost_angle'],
        results_dict['cellPC2_antPost_angle'],
        results_dict['cellPC3_antPost_angle'],
        results_dict['C1_antPost_angle'],
        results_dict['C2_antPost_angle'],
        results_dict['C3_antPost_angle'],
        results_dict['spindle_antPost_angle'],
        results_dict['spindleApicalProjection_C2_angle'],
        results_dict['cell_volume'],
        results_dict['spindle_displacement_distance_raw'],
        results_dict['spindle_apical_basal_displacement_towards_dorsal'],
        results_dict['displacement_towards_anterior_daughter'],
        results_dict['actualOrientation_actualPosition_ratio'],
        results_dict['actualOrientation_predictedPosition_ratio']]

    return measurements
