from skimage.io import imread
import numpy as np
from sympy import Plane, Point3D, Line3D, N, Ray3D


from czifile import CziFile
import spindle_analysis_functions

def sec_last_spindle_analysis(
    labelmatrix_path, cell_label, outside_label, ring_dilations,
    x_scale, y_scale, z_scale, spindle_coords, cell_name, half_name,
    apical_image_return, date, ant_point_x, ant_point_y, post_point_x,
    post_point_y):

    """Given a given segmented cell in a labelmatrix tiff and the coordinates of
    its spindle poles, along with a few other pieces of information, this
    function measures many aspects of spindle orientation and displacement
    """

    #import labelmatrix
    labelmatrix = np.swapaxes(imread(labelmatrix_path), 0, 2)

    #find coordinates of cell in um and perform PCA
    cell_coordinates = spindle_analysis_functions.coordinate_creator(
        labelmatrix, cell_label, x_scale, y_scale, z_scale)
    cell_pc1, cell_pc2, cell_pc3, cell_midpoint, cell_pc1_loading, cell_pc2_loading, cell_pc3_loading = spindle_analysis_functions.pca_analysis(cell_coordinates)

    #find apical surface, get coordinates in um, and perform PCA
    apical_surface_labelmatrix = spindle_analysis_functions.apical_surface_extractor(
        labelmatrix, outside_label, cell_label,
        return_image = apical_image_return,
        apical_surface_name = '{0}_{1}_{2}_apical_surface'.format(labelmatrix_path[:6],
        cell_name, half_name))
    apical_surface_coordinates = spindle_analysis_functions.coordinate_creator(
        apical_surface_labelmatrix, 1, x_scale, y_scale, z_scale)
    apical_surface_pc1, apical_surface_pc2, apical_surface_pc3, apical_surface_midpoint, apical_surface_pc1_loading, apical_surface_pc2_loading, apical_surface_pc3_loading = spindle_analysis_functions.pca_analysis(
        apical_surface_coordinates)

    #find apical belt, get coordinates in um, and perform PCA
    cell_centroid = spindle_analysis_functions.centroid_finder(
        labelmatrix, x_scale, z_scale, cell_label)
    apical_belt_labelmatrix = spindle_analysis_functions.apical_belt_extractor_planes(
        labelmatrix, outside_label, cell_label, 3, x_scale, y_scale,
        z_scale, cell_centroid, return_image = apical_image_return,
        apical_belt_name = '{0}_{1}_{2}_apical_belt'.format(
        labelmatrix_path[:6],cell_name, half_name))
    apical_belt_coordinates = spindle_analysis_functions.coordinate_creator(
        apical_belt_labelmatrix, 1, x_scale, y_scale, z_scale)
    apical_belt_pc1, apical_belt_pc2, apical_belt_pc3, apical_belt_midpoint, apical_belt_pc1_loading, apcial_belt_pc2_loading, apcial_belt_pc3_loading = spindle_analysis_functions.pca_analysis(
        apical_belt_coordinates)

    #get animal/vegetal separating plane
    embryo_labelmatrix = np.array(labelmatrix != outside_label).astype(np.uint16)
    embryo_coordinates = spindle_analysis_functions.coordinate_creator(
        embryo_labelmatrix, 1, x_scale, y_scale, z_scale)
    embryo_pc1, embryo_pc2, embryo_pc3, embryo_midpoint, embryo_pc1_loading, embryo_pc2_loading, embryo_pc3_loading = spindle_analysis_functions.pca_analysis(embryo_coordinates)
    ani_veg_plane = Plane(
        Point3D(embryo_midpoint),
        Point3D(embryo_midpoint + embryo_pc1),
        Point3D(embryo_midpoint + embryo_pc2))


    #converting spindle voxel coordinates into um coordinates
    um_scaling_array = np.array([x_scale, y_scale, z_scale])
    spindle_coords = spindle_coords * um_scaling_array

    '''creating sympy 3D entities for desired reference lines, points, and planes'''
    #apical plane
    apical_belt_plane = Plane(
        Point3D(apical_belt_midpoint),
        Point3D(apical_belt_midpoint + apical_belt_pc1),
        Point3D(apical_belt_midpoint + apical_belt_pc2))

    #PCs of cell
    cell_pc1_line = Line3D(
        Point3D(cell_midpoint),
        direction_ratio = cell_pc1)
    cell_pc2_line = Line3D(
        Point3D(cell_midpoint),
        direction_ratio = cell_pc2)
    cell_pc3_line = Line3D(
        Point3D(cell_midpoint),
        direction_ratio = cell_pc3)

    #PC of apical belt
    apical_belt_pc1_line = Line3D(
        Point3D(apical_belt_midpoint),
        direction_ratio = apical_belt_pc1)
    apical_belt_pc2_line = Line3D(
        Point3D(apical_belt_midpoint),
        direction_ratio = apical_belt_pc2)
    apical_belt_pc3_line = Line3D(
        Point3D(apical_belt_midpoint),
        direction_ratio = apical_belt_pc3)

    #centroid points
    cell_centroid_point = Point3D(spindle_analysis_functions.centroid_finder(
        labelmatrix, x_scale, z_scale, cell_label))
    apical_belt_centroid = Point3D(spindle_analysis_functions.centroid_finder(
        apical_belt_labelmatrix, x_scale, z_scale, 1))

    #spindle references
    spindle_line = Line3D(
        Point3D(spindle_coords[0]),
        Point3D(spindle_coords[1]))
    spindle_apical_projection = apical_belt_plane.projection_line(spindle_line)
    spindle_plane = apical_belt_plane.perpendicular_plane(
        spindle_apical_projection.p1, spindle_apical_projection.p2)
    spindle_midpoint_point = Point3D(np.mean(spindle_coords, axis = 0))
    posterior_pole_point = Point3D(spindle_coords[1])
    anterior_pole_point = Point3D(spindle_coords[0])

    #find how cell PCs are oriented in regards to the apical surface
    ortho_pc, largest_apical_plane_pc, smallest_apical_plane_pc = spindle_analysis_functions.pc_apical_relationship(
        cell_pc1_line, cell_pc2_line, cell_pc3_line,
        apical_belt_plane, cell_pc1_loading, cell_pc2_loading, cell_pc3_loading)

    #needed for finding displacement towards anterior daughter
    cellCentroid_spindleLine_projection = spindle_line.projection(cell_centroid_point)

    #creating planes that will be used to artifically divide the cell to assess resulting sibling asymmetry
    spindle_direction_ratio = (np.mean(spindle_coords, axis = 0))-(spindle_coords[1])

    C2 = apical_belt_plane.projection_line(
        largest_apical_plane_pc)

    predictedOrientation_cellCentroidPosition_divisionPlane = Plane(
        p1 = cell_centroid_point, normal_vector = C2.direction_ratio)

    predictedOrientation_midpointPosition_divisionPlane = Plane(
        p1 = spindle_midpoint_point, normal_vector = C2.direction_ratio)

    predictedOrientation_apicalBeltCentroidPosition_divisionPlane = Plane(
        p1 = apical_belt_centroid, normal_vector = C2.direction_ratio)

    actualOrientation_midpointPosition_divisionPlane = Plane(
        spindle_midpoint_point,
        normal_vector = spindle_direction_ratio)

    actualOrientation_cellCentroidPosition_divisionPlane = Plane(
        cell_centroid_point,
        normal_vector = spindle_direction_ratio)

    actualOrientation_apicalBeltCentroidPosition_divisionPlane = Plane(
        apical_belt_centroid,
        normal_vector = spindle_direction_ratio)



    #entities needed for anterior posterior orientation analysis
    reference_plane = Plane(
        Point3D(0, 0, 0),
        normal_vector = (0, 0, 1))
    a_p_line = Line3D(
        (ant_point_x, ant_point_y, 0),
        (post_point_x, post_point_y, 0))
    a_p_reference_projection = reference_plane.projection_line(a_p_line)
    a_p_apicalBeltPlane_projection = apical_belt_plane.projection_line(
        a_p_line)
    midline_plane = reference_plane.perpendicular_plane(a_p_reference_projection.p1,
                                                              a_p_reference_projection.p2)
    ant_post_vector = midline_plane.intersection(ani_veg_plane)[0]

    #needed for measuring spindle vs C2 (lateral deviation only)
    spindle_line_apicalBeltPlane_projection = apical_belt_plane.projection_line(
        spindle_line)



    #needed for measuring apical basal spindle displacement
    apical_surface_pc3_line = Line3D(
        Point3D(apical_surface_midpoint),
        direction_ratio = apical_surface_pc3)

    apicalBeltCentroid_apicalSurfacePC3_projection = apical_surface_pc3_line.projection(
        apical_belt_centroid)
    cellCentroid_apicalSurfacePC3_projection = apical_surface_pc3_line.projection(
        cell_centroid_point)
    spindleMidpoint_apicalSurfacePC3_projection = apical_surface_pc3_line.projection(
        spindle_midpoint_point)

    C1 = Line3D(apical_belt_centroid,
        apical_belt_centroid + apical_belt_plane.normal_vector)
    C3 = apical_belt_plane.projection_line(
        smallest_apical_plane_pc)

    #needed for masuring angles with respect to AP axis
    C1_aniveg_projection = ani_veg_plane.projection_line(C1)
    C2_aniveg_projection = ani_veg_plane.projection_line(largest_apical_plane_pc)
    C3_aniveg_projection = ani_veg_plane.projection_line(smallest_apical_plane_pc)
    PC1_aniveg_projection = ani_veg_plane.projection_line(cell_pc1_line)
    PC2_aniveg_projection = ani_veg_plane.projection_line(cell_pc2_line)
    PC3_aniveg_projection = ani_veg_plane.projection_line(cell_pc3_line)
    spindle_aniveg_projection = ani_veg_plane.projection_line(spindle_line)



    '''finding angles between references lines and planes'''
    results_dict = {}

    #spindle vs cell PC angles
    results_dict['spindle_pc1_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, cell_pc1_line)
    results_dict['spindle_pc2_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, cell_pc2_line)
    results_dict['spindle_pc3_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, cell_pc3_line)

    #spindle vs C1-C3 angles
    results_dict['spindle_C1_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, C1)
    results_dict['spindle_C2_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, C2)
    results_dict['spindle_C3_angle'] = spindle_analysis_functions.angle_return(
        spindle_line, C3)

    #apical plane vs cell PC angles
    results_dict['cellPC1_apicalBeltPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_plane, cell_pc1_line)
    results_dict['cellPC2_apicalBeltPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_plane, cell_pc2_line)
    results_dict['cellPC3_apicalBeltPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_plane, cell_pc3_line)

    #apical plane vs C1-C3 angles
    results_dict['C1_apicalBeltPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_plane, C1)
    results_dict['C2_apicalBeltPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_plane, C2)
    results_dict['C3_apicalBeltPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_plane, C3)

    #AP vs cell PC angles
    results_dict['cellPC1_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, PC1_aniveg_projection)
    results_dict['cellPC2_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, PC2_aniveg_projection)
    results_dict['cellPC3_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, PC3_aniveg_projection)

    #AP vs C1-C3 angles
    results_dict['C1_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, C1_aniveg_projection)
    results_dict['C2_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, C2_aniveg_projection)
    results_dict['C3_antPost_angle'] = spindle_analysis_functions.angle_return(
        ant_post_vector, C3_aniveg_projection)

    #C2 vs apical belt PC1 angle
    results_dict['C2_apicalBeltPC1_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_pc1_line, C2)

    #spindle vs AP angle
    results_dict['spindle_antPost_angle'] = spindle_analysis_functions.angle_return(
        spindle_aniveg_projection, ant_post_vector)

    #spindle vs C2 (lateral deviation only)
    results_dict['spindleApicalBeltPlaneProjeciton_C2_angle'] = spindle_analysis_functions.angle_return(
        spindle_line_apicalBeltPlane_projection, C2)

    #spindle vs apical plane angle
    results_dict['spindle_apicalBeltPlane_angle'] = spindle_analysis_functions.angle_return(
        apical_belt_plane, spindle_line)


    '''get volume of cell'''
    czi_image = CziFile('{}.czi'.format(labelmatrix_path[:-16]))
    image_metadata = czi_image.metadata
    size_element = image_metadata[0][3][0][0]
    pixel_size = float(size_element[0].text)
    results_dict['cell_volume'] = (cell_coordinates.shape[0]*(
        pixel_size * pixel_size * 0.0000003) * (1000000 ** 3))

    '''finding distances between reference points, lines, and planes'''
    #raw distance from spindle midpoint to cell centroid
    results_dict['spindleMidpoint_cellCentroid_distance'] = N(
        spindle_midpoint_point.distance(cell_centroid_point))

    #raw distance from spindle midpoint to apical belt centroid
    results_dict['spindleMidpoint_apicalBeltCentroid_distance'] = N(
        spindle_midpoint_point.distance(apical_belt_centroid))

    #apical basal displacement of spindle midpoint from cell and apical belt centroids
    apicalBeltCentroid_spindleMidpoint_ofApicalSurfacePC3projection_distance = N(
        apicalBeltCentroid_apicalSurfacePC3_projection.distance(
        spindleMidpoint_apicalSurfacePC3_projection))
    apicalBeltCentroid_cellCentroid_ofApicalSurfacePC3projection_distance = N(
        apicalBeltCentroid_apicalSurfacePC3_projection.distance(
        cellCentroid_apicalSurfacePC3_projection))
    cellCentroid_spindleMidpoint_ofApicalSurfacePC3projection_distance = N(
        cellCentroid_apicalSurfacePC3_projection.distance(
        spindleMidpoint_apicalSurfacePC3_projection))
    if apicalBeltCentroid_cellCentroid_ofApicalSurfacePC3projection_distance < apicalBeltCentroid_spindleMidpoint_ofApicalSurfacePC3projection_distance:
        cellCentroid_spindleMidpoint_ofApicalSurfacePC3projection_distance = cellCentroid_spindleMidpoint_ofApicalSurfacePC3projection_distance * -1

    results_dict['spindleMidpoint_displacementFromApicalBeltCentroidAwayFromApicalSurface'] = apicalBeltCentroid_spindleMidpoint_ofApicalSurfacePC3projection_distance
    results_dict['spindleMidpoint_displacementFromCellCentroidAwayFromApicalSurface'] = cellCentroid_spindleMidpoint_ofApicalSurfacePC3projection_distance

    #displacement towards anterior daughter on spindle vector
    midpoint_anterior_pole_distance = N(
        spindle_midpoint_point.distance(anterior_pole_point))
    centroid_anterior_pole_distance = N(
        cellCentroid_spindleLine_projection.distance(anterior_pole_point))
    results_dict['spindleDisplacement_towardsAnteriorDaughter'] = (
        centroid_anterior_pole_distance - midpoint_anterior_pole_distance)



    '''predicting sibling cell volume ratios resulting from artifical division planes'''
    results_dict['predictedOrientation_cellCentroidPosition_volumeRatio'] = spindle_analysis_functions.sibling_volume_evaluator(
        predictedOrientation_cellCentroidPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)
    results_dict['predictedOrientation_midpointPosition_volumeRatio'] = spindle_analysis_functions.sibling_volume_evaluator(
        predictedOrientation_midpointPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)
    results_dict['actualOrientation_midpointPosition_volumeRatio'] = spindle_analysis_functions.sibling_volume_evaluator(
        actualOrientation_midpointPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)
    results_dict['actualOrientation_cellCentroidPosition_volumeRatio'] = spindle_analysis_functions.sibling_volume_evaluator(
        actualOrientation_cellCentroidPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)
    results_dict['predictedOrientation_apicalBeltCentroidPosition_volumeRatio'] = spindle_analysis_functions.sibling_volume_evaluator(
        predictedOrientation_apicalBeltCentroidPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)
    results_dict['actualOrientation_apicalBeltCentroidPosition_volumeRatio'] = spindle_analysis_functions.sibling_volume_evaluator(
        actualOrientation_apicalBeltCentroidPosition_divisionPlane.equation(), cell_coordinates, spindle_coords)

    measurements = [
        date,
        labelmatrix_path[(labelmatrix_path.find('/') + 1):(labelmatrix_path.find('/') + 7)],
        cell_name,
        half_name,
        results_dict['spindle_pc1_angle'],
        results_dict['spindle_pc2_angle'],
        results_dict['spindle_pc3_angle'],
        results_dict['spindle_C1_angle'],
        results_dict['spindle_C2_angle'],
        results_dict['spindle_C3_angle'],
        results_dict['cellPC1_apicalBeltPlane_angle'],
        results_dict['cellPC2_apicalBeltPlane_angle'],
        results_dict['cellPC3_apicalBeltPlane_angle'],
        results_dict['C1_apicalBeltPlane_angle'],
        results_dict['C2_apicalBeltPlane_angle'],
        results_dict['C3_apicalBeltPlane_angle'],
        results_dict['cellPC1_antPost_angle'],
        results_dict['cellPC2_antPost_angle'],
        results_dict['cellPC3_antPost_angle'],
        results_dict['C1_antPost_angle'],
        results_dict['C2_antPost_angle'],
        results_dict['C3_antPost_angle'],
        results_dict['C2_apicalBeltPC1_angle'],
        results_dict['spindle_antPost_angle'],
        results_dict['spindleApicalBeltPlaneProjeciton_C2_angle'],
        results_dict['spindle_apicalBeltPlane_angle'],
        results_dict['cell_volume'],
        results_dict['spindleMidpoint_cellCentroid_distance'],
        results_dict['spindleMidpoint_apicalBeltCentroid_distance'],
        results_dict['spindleMidpoint_displacementFromApicalBeltCentroidAwayFromApicalSurface'],
        results_dict['spindleMidpoint_displacementFromCellCentroidAwayFromApicalSurface'],
        results_dict['spindleDisplacement_towardsAnteriorDaughter'],
        results_dict['predictedOrientation_cellCentroidPosition_volumeRatio'],
        results_dict['predictedOrientation_midpointPosition_volumeRatio'],
        results_dict['actualOrientation_midpointPosition_volumeRatio'],
        results_dict['actualOrientation_cellCentroidPosition_volumeRatio'],
        results_dict['predictedOrientation_apicalBeltCentroidPosition_volumeRatio'],
        results_dict['actualOrientation_apicalBeltCentroidPosition_volumeRatio']]

    return measurements
