import numpy as np
import pandas as pd
import pyproj as proj

def utility_get_gps_file_path(base_dir, gps_file_num):
    '''
        Slightly different function from the function with the same name in the class 
        LargeScaleProp
    
    '''

    if gps_file_num > 9:
        file_path = base_dir + 't00000' + str(gps_file_num) + '.gps'
    else:
        file_path = base_dir + 't000000' + str(gps_file_num) + '.gps'
    
    return file_path


def convert_DDDMS_to_planar(epsg_in, epsg_out, input_lon, input_lat):
    """
    A wrapper to pyproj methods. DD stands for Decimal degrees, while DMS stands for
    Degrees, Minutes, Seconds format
    
    """

    coord = {}

    # setup your projections, assuming you're using WGS84 geographic
    crs_wgs = proj.Proj(init='epsg:' + str(epsg_in))
    crs_bng = proj.Proj(init='epsg:' + str(epsg_out))  # use the Belgium epsg code

    # then cast your geographic coordinate pair to the projected system
    coord['x'], coord['y'] = proj.transform(crs_wgs, crs_bng, input_lon, input_lat)

    return coord