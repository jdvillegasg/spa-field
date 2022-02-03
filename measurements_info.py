import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from shapely.geometry import Polygon, Point, LineString
import util_fcns as uf


class measurements_info():

    # Define the Properties
     # From the 'buildings' file, 'buildingsbyrow'  specifies the rows of 'buildings' that correspond to an entire building
    buildingsbyrow = [[0, 35], [36, 57],[58, 183], [184, 213],
                      [214, 223], [224, 231], [232, 298], [299, 305],
                      [306, 319], [320, 331], [332, 339], [340, 361],
                      [362, 369], [370, 373], [374, 377], [378, 393],
                      [394, 439], [440, 463]]

    # Lowest y-axis value
    abs_y_axis_min = 150771.95911315177

    # After manual inspection, remove these gps MS coordinates, due to inconsistencies and randomness in the route 
    switcher_del_coords = {'MKT': {0: [],
                            1: np.concatenate((np.arange(start=4, stop=7), np.arange(start=167, stop=215))),
                            2: np.concatenate((np.arange(start=0, stop=23), np.arange(start=190, stop=213))),
                            3: np.concatenate((np.arange(start=0, stop=47), np.arange(start=181, stop=225))),
                            4: np.concatenate((np.arange(start=0, stop=14), np.arange(start=161, stop=210))),
                            5: np.concatenate((np.arange(start=0, stop=14), np.arange(start=192, stop=195))),
                            6: [],
                            7: [],
                            8: np.concatenate((np.arange(start=0, stop=14), np.arange(start=163, stop=182))),
                            9: [],
                            10:[]},
                'MXW': {0: [],
                        1: np.concatenate((np.arange(start=0, stop=58), np.arange(start=238, stop=283))),
                        2: np.concatenate((np.arange(start=0, stop=12), np.arange(start=204, stop=227))),
                        3: np.concatenate((np.arange(start=0, stop=13), np.arange(start=201, stop=242))),
                        4: np.concatenate((np.arange(start=0, stop=20), np.arange(start=212, stop=246))),
                        5: np.concatenate((np.arange(start=0, stop=27), np.arange(start=170, stop=196))),
                        6: np.concatenate((np.arange(start=0, stop=19), np.arange(start=150, stop=176))),
                        7: np.concatenate((np.arange(start=0, stop=14), np.arange(start=147, stop=173))),
                        8: np.concatenate((np.arange(start=0, stop=19), np.arange(start=142, stop=203))),
                        9: np.concatenate((np.arange(start=0, stop=39), np.arange(start=151, stop=181))),
                        10: np.concatenate((np.arange(start=0, stop=17), np.arange(start=132, stop=163))),
                        11: np.concatenate((np.arange(start=0, stop=18), np.arange(start=66, stop=71), np.arange(start=138, stop=171))),
                        12: np.concatenate((np.arange(start=0, stop=14), np.arange(start=132, stop=164)))}}

    # MS GPS FILE NUMBERING SYSTEM, obtained after inspection of each file and ploting the received power 
    # for each route
    ms_gps_file_num_sys = {"MKT": {"StBarbe": [9], "ParkingLot": [1, 2, 3, 4], "MarieCurie": [5, 6, 7, 8]},
                   "MXW": {"StBarbe": [1, 2], "ParkingLot": [9, 10], "MarieCurie": [6, 7, 8]}}
 
    nCycles_gps_file_num_sys = {"MKT": {"StBarbe": [6838], "ParkingLot": [5651, 6207, 5259, 5808], "MarieCurie": [5752, 5740, 5543, 5586]},
                   "MXW": {"StBarbe": [7314, 6686], "ParkingLot": [4768, 4833], "MarieCurie": [5146, 5082, 5943]}}

    ms_center_traj_st_barbe = [50.668272, 4.621210]
    #ms_center_traj_p_lot = [50.668272, 4.621210]
    #ms_center_traj_m_curie = [50.668272, 4.621210] 

    bs = {'MXW': [50.668627, 4.623663], 'MKT': [50.669280, 4.620146]}