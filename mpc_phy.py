import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from shapely.geometry import Polygon, Point, LineString
import util_fcns as uf

class mpc_phy():
    
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

    def __init__(self, path_buildings, omit_this_buildings):

        # Load the buildings matfile with the vertexes of the buildings
        matfile = sio.loadmat(path_buildings)
        tmp = matfile['buildings']

        buildings_MATX_erase_rows = self.omit_buildings(tmp, omit_this_buildings)

        tmp = np.delete(tmp, buildings_MATX_erase_rows, 0)

        st_pt = uf.convert_DDDMS_to_planar(4326, 31370, tmp[:, 0], tmp[:, 2])
        fn_pt = uf.convert_DDDMS_to_planar(4326, 31370, tmp[:, 1], tmp[:, 3])

        self.bs['MKT'] = uf.convert_DDDMS_to_planar(4326, 31370, self.bs['MKT'][1], self.bs['MKT'][0])
        self.bs['MXW'] = uf.convert_DDDMS_to_planar(4326, 31370, self.bs['MXW'][1], self.bs['MXW'][0])
        
        self.ms_center_traj_st_barbe = uf.convert_DDDMS_to_planar(4326, 31370, self.ms_center_traj_st_barbe[1], self.ms_center_traj_st_barbe[0])

        # This is to set the bottom left corner the (0,0) coordinates:
        # All mobile station and base station coordinates are inside the extremes of the buildings considered
        # except for the abs_y_axis_min coordinate which corresponds to the minimim y_axis coordinate encountered
        # in one of the Parking Lot routes

        self.offset_scenario = {}
        self.offset_scenario['x'] = min([st_pt['x'].min(), fn_pt['x'].min()])
        self.offset_scenario['y'] = min([st_pt['y'].min(), fn_pt['y'].min(), self.abs_y_axis_min])
        
        #
        self.ms_center_traj_st_barbe['x'] = self.ms_center_traj_st_barbe['x'] - self.offset_scenario['x']
        self.ms_center_traj_st_barbe['y'] = self.ms_center_traj_st_barbe['y'] - self.offset_scenario['y']

        #
        self.bs['MKT']['x'] = self.bs['MKT']['x'] - self.offset_scenario['x']
        self.bs['MKT']['y'] = self.bs['MKT']['y'] - self.offset_scenario['y'] 
        self.bs['MXW']['x'] = self.bs['MXW']['x'] - self.offset_scenario['x']
        self.bs['MXW']['y'] = self.bs['MXW']['y'] - self.offset_scenario['y'] 

        #
        st_pt['x'] = st_pt['x'] - self.offset_scenario['x']
        fn_pt['x'] = fn_pt['x'] - self.offset_scenario['x']
        st_pt['y'] = st_pt['y'] - self.offset_scenario['y']
        fn_pt['y'] = fn_pt['y'] - self.offset_scenario['y']

        # Dictionary with starting and ending points of each line of each building in the scenario
        self.buildings = np.vstack((st_pt['x'], fn_pt['x'], st_pt['y'], fn_pt['y'])).T

    def omit_buildings(self, buildings, list_building_number):
        '''
            The numbers for the buildings are predefined
        '''

        buildings_MATX_erase_rows = np.array([x for idx in list_building_number for x in range(self.buildingsbyrow[idx][0], self.buildingsbyrow[idx][1]+1)])
        #self.buildingsbyrow = [ele for idx, ele in enumerate(self.buildingsbyrow) if idx not in list_building_number]

        new_buidlingsbyrow = [[0, -1]]
        for idx, ele in enumerate(self.buildingsbyrow):
            if idx not in list_building_number:
                row_diff = ele[1] - ele[0] + 1
                new_buidlingsbyrow.append([new_buidlingsbyrow[-1][1]+1, new_buidlingsbyrow[-1][1] + row_diff])
                
        del new_buidlingsbyrow[0]
        self.buildingsbyrow = new_buidlingsbyrow

        return buildings_MATX_erase_rows

    def make_polygon_buildings(self):
        '''
            Make a list of Polygon objects, each one representing the contour of a building
        '''

        building_polygons = []
        for i in self.buildingsbyrow:
            vertexes_polygons = []
            for j in range(i[0], i[1]+1):
                vertexes_polygons.append((self.buildings[j, 0],self.buildings[j, 2]))
            vertexes_polygons.append((self.buildings[j, 1],self.buildings[j, 3]))

            building_polygons.append(Polygon(vertexes_polygons))

        return building_polygons

    def get_ms_coordinates(self, ms_file_path, gps_file_num, BS, nCycles, cycleRate):
        
        '''  
            Get MS coordinates from gps file

            BS ---> 'MKT' or 'MXW' 
        
        '''

        ms_info = {}
        ms = {}

        df = pd.read_csv(ms_file_path, sep=' :', names=['GPS', 'TAG', 'QUALFIX', 'SAT', 'UTC', 'LAT', 'LON', 'ALT'])
        df.drop(columns=['GPS', 'QUALFIX', 'SAT'])

        # Process ALT column
        try:
            ms_info['ALT'] = np.asarray([float(val[1][1:-1]) for key, val in df.ALT.str.split(':').to_dict().items()])
        except:
            print('WARN: ALT is NONE')

        # Process LON column
        tmp = [val[1][1:-3] for key, val in df.LON.str.split(':').to_dict().items()]
        ms_info['LON'] = np.asarray([float(i[2]) + float(i[3:]) / 60 for i in tmp])

        # Process LAT column
        tmp = [val[1][1:-3] for key, val in df.LAT.str.split(':').to_dict().items()]
        ms_info['LAT'] = np.asarray([float(i[0:2]) + float(i[2:]) / 60 for i in tmp])

        # Process UTC column
        try:
            ms_info['UTC'] = np.asarray([float(val[1][1:-1]) for key, val in df.UTC.str.split(':').to_dict().items()])
        except:
            print('WARN: UTC is NONE')

        # Process TAG column: save it just in case --->> not known its functionality
        try:
            ms_info['TAG'] = np.asarray([int(val[1]) for key, val in df.TAG.str.split(':').to_dict().items()])
        except:
            print('WARN: TAG is NONE')

        ms = uf.convert_DDDMS_to_planar(4326, 31370, ms_info['LON'], ms_info['LAT'])

        # Make the coordinates be in the reference system whose (0,0) is the bottom left most point
        #  among the buildings and coordinates
        ms['x'] = ms['x'] - self.offset_scenario['x']
        ms['y'] = ms['y'] - self.offset_scenario['y']


        # Prune the MS coordinates from randomness in the gps coordinates (due to errors in the gps device)
        ms_pruned = {}
        ms_pruned['x'] = np.delete(ms['x'], self.switcher_del_coords[BS][gps_file_num])
        ms_pruned['y'] = np.delete(ms['y'], self.switcher_del_coords[BS][gps_file_num])

        # Linear interpolation to increase the size of the coordinates to equal the number of received power
        # samples. The number of IRFs per second is used as a criteria for the interpolation
        ms = np.array([ms_pruned['x'].tolist(), ms_pruned['y'].tolist()]).T  # recycle used variables
        ms_rotated = np.concatenate((ms[1:, :], ms[0, :][np.newaxis]))

        adjacent_distances = np.sqrt(np.sum(np.square(ms - ms_rotated), axis=1))

        # Estimated speed of the MS
        vel_ms = (cycleRate/nCycles)*np.sum(adjacent_distances)
        
        # Number of linear interpolated coordinates between 2 consecutive IRFs
        n_irfs_between_coords = np.floor((cycleRate / vel_ms) * adjacent_distances)
        samples_to_add = int(nCycles - np.sum(n_irfs_between_coords))

        # Sort distances in decreasing order
        idx_distances = np.fliplr(np.argsort(adjacent_distances)[np.newaxis])[0]

        n_irfs_between_coords[idx_distances[0:samples_to_add]] = n_irfs_between_coords[idx_distances[0:samples_to_add]] + 1
        n_irfs_between_coords = n_irfs_between_coords - 1

        coordinates_interpolated = []

        # Possibly sub-optimal python coding (translated from MATLAB)
        for idx, ms_i in enumerate(ms):
            coordinates_interpolated.append(ms_i.tolist())

            route_pointing_vector = ms_rotated[idx, :] - ms_i
            unitary_route_pointing_vector = route_pointing_vector/np.linalg.norm(route_pointing_vector)

            step_intermediate_points = adjacent_distances[idx] / n_irfs_between_coords[idx]

            for k in np.arange(start=1, stop=n_irfs_between_coords[idx] + 1, step=1):
                coordinates_interpolated.append((ms_i + k*step_intermediate_points*unitary_route_pointing_vector).tolist())

        ms_interpolated = np.asarray(coordinates_interpolated)

        nCycles_Comp = ms_interpolated.shape[0]

        # Just  drop first coordinates if bigger size
        if nCycles_Comp > nCycles:
            ms_interpolated = np.delete(ms_interpolated, np.arange(start=0, stop=nCycles_Comp - nCycles, step=1), axis=0)

        return ms_interpolated

    def set_aod_reference(self, ms, BS):
        '''
            LoS direction for the MS coordinates entered by parameters

        '''
        LoS_direction = np.array([self.bs[BS]['x'], self.bs[BS]['y']]) - ms
        
        return np.arctan2(LoS_direction[:, 1], LoS_direction[:,0])  
    
    def set_aoa_reference(self, BS, route='st_barbe'):
        '''
            The Azimuthal AoA reference. The center trajectories for the other routes
            must be provided
        
        '''

        if route == 'st_barbe':
            center_traj_direction = [self.ms_center_traj_st_barbe[key] - val for key, val in self.bs[BS].items()] 
        elif route == 'm_curie':
            center_traj_direction = [self.ms_center_traj_m_curie[key] - val for key, val in self.bs[BS].items()]
        elif route == 'p_lot':
            center_traj_direction = [self.ms_center_traj_p_lot[key] - val for key, val in self.bs[BS].items()]

        return np.arctan2(center_traj_direction[1], center_traj_direction[0])

    def build_mpc_path_points(self, aoa, aod, ms, bs, poly_buildings):
        '''
            Build the points (vertexes) of the piecewise curve

            The parameter rref controls how much to move from ms coordinate in the direction given by the
            by the unitary direction vector (cos(theta), sen(theta))

            This code is made for piecewise-curves composed by exactly 3 Line segments.
        
        '''        
        rref = 100

        path_first_line_seg = []

        
        for idx_t, ang_d in enumerate(aod):
            end_pt_first_line_seg = ms[idx_t, :] + rref*np.column_stack((np.cos(ang_d), np.sin(ang_d)))
            
            path_first_line_seg.append([])
            for idx_pth, end_pt_first_seg in enumerate(end_pt_first_line_seg):
                
                first_line = LineString([(ms[idx_t, :]), (end_pt_first_seg)])
                path_first_line_seg.append(first_line)

                first_bounce = list(poly_buildings.intersection(first_line).coords)   

                print(first_line, first_bounce)

                break
        
            break

        # First intercept from aoa



    def plot_buildings(self, building_polygons):
        '''
            Plot the contour of the buildings
        '''
        fig, ax = plt.subplots()

        for i in building_polygons:
            plt.plot(*i.exterior.xy)

        #plt.show()

        return fig, ax

