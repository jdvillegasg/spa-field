import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse


class uslf_weights():
    # Properties shared by a uslf_weights class

    # Dummy initialization of the properties of the class
    Nx = 10
    Ny = 10
    x_max = 100
    y_max = 100

    # Constructor
    def __init__(self, Nx, Ny, xmax, ymax):
        self.Nx = Nx
        self.Ny = Ny
        self.x_max = xmax
        self.y_max = ymax
    
    # Methods

    def scenario_grid(self, mat_grid_as_scenario=False, return_edges=False):
        ''' 
        Nx ---> number of center x coordinate dimensions, columns of matrix
        Ny ---> number of y coordinate dimensions, rows of matrix
        mat_grid_as_scenario ---> if the output scenario matrix lower left entry is the (0,0) or the (0, ymax)
        Coordinates are measured respect to (0,0)'''
        
        x_edges = np.linspace(0, self.x_max, self.Nx+1)
        y_edges = np.linspace(0, self.y_max, self.Ny+1)
        
        x_centers = (x_edges[:-1] + x_edges[1:])*0.5
        y_centers = (y_edges[:-1] + y_edges[1:])*0.5
        
        x_centers_grid, y_centers_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
        
        if mat_grid_as_scenario:
            y_centers_grid = np.flip(y_centers_grid, 0)
        
        if return_edges:
            return x_centers_grid, y_centers_grid, x_edges, y_edges
        else:
            return x_centers_grid, y_centers_grid

    def line_supercover(self, p0, p1, x_edges=0, y_edges=0):

        '''
        Thought to be used for 1 p0 and 1 p1. For multiple p0 and p1, call repeatedly this function. If critical cythonize 

        Modified from https://gist.github.com/amccaugh/f459e45650915351bb65070141a28e3f

        Indexing starts from 1 and goes to Nx in x-coordinate (columns)
        Indexing starts from 1 and goes to Ny in y-coordinate (rows)

        '''

        x0 = p0[0]
        x1 = p1[0]
        y0 = p0[1]
        y1 = p1[1]
        
        Nx = len(x_edges)
        Ny = len(y_edges)
        
        x0 = np.digitize(x0, x_edges, right=True) # x_pixel_idx
        y0 = np.digitize(y0, y_edges, right=True) # y_pixel_idx 
        x1 = np.digitize(x1, x_edges, right=True) # x_pixel_idx
        y1 = np.digitize(y1, y_edges, right=True) # y_pixel_idx 

        dx = abs(x1-x0)
        dy = abs(y1-y0)
        x = x0
        y = y0
        ii = 0
        n = dx + dy
        err = dx - dy
        x_inc = 1 
        y_inc = 1 

        max_length = (max(dx,dy)+1)*3

        rr = np.zeros(max_length, dtype=np.intp)
        cc = np.zeros(max_length, dtype=np.intp)

        if x1 > x0: x_inc = 1 
        else:       x_inc = -1
        if y1 > y0: y_inc = 1 
        else:       y_inc = -1

        dx = 2 * dx
        dy = 2 * dy

        while n > 0:
            rr[ii] = y
            cc[ii] = x
            ii = ii + 1
            if (err > 0):
                x += x_inc
                err -= dy
            elif (err < 0):
                y += y_inc
                err += dx
            else: # If err == 0 the algorithm is on a corner
                rr[ii] = y + y_inc
                cc[ii] = x
                rr[ii+1] = y
                cc[ii+1] = x + x_inc
                ii = ii + 2
                x += x_inc
                y += y_inc
                err = err + dx - dy
                n = n - 1
            n = n - 1 
        rr[ii] = y
        cc[ii] = x
            
        return np.asarray(rr[0:ii+1]), np.asarray(cc[0:ii+1])
    
    def fill_matrix_apm_1(self, xe, ye, start_p, stop_p, mat2vec_ind='cols'):
        '''
        Fill an Ny-by-Nx matrix with 1's only at the 'line_supercover' identified pixels for a given pair of 
        start_p and stop_p points. If multiple pair points are providen, flatten each matrix and save it as 
        a row in the output matrix to be returned. 

        If the start_p and stop_p parameters have more than 1 entry, it implies we will save each matrix as a row vector of the output matrix

        '''
        
        if len(start_p) != len(stop_p):
            return 0    
        
        # Multiple line segment
        if start_p.ndim > 1:
            output_matrix = np.zeros((len(start_p), self.Nx*self.Ny))
            
            # Loop over line segments
            for idx_line, line in enumerate(zip(start_p, stop_p)):
                r, c = self.line_supercover(line[0], line[1], x_edges=xe, y_edges=ye)
                
                # Linear index of matrix goes over columns and then changes to row:
                # Example with a 3 x 4 matrix
                # | 1 , 2 , 3 , 4 |
                # | 5 , 6 , 7 , 8 |
                # | 9, 10, 11, 12 |            
                
                for idx in range(len(r)):
                    output_matrix[idx_line, (r[idx]-1)*self.Nx + (c[idx]-1)] = 1
        
        # Only one segment
        else:
            output_matrix = np.zeros((self.Ny, self.Nx))
            
            r, c = self.line_supercover(start_p, stop_p, x_edges=xe, y_edges=ye)
            for idx in range(len(r)):
                output_matrix[r[idx]-1, c[idx]-1] = 1

        return output_matrix     

    def fill_matrix_apm_2(self, xcg, ycg, alpha, start_p, stop_p, verbose=False, save_ell_plot=False):
        '''
        
        alpha---> Angle [in degrees] made by the the most northern (or southern, is equivalent) point in the ellipse,
                 the most eastern (or west, is equivalent) point, and the center.  

        Fill an Ny-by-Nx matrix with 1's at the inside of the ellipse whose foci is  given by a pair of 
        start_p and stop_p points. If multiple pair points are providen, flatten each matrix and save it as 
        a row in the output matrix to be returned

        '''
        
        # If the start_p and stop_p parameters have more than 1 entry, it implies we will save each matrix as a row vector of the output matrix
        if len(start_p) != len(stop_p):
            return 0    
        
        g_ellipse = [] # This is replaced by the ellipse definition if start_p.ndim < 1
        
        # Multiple line segment
        if start_p.ndim > 1:
            output_matrix = np.zeros((len(start_p), self.Nx*self.Ny))
            
            # Loop over line segments
            for idx_line, line in enumerate(zip(start_p, stop_p)):
                
                # The foci
                foc_1 = line[0]
                foc_2 = line[1]
                
                # Center of the ellipse
                g_ell_center = (foc_1 + foc_2)/2
                
                # Angle w.r.t the real positive line
                angle = math.atan2(foc_2[1]-foc_1[1], foc_2[0]-foc_1[0])*(180.0/np.pi)

                ell_variable_c = np.linalg.norm(g_ell_center - foc_2)
                
                # Define the angle (always positive) between the first foci and the point at the height in the center
                #alpha = 60

                # Beta is the angle from the height at the center to the first foci
                beta = 180 - 90 - alpha

                # Law of sines: 
                #   height         width          ||center - foc_1||
                #  -------    =  --------   =  ----------------------
                # sin(alpha)     sin(90)             sin(beta)

                g_ell_width = ell_variable_c*(np.sin(90*(np.pi/180.))/np.sin(beta*(np.pi/180.)))
                g_ell_height = np.sqrt(g_ell_width**2 - ell_variable_c**2)
                
                if verbose:
                    print('Counter-clockwise angle between foci: ', angle)
                    print('Distance center to foci: ', ell_variable_c)
                    print('Distance center to foci: ', np.linalg.norm(g_ell_center - foc_1))
                    print('Width should be always greater than c. Is it true in this case: ', g_ell_width > ell_variable_c)
                    print('Width of ellipse: ', g_ell_width)
                    print('Height of ellipse: ', g_ell_height)
                    
                if save_ell_plot:
                    g_ellipse.append(Ellipse(g_ell_center, 2*g_ell_width, 2*g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2))
                else:
                    g_ellipse = Ellipse(g_ell_center, 2*g_ell_width, 2*g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2)
                
                cos_angle = np.cos(np.radians(180.-angle))
                sin_angle = np.sin(np.radians(180.-angle))
            
                xc = xcg - g_ell_center[0]
                yc = ycg - g_ell_center[1]
                xct = xc * cos_angle - yc * sin_angle
                yct = xc * sin_angle + yc * cos_angle 
                
                # Border of ellipse equation
                rad_cc = (xct**2/(g_ell_width)**2) + (yct**2/(g_ell_height)**2)
                
                idx_inside_ellipse = np.where(rad_cc <= 1.)

                for cnt, val in enumerate(idx_inside_ellipse[0]):
                    output_matrix[idx_line, val*self.Nx + idx_inside_ellipse[1][cnt]] = 1
                            
                # Linear index of matrix goes over columns and then changes to row:
                # Example with a 3 x 4 matrix
                # | 1 , 2 , 3 , 4 |
                # | 5 , 6 , 7 , 8 |
                # | 9, 10, 11, 12 |  
                    
        # Only one segment
        else:
            # The foci
            foc_1 = start_p
            foc_2 = stop_p
        
            # Center of the ellipse
            g_ell_center = (foc_1 + foc_2)/2
                
            # Angle w.r.t the real positive line
            angle = math.atan2(foc_2[1]-foc_1[1], foc_2[0]-foc_1[0])*(180.0/np.pi)

            ell_variable_c = np.linalg.norm(g_ell_center - foc_2)
                
            # Define the angle (always positive) between the first foci and the point at the height in the center
            #alpha = 60

            # Beta is the angle from the height at the center to the first foci
            beta = 180 - 90 - alpha

            # Law of sines: 
            #   height         width          ||center - foc_1||
            #  -------    =  --------   =  ----------------------
            # sin(alpha)     sin(90)             sin(beta)

            g_ell_width = ell_variable_c*(np.sin(90*(np.pi/180.))/np.sin(beta*(np.pi/180.)))
            g_ell_height = np.sqrt(g_ell_width**2 - ell_variable_c**2)
                
            if verbose:
                print('Center coordinates: ', g_ell_center)
                print('Counter-clockwise angle between foci: ', angle)
                print('Distance center to foci: ', ell_variable_c)
                print('Distance center to foci: ', np.linalg.norm(g_ell_center - foc_1))
                print('Width should be always greater than c. Is it true in this case: ', g_ell_width > ell_variable_c)
                print('Width of ellipse: ', g_ell_width)
                print('Height of ellipse: ', g_ell_height)
                    
            g_ellipse = Ellipse(g_ell_center, 2*g_ell_width, 2*g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2)
            #ax.add_patch(g_ellipse)
            #ax.scatter(g_ell_center[0], g_ell_center[1])
            
            cos_angle = np.cos(np.radians(180.-angle))
            sin_angle = np.sin(np.radians(180.-angle))
            
            xc = xcg - g_ell_center[0]
            yc = ycg - g_ell_center[1]
            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle 
                
            # Border of ellipse equation
            rad_cc = (xct**2/(g_ell_width)**2) + (yct**2/(g_ell_height)**2)
            
            output_matrix = np.zeros(rad_cc.shape)
            idx_inside_ellipse = np.where(rad_cc <= 1.)
            
            for cnt, val in enumerate(idx_inside_ellipse[0]):
                output_matrix[val, idx_inside_ellipse[1][cnt]]= 1
    
            output_matrix = output_matrix.reshape(self.Ny, self.Nx)
            
        return output_matrix, g_ellipse

    def fill_matrix_apm_3(self, xcg, ycg, alpha, start_p, stop_p, verbose=False, save_ell_plot=False):
        '''
        
        Very similar code structure to 'fill_matrix_apm_3'. However, the interior of the ellipse is filled
        with values depending on the 'inverse area' of the concentric ellipse touching the actual pixel, if
        the angle alpha is preserved and as a consequence the foci are moved towards the center.
        
        '''

        # If the start_p and stop_p parameters have more than 1 entry, it implies we will save each matrix as a row vector of the output matrix
        if len(start_p) != len(stop_p):
            return 0    
        
        g_ellipse = [] # This is replaced by the ellipse definition if start_p.ndim < 1
        
        # Multiple line segment
        if start_p.ndim > 1:
            output_matrix = np.zeros((len(start_p), self.Nx*self.Ny))
            
            # Loop over line segments
            for idx_line, line in enumerate(zip(start_p, stop_p)):
                
                # The foci
                foc_1 = line[0]
                foc_2 = line[1]
                
                # Center of the ellipse
                g_ell_center = (foc_1 + foc_2)/2
                
                # Angle w.r.t the real positive line
                angle = math.atan2(foc_2[1]-foc_1[1], foc_2[0]-foc_1[0])*(180.0/np.pi)

                ell_variable_c = np.linalg.norm(g_ell_center - foc_2)

                # Beta is the angle from the height at the center to the first foci
                beta = 180 - 90 - alpha

                # Law of sines: 
                #   height         width          ||center - foc_1||
                #  -------    =  --------   =  ----------------------
                # sin(alpha)     sin(90)             sin(beta)

                g_ell_width = ell_variable_c*(np.sin(90*(np.pi/180.))/np.sin(beta*(np.pi/180.)))
                g_ell_height = np.sqrt(g_ell_width**2 - ell_variable_c**2)
                
                if verbose:
                    print('Counter-clockwise angle between foci: ', angle)
                    print('Distance center to foci: ', ell_variable_c)
                    print('Distance center to foci: ', np.linalg.norm(g_ell_center - foc_1))
                    print('Width should be always greater than c. Is it true in this case: ', g_ell_width > ell_variable_c)
                    print('Width of ellipse: ', g_ell_width)
                    print('Height of ellipse: ', g_ell_height)
                    
                if save_ell_plot:
                    g_ellipse.append(Ellipse(g_ell_center, 2*g_ell_width, 2*g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2))
                else:
                    g_ellipse = Ellipse(g_ell_center, 2*g_ell_width, 2*g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2)
                
                cos_angle = np.cos(np.radians(180.-angle))
                sin_angle = np.sin(np.radians(180.-angle))
            
                xc = xcg - g_ell_center[0]
                yc = ycg - g_ell_center[1]
                xct = xc * cos_angle - yc * sin_angle
                yct = xc * sin_angle + yc * cos_angle 
                
                # Border of ellipse equation
                rad_cc = (xct**2/(g_ell_width)**2) + (yct**2/(g_ell_height)**2)
                
                idx_inside_ellipse = np.where(rad_cc <= 1.)

                for cnt, val in enumerate(idx_inside_ellipse[0]):
                    output_matrix[idx_line, val*self.Nx + idx_inside_ellipse[1][cnt]] = 1
                            
                # Linear index of matrix goes over columns and then changes to row:
                # Example with a 3 x 4 matrix
                # | 1 , 2 , 3 , 4 |
                # | 5 , 6 , 7 , 8 |
                # | 9, 10, 11, 12 |  
                    
        # Only one segment
        else:
            # The foci
            foc_1 = start_p
            foc_2 = stop_p
        
            # Center of the ellipse
            g_ell_center = (foc_1 + foc_2)/2
                
            # Angle w.r.t the real positive line
            angle = math.atan2(foc_2[1]-foc_1[1], foc_2[0]-foc_1[0])*(180.0/np.pi)

            ell_variable_c = np.linalg.norm(g_ell_center - foc_2)

            # Beta is the angle from the height at the center to the first foci
            beta = 180 - 90 - alpha

            # Law of sines: 
            #   height         width          ||center - foc_1||
            #  -------    =  --------   =  ----------------------
            # sin(alpha)     sin(90)             sin(beta)

            g_ell_width = ell_variable_c*(np.sin(90*(np.pi/180.))/np.sin(beta*(np.pi/180.)))
            g_ell_height = np.sqrt(g_ell_width**2 - ell_variable_c**2)
                
            if verbose:
                print('Center coordinates: ', g_ell_center)
                print('Counter-clockwise angle between foci: ', angle)
                print('Distance center to foci: ', ell_variable_c)
                print('Distance center to foci: ', np.linalg.norm(g_ell_center - foc_1))
                print('Width should be always greater than c. Is it true in this case: ', g_ell_width > ell_variable_c)
                print('Width of ellipse: ', g_ell_width)
                print('Height of ellipse: ', g_ell_height)
                    
            g_ellipse = Ellipse(g_ell_center, 2*g_ell_width, 2*g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2)
            #ax.add_patch(g_ellipse)
            #ax.scatter(g_ell_center[0], g_ell_center[1])
            
            cos_angle = np.cos(np.radians(180.-angle))
            sin_angle = np.sin(np.radians(180.-angle))
            
            xc = xcg - g_ell_center[0]
            yc = ycg - g_ell_center[1]
            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle 
                
            # Border of ellipse equation
            rad_cc = (xct**2/(g_ell_width)**2) + (yct**2/(g_ell_height)**2)
            
            output_matrix = np.zeros(rad_cc.shape)
            idx_inside_ellipse = np.where(rad_cc <= 1.)
            
            # Give the inverse of the area value to ellipse touched by the pixel
            
            for cnt, val in enumerate(idx_inside_ellipse[0]):
                output_matrix[val, idx_inside_ellipse[1][cnt]]= 1/(np.pi*g_ell_height*rad_cc[val, idx_inside_ellipse[1][cnt]]*g_ell_width*rad_cc[val, idx_inside_ellipse[1][cnt]])
            
            output_matrix = output_matrix.reshape(self.Ny, self.Nx)
            
        return output_matrix, g_ellipse

    def plot_apm(self, start_p, stop_p, xe, ye, output_matrix, cmp=None, figsz=(8,5)):
        '''
        xe--> edges in x-axis
        ye--> edges in y-axis

        Plot the APM matrix for one pair of start_p and stop_p points

        '''

        fig, ax = plt.subplots(figsize=figsz)

        ax.scatter([start_p[0], stop_p[0]], [start_p[1], stop_p[1]])
        
        if cmp is None:
            ax.imshow(np.flip(output_matrix, 0), cmap ='Greys', extent =[0, self.x_max, 0, self.y_max])
        else:
            ax.imshow(np.flip(output_matrix, 0), cmap =cmp, extent =[0, self.x_max, 0, self.y_max])
        
        plt.xticks(xe)
        plt.yticks(ye)
        ax.grid(True)

        #plt.show()
        return ax

    def plot_line_supercover(self, xe, ye, figsize=None, **kwargs):
        '''
        xe--> edges in x-axis
        ye--> edges in y-axis

        Plot the pixels selected by the line_supercover algorithm

        '''

        start_p = np.random.uniform(low=0, high = min(self.x_max, self.y_max), size=(2,))
        stop_p = np.random.uniform(low=0, high = min(self.x_max, self.y_max), size=(2,))
        
        pix_activate = np.zeros((self.Ny, self.Nx))
        
        for key, val in kwargs.items():
            if key=='start_p':
                start_p = val
            if key=='stop_p':
                stop_p = val
            if key=='output_matrix':
                pix_activate = val

        r, c = self.line_supercover(start_p, stop_p, x_edges=xe, y_edges=ye)
        
        # If output_matrix was passed, there is no need of creating pix_activate
        if not np.any(pix_activate):
            for idx in range(len(r)):
                pix_activate[r[idx]-1, c[idx]-1] = 1

        if figsize is not None:
            _ = plt.figure(figsize=figsize)
            
        plt.scatter([start_p[0], stop_p[0]], [start_p[1], stop_p[1]])
        plt.imshow(np.flip(pix_activate, 0), cmap ='Greys', extent =[0, self.x_max, 0, self.y_max])
        plt.xticks(xe)
        plt.yticks(ye)
        plt.grid(True)
        plt.show()

    
