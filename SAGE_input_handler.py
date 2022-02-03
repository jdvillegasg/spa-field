import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cmath as cm
import math
import csv


class SAGE_input_handler():

    idx_empty_ts = []

    def __init__(self, path_mpc_params, path_mpc_powers, max_num_paths):
        self.path_mpc_params = path_mpc_params
        self.path_mpc_powers = path_mpc_powers
        self.max_num_paths = max_num_paths

        self.tau = []
        self.aoa = []
        self.aod = []
        self.eod = []
        self.eoa = []

        # Support for Vertical and Horizontal poralizations (or equivalently 2 polarizations)
        # We keep the notation 'H' and 'V' since it is more intuitive
        self.alpha_H = []
        self.alpha_V = []

        self.set_sage_param(first_time=True)
        
    def set_sage_param(self, first_time=False, ts=None, return_empty_idx=False, **kwargs):
        ''' 
            Set SAGE parameters for the first time or if once created, pass the time snapshot(ts),
            and the parameter name as the key value with the corresponding new value for the parameter
            at that ts

        '''

        if first_time:
            with open(self.path_mpc_params, "r") as file_mpc_param:
                for line in file_mpc_param:
                    current_line = line.split(",")
                    current_line = np.array([float(i) for i in current_line])

                    tmp_tau = current_line[:self.max_num_paths]
                    tmp_tau = tmp_tau[np.nonzero(tmp_tau)]

                    tmp_aoa = current_line[self.max_num_paths:2*self.max_num_paths]
                    tmp_aoa = tmp_aoa[np.nonzero(tmp_aoa)]

                    tmp_eoa = current_line[2*self.max_num_paths:3*self.max_num_paths]
                    tmp_eoa = tmp_eoa[np.nonzero(tmp_eoa)]

                    tmp_aod = current_line[3*self.max_num_paths:4*self.max_num_paths]
                    tmp_aod = tmp_aod[np.nonzero(tmp_aod)]
                
                    tmp_eod = current_line[4*self.max_num_paths:5*self.max_num_paths]
                    tmp_eod = tmp_eod[np.nonzero(tmp_eod)]

                    self.tau.append(tmp_tau)
                    self.aoa.append(tmp_aoa)
                    self.aod.append(tmp_aod)
                    self.eod.append(tmp_eod)
                    self.eoa.append(tmp_eoa)


            with open(self.path_mpc_powers, "r") as file_mpc_powers:
                for line in file_mpc_powers:
                    current_line = line.split(",")
                    current_line = np.array([float(i) for i in current_line])

                    tmp_alph_V = current_line[:self.max_num_paths] - 1j*current_line[2*self.max_num_paths:3*self.max_num_paths] 
                    tmp_alph_H = current_line[self.max_num_paths:2*self.max_num_paths] - 1j*current_line[3*self.max_num_paths:4*self.max_num_paths]

                    tmp_alph_V = tmp_alph_V[np.nonzero(tmp_alph_V)]                    
                    tmp_alph_H = tmp_alph_H[np.nonzero(tmp_alph_H)]

                    self.alpha_V.append(tmp_alph_V)
                    self.alpha_H.append(tmp_alph_H)

        else:

            for key, value in kwargs.items():
                if key =='aoa':
                    # 'ts' must not be None (USER MUST KNOW WHAT TO PUT SINCE THERE IS NO ERROR HANDLING)
                    # The value accompanying the key 'aoa' (or whatever) must be a 1D array
                    self.aoa[ts] = value
                if key =='aod':
                    self.aod[ts] = value
                if key =='eod':
                    self.eod[ts] = value
                if key =='eoa':
                    self.eoa[ts] = value
                if key == 'tau':
                    self.tau[ts] = value
                if key == 'alpha_V':
                    self.alpha_V[ts] = value
                if key == 'alpha_H':
                    self.alpha_H[ts] = value 

        '''
        
            Erase the empty ts, since they do not provide any info
            IF THE TIME SNAPS ARE SAMPLED AT A LOW RATE, IT MAY BE USEFULL TO KEEP TRACK 
            OF THE EMPTY SNAPS ----> IN THAT CASE, RETURN THE EMPTY TS INDEXES

        '''

        idx_empty_ts = np.where(np.array([i.size for i in self.alpha_V]) == 0)[0]

        self.idx_empty_ts = idx_empty_ts

        self.aod = [ele for idx, ele in enumerate(self.aod) if idx not in idx_empty_ts]
        self.aoa = [ele for idx, ele in enumerate(self.aoa) if idx not in idx_empty_ts]
        self.eoa = [ele for idx, ele in enumerate(self.eoa) if idx not in idx_empty_ts]
        self.eod = [ele for idx, ele in enumerate(self.eod) if idx not in idx_empty_ts]
        self.tau = [ele for idx, ele in enumerate(self.tau) if idx not in idx_empty_ts]
        self.alpha_H = [ele for idx, ele in enumerate(self.alpha_H) if idx not in idx_empty_ts]
        self.alpha_V = [ele for idx, ele in enumerate(self.alpha_V) if idx not in idx_empty_ts]


        if return_empty_idx:
            return idx_empty_ts

    def power_definition(self, power_combination='2-pol-real'):
        '''
            How to combine the power gains (alpha)
            power_combination -->  '2-pol-real'
                              -->  '2-power-comp'
                              -->  'H-pol'
                              -->  'V-pol'
        '''
        pg = []
        if power_combination == '2-pol-real':
            for idx, g in enumerate(zip(self.alpha_H, self.alpha_V)):
                
                H = g[0]
                V = g[1]

                pg.append(np.sqrt(H.real**2 + V.real**2))

        elif power_combination == '2-pol-comp':
            for idx, g in enumerate(zip(self.alpha_H, self.alpha_V)):
                
                H = g[0]
                V = g[1]

                pg.append(np.sqrt(H.__abs__()**2 + V.__abs__()**2))

        elif power_combination == 'H-pol':
            for idx, g in enumerate(zip(self.alpha_H, self.alpha_V)):
                
                H = g[0]
                V = g[1]

                pg.append(H.__abs__()**2)

        elif power_combination == 'V-pol':
            for idx, g in enumerate(zip(self.alpha_H, self.alpha_V)):
                
                H = g[0]
                V = g[1]

                pg.append(V.__abs__()**2) 

        return pg


    def angular_correction_UCA(self, pg, aod_ref, aoa_ref):
        '''
            Apply some angular offset to the angles estimated, since the UCA was moving.
            Also, set the absolute angular reference to the x-axis (EAST-WEST axis), assuming
            the LoS between the transmitter and the receiver is the angle of maximum power gain

            ang_LoS --> list of the same size that pg (angles in radians)

            aoa_corr --> list of the same size that pg (angles in radians)
                        computed from the MS and BS coordinates and the the x,y coordinates of the center of gravity
                        of the station with the UCA                     
        '''
        
        idx_max_pg = [np.argmax(g) for g in pg]
        
        # The UCA was at the TX, so we have to take the AoD's
        aod_max_pg = [self.aod[idx_t][val_idx_max_pg] for idx_t, val_idx_max_pg in enumerate(idx_max_pg)]

        # Absolute referencing with the x-axis (East-West)
        aod_corr = aod_ref - aod_max_pg

        # Circular correction real numbers refering to angles outside [-pi, pi]
        aod_corr[np.where(aod_corr < - np.pi)] = 2*np.pi + aod_corr[np.where(aod_corr < - np.pi)]
        aod_corr[np.where(aod_corr >  np.pi)] = -2*np.pi + aod_corr[np.where(aod_corr >  np.pi)] 

        for cnt, ag in enumerate(self.aod):
            self.aod[cnt] = self.aod[cnt] + aod_corr[cnt]
            self.aoa[cnt] = self.aoa[cnt] + aoa_ref

            # Circular correction real numbers refering to angles outside [-pi, pi]
            self.aod[cnt][np.where(ag < - np.pi)] = 2*np.pi + ag[np.where(ag < - np.pi)]
            self.aod[cnt][np.where(ag >  np.pi)] = -2*np.pi + ag[np.where(ag >  np.pi)]

            self.aoa[cnt][np.where(self.aoa[cnt] < - np.pi)] = 2*np.pi + self.aoa[cnt][np.where(self.aoa[cnt] < - np.pi)]
            self.aoa[cnt][np.where(self.aoa[cnt] >  np.pi)] = -2*np.pi + self.aoa[cnt][np.where(self.aoa[cnt] >  np.pi)]
    