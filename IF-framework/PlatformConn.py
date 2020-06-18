# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:31:47 2020

@author: ivyer
"""
"""
This module contains Connection class, which include connections 
between source-population and target-population

This version is used for Pytorch and have been tested on Dell
"""

import numpy as np
from IFpoint import IF_PointNeuron
import utility as util

class ConnCluster(object):
    '''
    Define a conn-cluster, conns within the same cluster share the same (homogeneous) pre- and post-population 
    '''
    def __init__(self,pre_p,pre_type,post_p,post_type,weight,timescale,tau_d):
        self.pre_p   = pre_p
        self.post_p  = post_p
        self.pre_type  = pre_type
        self.post_type = post_type
        self.weight    = weight
        self.timescale = timescale
        self.tau_d     = tau_d
        self.platform  = None
    @property
    def homo_clust(self):
        return (tuple([self.pre_p, self.post_p]),tuple([self.post_type,self.pre_type]),\
                tuple([self.timescale]),tuple([self.tau_d]),tuple([self.weight]))#,tuple([self.timescale,self.EItype]))
    
class Connection(object):
    """
    Parameters:
    pre-neuron
    post-neuron
    connection weight
    may have synaptic delay
    
    Output pair
    """
    def __init__(self,pre_p,post_p,weights,conn_type):
        self.pre_p   = pre_p
        self.post_p  = post_p
        self.pre_type   = self.pre_p.eitype
        self.post_type  = self.post_p.eitype
        self.weights   = weights
        self.spacescale = conn_type['space_scale']
        self.timescale  = conn_type['time_scale']
        self.tau_d      = conn_type['tau_d']
#        self.EItype = conn_type #pre,post-neuron
#        self.timescale = time_scale
#        ''' self voltage '''
#        self.v_min = v_min
#        self.v_max = v_max
#        self.dv    = dv
        """
        1) curr_firing_rate 
        2) simulation could be used to find original platform
        """
        # initialize None and Initialize when simulation
        self.firing_rate = 0.0
        self.platform = None
        # long range
        self.inmda = 0.0
    # initialize by hand! when start simulation
    def initialize(self):
        self.initialize_firing_rate()
        self.initialize_I_nmda()
        self.initialize_clust()
       
    def initialize_firing_rate(self):
        self.firing_rate = self.pre_p.total_firing_rate

    def initialize_clust(self):
        Connkey = ConnCluster(self.pre_p,self.pre_type,self.post_p,self.post_type,\
                              self.weights,self.timescale,self.tau_d)
#        print('Connkey',Connkey.homo_clust)
        if Connkey.homo_clust not in self.platform.PlatformConnClust:
            self.platform.PlatformConnClust[Connkey.homo_clust] = Connkey
        self.conn_clust = Connkey
    def update(self):
        self.firing_rate = self.pre_p.total_firing_rate
    @property
    def curr_firing_rate(self):
        return self.firing_rate
    
class platform(object):
    def __init__(self,neurons_list,conns_list,Net_params,):
        self.neurons_list = neurons_list
        self.conns_list = conns_list
        # Conns-clust of Platform
        self.PlatformConnClust = {}
        # platform
        for c in self.conns_list:
            c.platform = self
        for nn in self.neurons_list:
            nn.platform = self
        # Network settings
        self.nn_pop = Net_params['num_populations']
        self.ne_per_pop = Net_params['numexc_ppopulations']
        self.ni_per_pop = Net_params['numinh_ppopulations']
        # initializing matrices       
        self.VEmat = np.random.rand(self.ne_per_pop,self.nn_pop)#np.zeros((self.ne_per_pop,self.nn_pop))
        self.VImat = np.random.rand(self.ni_per_pop,self.nn_pop)#np.zeros((self.ni_per_pop,self.nn_pop))
        self.pop_idx_E = np.zeros((self.ne_per_pop,self.nn_pop),dtype=int)
        self.pop_idx_I = np.zeros((self.ni_per_pop,self.nn_pop),dtype=int)
        for i in range(self.nn_pop):
            self.pop_idx_E[:,i] = i
            self.pop_idx_I[:,i] = i
        # All types of conns        
        self.recurrent_conns = {}
        self.cell_type = Net_params['cell_type']
        # fast with time constants less than 10 ms
        for pre_type in self.cell_type:
            for post_type in self.cell_type:
                self.recurrent_conns['fast',post_type,pre_type] = np.zeros((self.nn_pop,self.nn_pop))
        # slow with time constants longer than 100 ms
        for pre_type in self.cell_type:
            for post_type in self.cell_type:
                self.recurrent_conns['slow',post_type,pre_type] = np.zeros((self.nn_pop,self.nn_pop))
    def init_allconns(self,):
        for c in self.conns_list:
            c.platform = self
            c.initialize_clust()
        for key,val in self.PlatformConnClust.items():
            self.recurrent_conns[val.timescale,val.post_type,val.pre_type][val.pre_p.p_idx,val.post_p.p_idx] = val.weight/val.tau_d      
    def init_allneurons(self):
        for nn in self.neurons_list:
            nn.platform = self
            nn.initialize()
    def update_Islow(self,isi):
        for nn in self.neurons_list:
            nn.update_accum_Islow(isi)  
            # assume there were no new firings
    def update_MFE(self):    
        E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = util.getMFE_ifdyn(self.VEmat,self.VImat,\
                                                                        self.recurrent_conns['fast','e','e']+self.recurrent_conns['slow','e','e'],\
                                                                        self.recurrent_conns['fast','e','i']-self.recurrent_conns['slow','e','i'],\
                                                                        self.recurrent_conns['fast','i','e']+self.recurrent_conns['slow','i','e'],\
                                                                        self.recurrent_conns['fast','i','i']-self.recurrent_conns['slow','i','i'],\
                                                                        self.pop_idx_E,self.pop_idx_I)
        
        # Calculate slow, (NMDA-type), synaptic inputs
        self.E_fired_num = np.zeros((self.ne_per_pop*self.nn_pop,1))
        self.E_fired_num[E_fired] = 1
        self.E_fired_num = np.reshape(self.E_fired_num,(self.nn_pop,self.ne_per_pop))  
#        self.E_fired_num = self.E_fired_num.T
        self.I_fired_num = np.zeros((self.ni_per_pop*self.nn_pop,1))
        self.I_fired_num[I_fired] = 1
        self.I_fired_num = np.reshape(self.I_fired_num,(self.nn_pop,self.ni_per_pop))
#        self.I_fired_num = self.I_fired_num.T
        # Voltage
        self.VEmat, self.VImat = VE_pos.copy(),VI_pos.copy()
#        E0 = np.where(np.squeeze(self.E_fired_num[0,:])>0)
#        E1 = np.where(np.squeeze(self.E_fired_num[1,:])>0)
#        print('Pf 161 Efire0:',self.VEmat[E0,0])
#        print('Pf 161 Efire1:',self.VEmat[E1,1])
        
        for nn in self.neurons_list:
            nn.update_FRpoint()
            nn.update_Vpoint()
        for nn in self.neurons_list:
            nn.update_Ispoint()
        for cc in self.conns_list:
            cc.update()       
        return (E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos)

    def update_vain(self):          
        # Calculate slow, (NMDA-type), synaptic inputs
        self.E_fired_num = np.zeros((self.ne_per_pop*self.nn_pop,1))
        self.E_fired_num = np.reshape(self.E_fired_num,(self.nn_pop,self.ne_per_pop))  
        self.I_fired_num = np.zeros((self.ni_per_pop*self.nn_pop,1))
        self.I_fired_num = np.reshape(self.I_fired_num,(self.nn_pop,self.ni_per_pop))
        for nn in self.neurons_list:
            nn.update_FRpoint()
            nn.update_Vpoint()
        for cc in self.conns_list:
            cc.update()          
    

