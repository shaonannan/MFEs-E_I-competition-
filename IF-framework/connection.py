# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:40:22 2020

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
        self.pre_pop   = pre_p
        self.post_pop  = post_p
        self.pre_type  = pre_type
        self.post_type = post_type
        self.weight    = weight
        self.timescale = timescale
        self.tau_d     = tau_d
        self.platform  = None
    @property
    def homo_clust(self):
        return (tuple([self.pre_pop, self.post_pop]),tuple([self.post_type,self.pre_type]),\
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
    def __init__(self,pre,post,weights,conn_type):
        self.pre_idx   = pre
        self.post_idx  = post
        self.pre_pop   = self.pre_idx.p_idx
        self.post_pop  = self.post_idx.p_idx
        self.pre_type   = self.pre_idx.eitype
        self.post_type  = self.post_idx.eitype
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
        self.firing_rate = self.pre_idx.curr_firing_rate
    # LONG RANGE 
    def initialize_I_nmda(self):
        self.inmda = self.pre_idx.curr_Inmda
    def initialize_clust(self):
        Connkey = ConnCluster(self.pre_pop,self.pre_type,self.post_pop,self.post_type,\
                              self.weights,self.timescale,self.tau_d)
#        print('Connkey',Connkey.homo_clust)
        if Connkey.homo_clust not in self.platform.PlatformConnClust:
            self.platform.PlatformConnClust[Connkey.homo_clust] = Connkey
        self.conn_clust = Connkey
    def update(self):
        self.firing_rate = self.pre_idx.curr_firing_rate
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
        self.VEmat = np.zeros((self.ne_per_pop,self.nn_pop))
        self.VImat = np.zeros((self.ni_per_pop,self.nn_pop))
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
#            for key,val in self.PlatformConnClust.items():
#                print(key,val)
        for key,val in self.PlatformConnClust.items():
            self.recurrent_conns[val.timescale,val.post_type,val.pre_type][val.pre_pop,val.post_pop] = val.weight           
    def init_allneurons(self):
        for nn in self.neurons_list:
            nn.platform = self
            nn.initialize()
    def update(self):
        # TEST
        self.VEmat[1,0] = 1.00
        self.VEmat[4,0] = 0.89
        self.VEmat[7,0] = 0.79
        self.VImat[3,0] = 0.90
        self.VImat[2,0] = 0.60
        
        self.VEmat[3,1] = 1.00
        self.VEmat[8,1] = 0.96
        self.VImat[6,1] = 0.90
        self.VImat[4,1] = 0.20
        LEE = np.zeros((self.ne_per_pop*self.nn_pop,self.ne_per_pop*self.nn_pop))
        LEE[:self.ne_per_pop,self.ne_per_pop:] = 0.2;LEE[self.ne_per_pop:,:self.ne_per_pop] = 0.2
        
        LIE = np.zeros((self.ni_per_pop*self.nn_pop,self.ne_per_pop*self.nn_pop))
        LIE[:self.ni_per_pop,self.ne_per_pop:] = 0.12;LIE[self.ni_per_pop:,:self.ne_per_pop] = 0.12
        
        E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = util.getMFE_ifdyn(self.VEmat,self.VImat,\
                                                                        self.recurrent_conns['fast','e','e'],self.recurrent_conns['fast','e','i'],\
                                                                        self.recurrent_conns['fast','i','e'],self.recurrent_conns['fast','i','i'],\
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
        for nn in self.neurons_list:
            nn.update_FRpoint()
        for cc in self.conns_list:
            cc.update()
        for nn in self.neurons_list:
#            nn.update_synaptic_MFE()
            nn.update_accum_Islow(0.1)
            normfac = np.exp(-0.1/108.0)/108.0
            if nn.eitype == 'e':
                print('neuron:',nn.n_idx,' ;MFE_Jump:',nn.LMFE)
                print('neuron:',nn.n_idx,' ;MFE_Jump:',nn.cLMFE/normfac)
        self.E_fired_num = np.reshape(self.E_fired_num,(self.ne_per_pop*self.nn_pop,1))  
        self.I_fired_num = np.reshape(self.I_fired_num,(self.ni_per_pop*self.nn_pop,1))        
        LEmfe = np.dot(LEE,self.E_fired_num)        
        print('LEmfe:',LEmfe)
        
        return (E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos)
        
    

Net_params = {'num_populations':2,
              'numexc_ppopulations':10,
              'numinh_ppopulations':10,
              'cell_type':['e','i'],
        }
time_params = {'tauf_e': 5,
               'tauf_i':7,
               'taus_e':108,
               'taus_i':108,
               'dt':0.1,
        }
vol_params = {'v_max': 1.0,
               'v_min':-1.0,
               'dv':1e-3,
        }

nneurons_list, connection_list = [],[]
'''
Generate Point Neurons
class IF_PointNeuron(object):
    def __init__(self,neuron_idx,population_idx,EItype,time_params,vol_params,):

nnum_pop,num_pop = 2,2
nneuron_array = np.reshape(np.arange(nnum_pop*num_pop),(num_pop,nnum_pop))
'''
nenum_pop,ninum_pop,num_pop = 10,10 ,2
nnum_pop = nenum_pop+ninum_pop
nneuron_array = np.reshape(np.arange((nenum_pop+ninum_pop)*num_pop),(num_pop,nenum_pop+ninum_pop))
neurons = {}
idx_neuron = 0
for idx_pop in range(num_pop):
    for per_pop in range(nenum_pop):
        neurons[idx_pop,per_pop,'e'] =  IF_PointNeuron(per_pop,idx_pop,'e',time_params,vol_params)
        idx_neuron +=1
    for per_pop in range(ninum_pop):
        neurons[idx_pop,per_pop,'i'] =  IF_PointNeuron(per_pop,idx_pop,'i',time_params,vol_params)
        idx_neuron +=1
nneurons_list = neurons.values()

'''
Generate Connections between Point Neurons
class Connection(object):
    def __init__(self,pre,post,pre_p,post_p,weights):#,conn_type,time_scale,v_min,v_max,dv):
'''    
# homogeneous conns within identical population
for ipop in range(1):
    conn_type={'space_scale':'short',
               'time_scale':'fast',
               'tau_d':1,
            }
    # fast E2E
    for ie_pre in range(nenum_pop):
        for ie_post in range(nenum_pop):
            pre,post = neurons[ipop,ie_pre,'e'],neurons[ipop,ie_post,'e']
            conn = Connection(pre,post,0.31,conn_type)
            connection_list.append(conn)
    for ie_pre in range(nenum_pop):
        for ii_post in range(ninum_pop):
            pre,post = neurons[ipop,ie_pre,'e'],neurons[ipop,ii_post,'i']
            conn = Connection(pre,post,0.29,conn_type)
            connection_list.append(conn)
    for ii_pre in range(ninum_pop):
        for ie_post in range(nenum_pop):
            pre,post = neurons[ipop,ii_pre,'i'],neurons[ipop,ie_post,'e']
            conn = Connection(pre,post,0.11,conn_type)
            connection_list.append(conn)
    for ii_pre in range(ninum_pop):
        for ii_post in range(ninum_pop):
            pre,post = neurons[ipop,ii_pre,'i'],neurons[ipop,ii_post,'i']
            conn = Connection(pre,post,0.19,conn_type)
            connection_list.append(conn)
            
for ipop in range(1,2,1):
    conn_type={'space_scale':'short',
               'time_scale':'fast',
               'tau_d':1,
            }
    # fast E2E
    for ie_pre in range(nenum_pop):
        for ie_post in range(nenum_pop):
            pre,post = neurons[ipop,ie_pre,'e'],neurons[ipop,ie_post,'e']
            conn = Connection(pre,post,0.29,conn_type)
            connection_list.append(conn)
    for ie_pre in range(nenum_pop):
        for ii_post in range(ninum_pop):
            pre,post = neurons[ipop,ie_pre,'e'],neurons[ipop,ii_post,'i']
            conn = Connection(pre,post,0.32,conn_type)
            connection_list.append(conn)
    for ii_pre in range(ninum_pop):
        for ie_post in range(nenum_pop):
            pre,post = neurons[ipop,ii_pre,'i'],neurons[ipop,ie_post,'e']
            conn = Connection(pre,post,0.30,conn_type)
            connection_list.append(conn)
    for ii_pre in range(ninum_pop):
        for ii_post in range(ninum_pop):
            pre,post = neurons[ipop,ii_pre,'i'],neurons[ipop,ii_post,'i']
            conn = Connection(pre,post,0.49,conn_type)
            connection_list.append(conn)   


conn_type={'space_scale':'long',
           'time_scale':'slow',
           'tau_d':108,
           }
for ie_pre in range(nenum_pop):
    for ie_post in range(nenum_pop):
        pre,post = neurons[1,ie_pre,'e'],neurons[0,ie_post,'e']
        conn = Connection(pre,post,0.2,conn_type)
        connection_list.append(conn)
        pre,post = neurons[0,ie_pre,'e'],neurons[1,ie_post,'e']
        conn = Connection(pre,post,0.2,conn_type)
        connection_list.append(conn)

for ie_pre in range(nenum_pop):
    for ii_post in range(ninum_pop):
        pre,post = neurons[1,ie_pre,'e'],neurons[0,ii_post,'i']
        conn = Connection(pre,post,0.12,conn_type)
        connection_list.append(conn)
        pre,post = neurons[0,ie_pre,'e'],neurons[1,ii_post,'i']
        conn = Connection(pre,post,0.12,conn_type)
        connection_list.append(conn)
## Network settings
#self.nn_pop = Net_params['num_populations']
#self.ne_per_pop = Net_params['numexc_ppopulations']
#self.ni_per_pop = Net_params['numinh_ppopulations']

Plat = platform(nneurons_list,connection_list,Net_params)
print(Plat.recurrent_conns)
Plat.init_allneurons()#init_allconns()
Plat.init_allconns()
print(Plat.recurrent_conns)

E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update()