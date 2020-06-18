 # -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:56:10 2020

@author: ivyer
"""

"""
Module contains maybe the most important part in our Moments Algorithm
Internal populations and their structures and dynamics!
This module combine utilities function, and describe dynamics of recurrent(internal) populations

This version is used for Pytorch and have been test on Dell
Last modified 2018/04/06
"""
import numpy as np

class IF_PointNeuron(object):
    def __init__(self,neuron_num,population_idx,EItype,time_params,vol_params,):
        '''
        neuron properties:
            excitatory/inhibitory neurons: EItype
            neuron index : n_idx
            population index : p_idx
        '''
        self.eitype = EItype
        self.p_idx  = population_idx
        '''
        time parameters:
            tau_m = 20.0 ms
            dt    = 1e-1 ms
            tauf(e/i)
            taus(e/i)
        '''
        self.I_slow    = 0.0
        self.I_hslow   = 0.0
        self.I_fast    = 0.0
        # Time parameters
        if self.eitype=='e':
            self.taum = time_params['taum_e']
            self.tauf = time_params['tauf_e']
            self.taus = time_params['taus_e']
            self.n_num  = neuron_num['numexc_ppopulations']
        elif self.eitype=='i':
            self.taum = time_params['taum_i']
            self.tauf = time_params['tauf_i']
            self.taus = time_params['taus_i']
            self.n_num  = neuron_num['numinh_ppopulations']
        '''
        voltage parameters(for simulation):
            max_vol = vol_params['v_max']
            min_vol = vol_params['v_min']
            dv      = vol_params['dv']
        '''
        self.v_max = vol_params['v_max']
        self.v_min = vol_params['v_min']
        self.dv    = vol_params['dv']
        self.vol   = 0.0
        self.vol_point = None
        #  >>>>>>>>>>>>>>>>>>>>>>>>> Simulation Platform >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.platform = None
        self.spin = 0.0
        self.LMFE = 0.0
        
    def initialize(self):
        self.update_Vpoint()
        self.initialize_synaptic_dicts()
        self.initialize_accum()
        

    def initialize_synaptic_dicts(self):
        self.jump_Islow_dicts  = {}
        self.total_Islow_dicts = {}
        # >>> 1st mean-value (mu), 2nd std-value (sig), 3rd LR-input (Inmda)
        for c in self.source_conn_list:
            if c.timescale =='slow':
                try:
                    syn_current = self.total_Islow_dicts.setdefault(c.conn_clust.homo_clust,0)
                    syn_current = self.jump_Islow_dicts.setdefault(c.conn_clust.homo_clust,0)
                except:
                    c.initialize_clust()
                self.total_Islow_dicts[c.conn_clust.homo_clust] = 0.0       
                self.jump_Islow_dicts[c.conn_clust.homo_clust] = 0.0       
    def initialize_accum(self):
        self.accum_Islow_dicts = {}
        self.has_conn_key = {}
        for c in self.source_conn_list:
            if c.timescale =='slow':
                try:
                    accum_current = self.accum_Islow_dicts.setdefault(c.conn_clust.homo_clust,0)
                except:
                    c.initialize_clust()
                self.accum_Islow_dicts[c.conn_clust.homo_clust] = 0.0    
        for c in self.source_conn_list:
            if c.timescale == 'slow':
                if c.conn_clust.homo_clust not in self.has_conn_key.keys():
                    self.has_conn_key.setdefault(c.conn_clust.homo_clust,c.tau_d)
                    
    def update_Vpoint(self):
        if self.eitype =='e':
            self.vol = self.platform.VEmat[:,self.p_idx]
        elif self.eitype =='i':
            self.vol = self.platform.VImat[:,self.p_idx]
            
    def update_FRpoint(self):
        if self.eitype =='e':
            self.spin = self.platform.E_fired_num[self.p_idx,:]
        elif self.eitype =='i':
            self.spin = self.platform.I_fired_num[self.p_idx,:]
            
    def update_Ispoint(self):
        for c in self.source_conn_list:
            if c.timescale =='slow':   
                self.jump_Islow_dicts[c.conn_clust.homo_clust] = 0.0
        for c in self.source_conn_list:
            if c.timescale =='slow':  
                self.jump_Islow_dicts[c.conn_clust.homo_clust] += c.pre_p.total_firing_rate*c.weights#/c.tau_d
#                print('pre-conn:',c.pre_p.p_idx,'; weights:',c.weights,'; firings:',c.pre_p.total_firing_rate)
#                if c.pre_p.eitype == 'e':
#                    print('sum_firing:',np.sum(np.squeeze(self.platform.E_fired_num[c.pre_p.p_idx,:])))
#                else:
#                    print('sum_firing:',np.sum(np.squeeze(self.platform.I_fired_num[c.pre_p.p_idx,:])))
        for key,val in self.accum_Islow_dicts.items():  
            self.accum_Islow_dicts[key] += self.jump_Islow_dicts[key]/self.has_conn_key[key] 
            
##         re-calculate VEmat and VImat
#        rangetotal = 0.0 
#        for key,val in self.accum_Islow_dicts.items():  
#            rangetotal += val
#        # Extract Efire and Ifire
#        if self.eitype=='e':
#            thisfire = np.where(np.squeeze(self.platform.E_fired_num[self.p_idx,:])>0)
#            # re-cal VEmat
#            checkV0 = np.squeeze(self.platform.VEmat[thisfire,self.p_idx])
#            iderr = np.where(checkV0>0)
#            if np.size(iderr)>0:
#                print('IFpoint.py err 134')
#                print('value:',checkV0)
#                print('err:',checkV0[iderr])
#            self.platform.VEmat[thisfire,self.p_idx] = rangetotal
#        elif self.eitype=='i':
#            thisfire = np.where(np.squeeze(self.platform.I_fired_num[self.p_idx,:])>0)
#            # re-cal VEmat
#            checkV0 = np.squeeze(self.platform.VImat[thisfire,self.p_idx])
#            iderr = np.where(checkV0>0)
#            if np.size(iderr)>0:
#                print('IFpoint.py err 145')
#                print('value:',checkV0)
#                print('err:',checkV0[iderr])
#            self.platform.VImat[thisfire,self.p_idx] = rangetotal   
                    
    def update_synaptic_MFE(self):
        # >>> 1st mean-value (mu), 2nd std-value (sig), 3rd LR-input (Inmda)
        for c in self.source_conn_list:
            if c.timescale =='slow':
                self.total_Islow_dicts[c.conn_clust.homo_clust] = 0.0    
                self.midi_Islow_dicts[c.conn_clust.homo_clust]  = 0.0
        for c in self.source_conn_list:
            if c.timescale =='slow':
                self.total_Islow_dicts[c.conn_clust.homo_clust] += c.pre_p.total_firing_rate*c.weights 
        
        MFE_jump = 0.0        
        for key,val in self.total_Islow_dicts.items():  
            MFE_jump += val         
        self.LMFE = MFE_jump     
    def update_accum_Islow(self,ISI):
        taum = self.taum
        gl   = 1/taum
        etm  = np.exp(-gl*ISI)
        self.vol = self.vol*etm
        for key,val in self.accum_Islow_dicts.items():  
            taud = self.has_conn_key[key]            
            etd  = np.exp(-ISI/taud)
            # update v
            self.vol +=(self.accum_Islow_dicts[key])*(etd-etm)*taum*taud/(taud-taum)
            self.accum_Islow_dicts[key] = self.accum_Islow_dicts[key]*etd   
            
        if self.eitype == 'e':
            self.platform.VEmat[:,self.p_idx]=self.vol
        elif self.eitype == 'i':
            self.platform.VImat[:,self.p_idx]=self.vol
          
    @property
    def source_conn_list(self):
        return [c for c in self.platform.conns_list if c.post_p == self]
    
    @property
    def target_conn_list(self):
        return [c for c in self.platform.conns_list if c.pre_p == self]

    @property
    def curr_voltage(self):
        return self.vol       
    
    @property
    def total_firing_rate(self):
        return np.sum(np.squeeze(self.spin))

    @property
    def curr_syn_inputs(self):
        return self.syn_currents


       

        
