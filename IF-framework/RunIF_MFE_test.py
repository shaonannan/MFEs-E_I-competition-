# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:53:04 2020
Run cases
@author: ivyer
"""
import numpy as np
from IFpoint import IF_PointNeuron
import utility as util
from PlatformConn import ConnCluster
from PlatformConn import Connection
from PlatformConn import platform  
    
    
Net_params = {'num_populations':2,
              'numexc_ppopulations':10,
              'numinh_ppopulations':10,
              'cell_type':['e','i'],
        }
time_params = {'tauf_e': 5,
               'tauf_i':7,
               'taus_e':108,
               'taus_i':108,
               'taum_e':20,
               'taum_i':20,
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
'''
nenum_pop,ninum_pop,num_pop = Net_params['numexc_ppopulations'],\
Net_params['numinh_ppopulations'],Net_params['num_populations']
neurons = {}
idx_neuron = 0
for idx_pop in range(num_pop):
    neurons[idx_pop,'e'] =  IF_PointNeuron(Net_params,idx_pop,'e',time_params,vol_params)
    neurons[idx_pop,'i'] =  IF_PointNeuron(Net_params,idx_pop,'i',time_params,vol_params)
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
    pre,post = neurons[ipop,'e'],neurons[ipop,'e']
    conn = Connection(pre,post,0.31,conn_type)
    connection_list.append(conn)
    # fast E2I
    pre,post = neurons[ipop,'e'],neurons[ipop,'i']
    conn = Connection(pre,post,0.29,conn_type)
    connection_list.append(conn)
    # fast I2E
    pre,post = neurons[ipop,'i'],neurons[ipop,'e']
    conn = Connection(pre,post,0.11,conn_type)
    connection_list.append(conn)
    # fast I2I
    pre,post = neurons[ipop,'i'],neurons[ipop,'i']
    conn = Connection(pre,post,0.19,conn_type)
    connection_list.append(conn)
    
for ipop in range(1,2,1):
    conn_type={'space_scale':'short',
               'time_scale':'fast',
               'tau_d':1,
            }
    # fast E2E
    pre,post = neurons[ipop,'e'],neurons[ipop,'e']
    conn = Connection(pre,post,0.29,conn_type)
    connection_list.append(conn)
    # fast E2I
    pre,post = neurons[ipop,'e'],neurons[ipop,'i']
    conn = Connection(pre,post,0.32,conn_type)
    connection_list.append(conn)
    # fast I2E
    pre,post = neurons[ipop,'i'],neurons[ipop,'e']
    conn = Connection(pre,post,0.30,conn_type)
    connection_list.append(conn)
    # fast I2I
    pre,post = neurons[ipop,'i'],neurons[ipop,'i']
    conn = Connection(pre,post,0.49,conn_type)
    connection_list.append(conn)
    
conn_type={'space_scale':'long',
           'time_scale':'slow',
           'tau_d':108,
           }

pre,post = neurons[1,'e'],neurons[0,'e']
conn = Connection(pre,post,2,conn_type)
connection_list.append(conn)
pre,post = neurons[0,'e'],neurons[1,'e']
conn = Connection(pre,post,2,conn_type)
connection_list.append(conn)


pre,post = neurons[1,'e'],neurons[0,'i']
conn = Connection(pre,post,2,conn_type)
connection_list.append(conn)
pre,post = neurons[0,'e'],neurons[1,'i']
conn = Connection(pre,post,2,conn_type)
connection_list.append(conn)

Plat = platform(nneurons_list,connection_list,Net_params)
# initialize point neurons and connections
Plat.init_allneurons()
Plat.init_allconns()

Plat.VEmat[1,0] = 0.96;
Plat.VEmat[4,0] = 0.89;
Plat.VImat[3,0] = 0.90;
Plat.VImat[2,0] = 0.60;

Plat.VEmat[3,1] = 0.88;
Plat.VEmat[8,1] = 0.96;
Plat.VImat[6,1] = 0.90;
Plat.VImat[4,1] = 0.20;
for nn in Plat.neurons_list:
    nn.update_Vpoint()

# Initializing a sequence of time parameters
dtmax = time_params['dt']
TMAX = 10.0;tbinsize=1.0;dtbin_ij=0
t_sum=0; dt_cur=0; T_rem=TMAX; dt_rem=tbinsize;
# Initializing a sequence of voltage parameters
NmEY = np.ones((1,num_pop))*0.375*400
NmIY = np.ones((1,num_pop))*0.375*100
DIY = np.ones((1,num_pop))*0.1
DEY = np.ones((1,num_pop))*0.12

VT = 1.0
RE,REj = np.zeros((1,num_pop)),np.zeros((1,num_pop),dtype=int)
RI,RIj = np.zeros((1,num_pop)),np.zeros((1,num_pop),dtype=int)
#RE[0,0] = 0.06;RE[0,1] = 0.6;REj[0,0] = 1;REj[0,1] = 0
#RI[0,0] = 0.6;RI[0,1] = 0.12;RIj[0,0] = 1;RIj[0,1] = 5
for i in range(num_pop):
    RE[0,i]  = -np.log(np.random.rand())/NmEY[0,i]
    REj[0,i] = 0+min(nenum_pop-1,max(0,np.floor(nenum_pop*np.random.rand())))
    RI[0,i]  = -np.log(np.random.rand())/NmIY[0,i] 
    RIj[0,i] = 0+min(ninum_pop-1,max(0,np.floor(ninum_pop*np.random.rand())))

# start updating and evoluting
while t_sum<TMAX:
#    print('t:',t_sum,';Trem:',T_rem)   
    min_RI,idx_RI = min(RI[0,:]),np.argmin(RI[0,:])
    min_RE,idx_RE = min(RE[0,:]),np.argmin(RE[0,:])
    print('RE:',min_RE,';RI:',min_RI,)
    if min_RI<min_RE:
        min_RT = min_RI
        idx_RT = idx_RI
    elif min_RI>=min_RE:
        min_RT = min_RE
        idx_RT = idx_RE
    if (T_rem<=min_RT)and(T_rem<=dtmax):
        print('condition 1')
        dt_cur = T_rem
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dtmax)
        Plat.update_vain()
    elif (dtmax<=min_RT)and(dtmax<=T_rem):
        print('condition 2')
        dt_cur = dtmax
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dtmax)
        Plat.update_vain()
#        print(Plat.VEmat[:,0])
    # trigger an Inhibitory neuron
    elif ((min_RI<min_RE)and(min_RI<dtmax)and(min_RI<T_rem)):
        print('condition 3')
        dt_cur = min_RI
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur 
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dt_cur) 
        # refresh VImat[RIj[0,idx_RI],idx_RI]
        Plat.VImat[RIj[0,idx_RI],idx_RI] += DIY[0,idx_RI]
        if (Plat.VImat[RIj[0,idx_RI],idx_RI]>=VT):
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
        else:
            Plat.update_vain()
#        print(Plat.VImat[:,0])
        RI[0,idx_RI] = 0.2#-np.log(np.random.rand())/NmIY[0,idx_RI]
        RIj[0,idx_RI] = 0+min(ninum_pop-1,max(0,np.floor(ninum_pop*np.random.rand())))
    elif ((min_RE<min_RI)and(min_RE<dtmax)and(min_RE<T_rem)):
        print('condition 4')
        dt_cur = min_RE
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur 
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dt_cur) 
        Plat.VEmat[REj[0,idx_RE],idx_RE] += DEY[0,idx_RE]
        if (Plat.VEmat[REj[0,idx_RE],idx_RE]>=VT):
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
        else:
            Plat.update_vain()
        '''
        '''
#        print(Plat.VEmat[:,1])
        RE[0,idx_RE] = 0.6#-np.log(np.random.rand())/NmEY[0,idx_RE]
        REj[0,idx_RE] = 0+min(nenum_pop-1,max(0,np.floor(nenum_pop*np.random.rand())))
    elif ((min_RE==min_RI)and(min_RE<dtmax)and(min_RE<T_rem)):
        print('condition 5')
        dt_cur = min_RE
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur 
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dt_cur) 
        # refresh VImat[RIj[0,idx_RI],idx_RI]
        Plat.VEmat[REj[0,idx_RE],idx_RE] += DEY[0,idx_RE]
        Plat.VEmat[RIj[0,idx_RI],idx_RI] += DIY[0,idx_RI]
        if (Plat.VEmat[REj[0,idx_RE],idx_RE]>=VT)or(Plat.VImat[RIj[0,idx_RI],idx_RI]>=VT):
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
        else:
            Plat.update_vain()
        RE[0,idx_RE] = -np.log(np.random.rand())/NmEY[0,idx_RE]
        REj[0,idx_RE] = 0+min(nenum_pop-1,max(0,np.floor(nenum_pop*np.random.rand())))
        
        RI[0,idx_RI] = -np.log(np.random.rand())/NmIY[0,idx_RI]
        RIj[0,idx_RI] = 0+min(ninum_pop-1,max(0,np.floor(ninum_pop*np.random.rand())))