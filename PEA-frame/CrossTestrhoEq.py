import numpy as np
import itertools
import time
import utilities as util
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

from internalpopulation import RecurrentPopulation
from externalpopulation import ExternalPopulation
from simulation import Simulation
from connection import Connection as Connection

import scipy.io as scio

import sys

"""
09/08/2018 version
edited by SYX

Considerable Large Long-range NMDA Type Connections
Have Been Tested by IF-model 
as well as Master Equation
"""
# >>>>>>>>>>> Intuitive Network Structure >>>>>>>>>>>>>>>>>>>>>>
Net_settings = {'hyp_num': 2,
                'xhyp': 2,
                'yhyp': 1,
                'xn': 1,
                'yn': 1,
                'nmax': 0,
                'Final_time': 10000.0,
                'dt': 0.10,
                'dv': 1e-3}
Net_settings['nmax'] = Net_settings['hyp_num'] * Net_settings['xn'] * Net_settings['yn']
# here we use orientation and phase
Fun_map_settings = {'ori_num': 1,
                    'phase_num': 1}

# >>>>>>>>>>>>>>>> Network Scale >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Cell_type_num = {'e': 400,
                 'i': 100}
taum = {'e': 20.0,
        'i': 20.0}
print('Here, Network Settings')
print('Number of Hyper-columns: ', Net_settings['hyp_num'])
print('Time resolution:', Net_settings['dt'], ' Final time:', Net_settings['Final_time'])
print('Voltage resolution:', Net_settings['dv'])

# >>>>>>>>>>>>>>>>> Simulation Hyper Parameters >>>>>>>>>>>>>>>>>>>>>>>>>>
t0 = 0.0
dt = Net_settings['dt']
tf = Net_settings['Final_time']
tfinal = tf
dv = Net_settings['dv']
# parameters may not be used
verbose = True
update_method = 'approx'
approx_order = 1
tol = 1e-14

# >>>>>>>>>>>>>>>>>>>>> External Poison Inputs >>>>>>>>>>>>>>>>>>>>>>>>>>>>
NPATCH = Net_settings['nmax']
mEY, mIY, fE, fI = np.zeros(NPATCH), np.zeros(NPATCH), np.zeros(NPATCH), np.zeros(NPATCH)
for i in range(NPATCH):
    mEY[i] = 0.375  # 76
    mIY[i] = 0.375  # 0.76#50*0.82#1.504
    fE[i], fI[i] = 0.1, 0.1#0.016, 0.015


# >>>>>>>>>>>>>>>>>>>> Generate Temporal Input Stream >>>>>>>>>>>>>>>>>>>>>>>>>>>
def input_signal_stream(t_properties, Net_settings, mEY, mIY):
    (dt, tfinal) = t_properties
    ntt = int(tfinal / dt)

    # >>>>>>> List of Populations >>>>>>>
    background_population_dict = {}
    internal_population_dict = {}
    CG_range = np.arange(Net_settings['nmax'])

    # >>>>>>> Structure External and Internal Populations >>>>>>>>>>>>
    id_order = 0
    print('Showing Input Streams in Order')
    for layer, celltype in itertools.product(CG_range, ['e', 'i']):
        if celltype == 'e':
            External_stimuli_dict = np.ones(ntt) * mEY[layer] * 1
        #            # >>>>> if we need change external input >>>>>>
        #            if layer==0:
        #                External_stimuli_dict[6000:6005] = 3.11*0.74
        else:
            External_stimuli_dict = np.ones(ntt) * mIY[layer] * 1

        background_population_dict[layer, celltype] = ExternalPopulation(External_stimuli_dict, dt, record=False)
        internal_population_dict[layer, celltype] = RecurrentPopulation(tau_m = taum[celltype],dt=dt, v_min=-1.0, v_max=1.0, dv=dv,
                                                                        update_method=update_method,
                                                                        approx_order=approx_order, tol=tol, hyp_idx=0,
                                                                        ei_pop=celltype,
                                                                        NumCell=Cell_type_num[celltype])
        print('order:', id_order, ' layer:', layer, ' type:', celltype)
        id_order += 1

    population_list = list(background_population_dict.values()) + list(internal_population_dict.values())
    return (background_population_dict, internal_population_dict, population_list)


''' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Done >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> '''
background_population_dict, internal_population_dict, population_list = input_signal_stream([dt, tfinal], Net_settings,
                                                                                            mEY, mIY)

NE_source = Cell_type_num['e']
NI_source = Cell_type_num['i']

''' >>>>>>>>>>>>>>> Generate Recurrent Connectivity Matrix >>>>>>>>>>>>>'''
DEE = np.zeros((NPATCH, NPATCH))
DEI = np.zeros((NPATCH, NPATCH))
DII = np.zeros((NPATCH, NPATCH))
DIE = np.zeros((NPATCH, NPATCH))

# >>>>>>>>>>>>>>>> i to i Self Connections >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
DEE[0, 0] = 0.01*400*1.0#0.369/1.0
DEE[1, 1] = 0.01*400*1.0#0.369/1.0
DIE[0, 0] = 0.01*400*1.0#0.294/1.0
DIE[1, 1] = 0.01*400*1.0#0.294/1.0
DEI[0, 0] = 0.01*100*3.60#0.01*100*2.50#0.294/1.0
DEI[1, 1] = 0.01*100*3.60#0.01*100*2.50#0.294/1.0
DII[0, 0] = 0.01*100*3.60#0.369/1.0
DII[1, 1] = 0.01*100*3.60#0.369/1.0

# >>>>>>>>>>>>>>> i to j Cross Fast Connections >>>>>>>>>>>>>>>>>>>>>>>>>
DEE[0, 1] = 0.01*400*0.0
DEE[1, 0] = 0.01*400*0.0
DIE[1, 0] = 0.01*400*0.0
DIE[0, 1] = 0.01*400*0.0
DEI[0, 1] = 0.01*100*0.00  # 150#125#25
DEI[1, 0] = 0.01*100*0.00  # 150#125#25
DII[0, 1] = 0.01*100*0.00  # 150#125#25
DII[1, 0] = 0.01*100*0.00  # 150#125#25

# >>>>>>>>>>>>>>> i to j Cross Long-range Slow Connections >>>>>>>>>>>>>>>
LEE = np.zeros((NPATCH, NPATCH))
LIE = np.zeros((NPATCH, NPATCH))
LEI = np.zeros((NPATCH, NPATCH))
LII = np.zeros((NPATCH, NPATCH))

LEE[0, 1] = 0.01*400*0.0  # 
LEE[1, 0] = 0.01*400*0.0   # 
LIE[0, 1] = 0.01*400*0.0   # 
LIE[1, 0] = 0.01*400*0.0   # 

LEI[0, 1] = 0.01*100*0.4  # 
LEI[1, 0] = 0.01*100*0.4 # 
LII[0, 1] = 0.01*100*0.4 # 
LII[1, 0] = 0.01*100*0.4 # 
# >>>>>>>>>>>>>>>>>>> Initiation Finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#''' >>>>>>>>>>>>>>>>>>> For Recording, Long Float Format Should Be Used >>>>>>> '''
#DEER = DEE * 1000000
#DEIR = DEI * 1000000
#DIER = DIE * 1000000
#DIIR = DII * 1000000
#
#mEYR = mEY * 1000000
#mIYR = mIY * 1000000
#
#LEER = LEE * 1000000
#LIER = LIE * 1000000
#ISOTIMEFORMAT = '%Y%m%d%H%M%S'
#fileparamname = str(time.strftime(ISOTIMEFORMAT)) + '_params.mat'
#scio.savemat(fileparamname,
#             {'DEE': DEER, 'DEI': DEIR, 'DIE': DIER, 'DII': DIIR, 'LEE': LEER, 'LIE': LIER, 'mEY': mEYR, 'mIY': mIYR,
#              'fE': fE, 'fI': fI})

# >>>>>>>>>>>>>>. Individual Neuron Connection-strength >>>>>>>>>>>>>>>>>>>>>>
DEE /= NE_source
DIE /= NE_source
DEI /= NI_source
DII /= NI_source

LEE /= NE_source
LIE /= NE_source


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> All Connections >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# connection_list = cortical_to_cortical_connection(background_population_dict,internal_population_dict,DEE,DIE,DEI,DII,lambdaE,lambdaI,fE,fI,0.0)
def cortical_to_cortical_connection(background_population_dict, internal_population_dict, DEE, DIE, DEI, DII, LEE, LIE,
                                    fE, fI, delay):
    connection_list = []
    # >>>>>>>>>>>>>>>>>>>>>>>>>>. Feedforward Connections >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for ifftarget in range(Net_settings['nmax']):
        ''' >>>>>>>>>>>>>> Excitatory Poison Inputs to Excite Excitatory Populations >>>>>>>>>>>>>>>> '''
        source_population = background_population_dict[ifftarget, 'e']
        target_population = internal_population_dict[ifftarget, 'e']
        curr_connection = Connection(source_population, target_population, nsyn=1.0, nsyn_post=Cell_type_num['e'],
                                     weights=fE[ifftarget], probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0,
                                     dv=dv)
        connection_list.append(curr_connection)

        ''' >>>>>>>>>>>>>>>Excitatory Poison Inputs to Excite Inhibitory Populations >>>>>>>>>>>>>>> '''
        source_population = background_population_dict[ifftarget, 'i']
        target_population = internal_population_dict[ifftarget, 'i']
        curr_connection = Connection(source_population, target_population, nsyn=1.0, nsyn_post=Cell_type_num['i'],
                                     weights=fI[ifftarget], probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0,
                                     dv=dv)
        connection_list.append(curr_connection)
    # >>>>>>>>>>>>>>>>. Short Range(Local) Connections >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
    for isource in range(Net_settings['nmax']):
        for itarget in range(isource, Net_settings['nmax']):
            # >>>>>>>>>>>>>>>>>>>>>> Self-connections >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            '''
            For Self-connections, Source and Target Populations Need not to be Inversed, Like What We Do in A_m^n
            '''
            if isource == itarget:  # self-connection
                # seeself
                source_population = internal_population_dict[isource, 'e']
                target_population = internal_population_dict[itarget, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['e'], weights=DEE[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)
                # seiself
                source_population = internal_population_dict[isource, 'i']
                target_population = internal_population_dict[itarget, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['e'], weights=-DEI[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                # sieself
                source_population = internal_population_dict[isource, 'e']
                target_population = internal_population_dict[itarget, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['i'], weights=DIE[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)
                # siiself
                source_population = internal_population_dict[isource, 'i']
                target_population = internal_population_dict[itarget, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['i'], weights=-DII[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)
            else:
                # >>>>>>>>>>>>>>>>>>>>>>>. Short-range Fast(Local) Connections between Different Populations >>>>>>>>>>>>>>>>>>
                '''
                Under This Circumstance, Source and Target Populations Should Take Other's Position
                '''
                # seeself
                source_population = internal_population_dict[isource, 'e']
                target_population = internal_population_dict[itarget, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['e'], weights=DEE[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'e']
                target_population = internal_population_dict[isource, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['e'], weights=DEE[isource, itarget],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)
                # seiself
                source_population = internal_population_dict[isource, 'i']
                target_population = internal_population_dict[itarget, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['e'], weights=-DEI[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'i']
                target_population = internal_population_dict[isource, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['e'], weights=-DEI[isource, itarget],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                # sieself
                source_population = internal_population_dict[isource, 'e']
                target_population = internal_population_dict[itarget, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['i'], weights=DIE[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'e']
                target_population = internal_population_dict[isource, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['i'], weights=DIE[isource, itarget],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)
                # siiself
                source_population = internal_population_dict[isource, 'i']
                target_population = internal_population_dict[itarget, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['i'], weights=-DII[itarget, isource],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'i']
                target_population = internal_population_dict[isource, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['i'], weights=-DII[isource, itarget],
                                             probs=1.0, conn_type='ShortRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                # seeself
                source_population = internal_population_dict[isource, 'e']
                target_population = internal_population_dict[itarget, 'e']

                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['e'], weights=LEE[itarget, isource],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'e']
                target_population = internal_population_dict[isource, 'e']

                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['e'], weights=LEE[isource, itarget],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                # sieself
                source_population = internal_population_dict[isource, 'e']
                target_population = internal_population_dict[itarget, 'i']

                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['i'], weights=LIE[itarget, isource],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'e']
                target_population = internal_population_dict[isource, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['e'],
                                             nsyn_post=Cell_type_num['i'], weights=LIE[isource, itarget],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)
                
                # Lei
                source_population = internal_population_dict[isource, 'i']
                target_population = internal_population_dict[itarget, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['e'], weights=-LEI[itarget, isource],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'i']
                target_population = internal_population_dict[isource, 'e']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['e'], weights=-LEI[isource, itarget],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                # siiself
                source_population = internal_population_dict[isource, 'i']
                target_population = internal_population_dict[itarget, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['i'], weights=-LII[itarget, isource],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

                source_population = internal_population_dict[itarget, 'i']
                target_population = internal_population_dict[isource, 'i']
                curr_connection = Connection(source_population, target_population, nsyn=Cell_type_num['i'],
                                             nsyn_post=Cell_type_num['i'], weights=-LII[isource, itarget],
                                             probs=1.0, conn_type='LongRange', v_min=-1.0, v_max=1.0, dv=dv)
                connection_list.append(curr_connection)

    return connection_list


# Net_settings,Hypm,Pham,Orim,index_2_loc = create_functional_columns(Net_settings,Fun_map_settings,ori)
(dxx, dyy) = (500.0 / Net_settings['xn'], 500.0 / Net_settings['yn'])
dxx = dxx ** 2
dyy = dyy ** 2
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. Connection Done >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>..
connection_list = cortical_to_cortical_connection(background_population_dict, internal_population_dict, DEE, DIE, DEI,
                                                  DII, LEE, LIE, fE, fI, 0.0)

"""
"""
tau_d = 128.0
simulation = Simulation(population_list, connection_list, Net_settings, Cell_type_num, DEE+LEE/tau_d, DIE+DIE/tau_d, DEI+LEI/tau_d, DII+LII/tau_d, LEE, LIE,
                         LEI, LII, tau_d,verbose=True)
(mEbin_ra, mIbin_ra, rEbin_ra, rIbin_ra, P_MFEbin_ra, NMDAEbin_ra, NMDAIbin_ra, VEavgbin_ra,
 VIavgbin_ra, VEstdbin_ra, VIstdbin_ra, \
 VEmubin_ra, VImubin_ra, VEsigbin_ra, VIsigbin_ra) = simulation.update(t0=t0, dt=dt, tf=tf)

#filename = str(time.strftime(ISOTIMEFORMAT)) + '_.mat'
#scio.savemat(filename,
#             {'mEbin_ra': mEbin_ra, 'mIbin_ra': mIbin_ra, 'VEavgbin_ra': VEavgbin_ra, 'VIavgbin_ra': VIavgbin_ra, \
#              'VEstdbin_ra': VEstdbin_ra, 'VIstdbin_ra': VIstdbin_ra, 'rEbin_ra': rEbin_ra, 'rIbin_ra': rIbin_ra, \
#              'P_MFEbin_ra': P_MFEbin_ra, 'VEmubin_ra': VEmubin_ra, 'VImubin_ra': VImubin_ra,
#              'VEsigbin_ra': VEsigbin_ra, 'VIsigbin_ra': VIsigbin_ra})
#filelrname = str(time.strftime(ISOTIMEFORMAT)) + '_LR.mat'
#scio.savemat(filelrname, {'NMDAEbin_ra': NMDAEbin_ra, 'NMDAIbin_ra': NMDAIbin_ra, 'HNMDAEbin_ra': HNMDAEbin_ra,
#                          'HNMDAIbin_ra': HNMDAIbin_ra})



