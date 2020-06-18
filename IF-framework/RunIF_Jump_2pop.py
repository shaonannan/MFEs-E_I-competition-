#-*- coding: utf-8 -*-
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
# import matplotlib.pylab as plt    
    
Net_params = {'num_populations':2,
              'numexc_ppopulations':100,
              'numinh_ppopulations':100,
              'cell_type':['e','i'],
        }
time_params = {'tauf_e': 5,
               'tauf_i':7,
               'taur_e': 1,
               'taur_i':1,
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
DEE = np.zeros((num_pop, num_pop))
DEI = np.zeros((num_pop, num_pop))
DII = np.zeros((num_pop, num_pop))
DIE = np.zeros((num_pop, num_pop))
## 04/04/2020
#DEE[0,0] = 0.32*1.1
#DEE[1,1] = 0.32*1.1
#DIE[0,0] = 0.32*1.1
#DIE[1,1] = 0.32*1.1
#DEI[0,0] = 0.04*1.1
#DEI[1,1] = 0.04*1.1
#DII[0,0] = 0.04*1.1
#DII[1,1] = 0.04*1.1
#
#DEE[0,1] = 0.015*1.0
#DEE[1,0] = 0.015*1.0
#DIE[1,0] = 0.015*1.0
#DIE[0,1] = 0.015*1.0
#DEI[0,1] = 0.15*1.0
#DEI[1,0] = 0.15*1.0
#DII[0,1] = 0.15*1.0
#DII[1,0] = 0.15*1.0
# >>>>>>>>>>>>>>>> i to i Self Connections >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
DEE[0,0] = 0.308*1.15
DEE[1,1] = 0.308*1.15
DIE[0,0] = 0.308*1.15
DIE[1,1] = 0.308*1.15
DEI[0,0] = 0.0363*1.15
DEI[1,1] = 0.0363*1.15
DII[0,0] = 0.0363*1.15
DII[1,1] = 0.0363*1.15
# >>>>>>>>>>>>>>> i to j Cross Fast Connections >>>>>>>>>>>>>>>>>>>>>>>>>
DEE[0,1] = 0.03
DEE[1,0] = 0.03
DIE[1,0] = 0.03
DIE[0,1] = 0.03
DEI[0,1] = 0.15
DEI[1,0] = 0.15
DII[0,1] = 0.15
DII[1,0] = 0.15     
# homogeneous conns within identical population
for iprepop in range(2):
    for ipostpop in range(2):
        conn_type={'space_scale':'short',
                   'time_scale':'fast',
                   'tau_d':1,
                }
        # fast E2E
        pre,post = neurons[iprepop,'e'],neurons[ipostpop,'e']
        conn = Connection(pre,post,DEE[iprepop,ipostpop]/nenum_pop,conn_type)
        connection_list.append(conn)
        # fast E2I
        pre,post = neurons[iprepop,'e'],neurons[ipostpop,'i']
        conn = Connection(pre,post,DIE[iprepop,ipostpop]/nenum_pop,conn_type)
        connection_list.append(conn)
        # fast I2E
        pre,post = neurons[iprepop,'i'],neurons[ipostpop,'e']
        conn = Connection(pre,post,DEI[iprepop,ipostpop]/ninum_pop,conn_type)
        connection_list.append(conn)
        # fast I2I
        pre,post = neurons[iprepop,'i'],neurons[ipostpop,'i']
        conn = Connection(pre,post,DII[iprepop,ipostpop]/ninum_pop,conn_type)
        connection_list.append(conn)
## fast I2E
#pre,post = neurons[0,'i'],neurons[1,'e']
#conn = Connection(pre,post,0.01*0.02,conn_type)
#connection_list.append(conn)
## fast I2I
#pre,post = neurons[0,'i'],neurons[1,'i']
#conn = Connection(pre,post,0.01*0.02,conn_type)
#connection_list.append(conn)
#
## fast I2E
#pre,post = neurons[1,'i'],neurons[0,'e']
#conn = Connection(pre,post,0.01*0.02,conn_type)
#connection_list.append(conn)
## fast I2I
#pre,post = neurons[1,'i'],neurons[0,'i']
#conn = Connection(pre,post,0.01*0.02,conn_type)
#connection_list.append(conn)
## >>>>>>>>>>>>>>> i to j Cross Long-range Slow Connections >>>>>>>>>>>>>>>
#LEI = np.zeros((num_pop, num_pop))
#LII = np.zeros((num_pop, num_pop))
#LEI[0, 1] = 0.10 # 
#LEI[1, 0] = 0.10 # 
#LII[0, 1] = 0.02 # 
#LII[1, 0] = 0.02 # 
#>>>>>>>>>>>>>>> i to j Cross Long-range Slow Connections >>>>>>>>>>>>>>>
# LEE = np.zeros((num_pop, num_pop))
# LIE = np.zeros((num_pop, num_pop))
# factor,ratio = 0.102,1.2#0.102
# LEE[0,1] = 0.285*factor*1.28*1.07*10.0
# LEE[1,0] = 0.285*factor*1.28*1.07*10.0
# LIE[0,1] = 0.285*factor*ratio*1.28*1.07*10.0
# LIE[1,0] = 0.285*factor*ratio*1.28*1.07*10.0
LEE = np.zeros((num_pop, num_pop))
LIE = np.zeros((num_pop, num_pop))
factor,ratio = 0.102,5.0#0.102
LEE[0,1] = 0.285*factor*10.0
LEE[1,0] = 0.285*factor*10.0
LIE[0,1] = 0.285*factor*ratio*10.0
LIE[1,0] = 0.285*factor*ratio*10.0

connection_list.append(conn)
conn_type={'space_scale':'long',
           'time_scale':'slow',
           'tau_d':128,
           }
 
pre,post = neurons[1,'e'],neurons[0,'e']
conn = Connection(pre,post,LEE[1,0]/nenum_pop,conn_type)
connection_list.append(conn)
pre,post = neurons[0,'e'],neurons[1,'e']
conn = Connection(pre,post,LEE[0,1]/nenum_pop,conn_type)
connection_list.append(conn)


pre,post = neurons[1,'e'],neurons[0,'i']
conn = Connection(pre,post,LIE[1,0]/nenum_pop,conn_type)
connection_list.append(conn)
pre,post = neurons[0,'e'],neurons[1,'i']
conn = Connection(pre,post,LIE[0,1]/nenum_pop,conn_type)
connection_list.append(conn)

#pre,post = neurons[1,'i'],neurons[0,'e']
#conn = Connection(pre,post,-LEI[1,0]/ninum_pop,conn_type)
#connection_list.append(conn)
#pre,post = neurons[0,'i'],neurons[1,'e']
#conn = Connection(pre,post,-LEI[0,1]/ninum_pop,conn_type)
#connection_list.append(conn)
#
#
#pre,post = neurons[1,'i'],neurons[0,'i']
#conn = Connection(pre,post,-LII[1,0]/ninum_pop,conn_type)
#connection_list.append(conn)
#pre,post = neurons[0,'i'],neurons[1,'i']
#conn = Connection(pre,post,-LII[0,1]/ninum_pop,conn_type)
#connection_list.append(conn)


#conn_type={'space_scale':'short',
#           'time_scale':'fast',
#           'tau_d':1,
#           }
#
#pre,post = neurons[1,'i'],neurons[0,'e']
#conn = Connection(pre,post,0.01*4*0.2,conn_type)
#connection_list.append(conn)
#pre,post = neurons[0,'i'],neurons[1,'e']
#conn = Connection(pre,post,0.01*4*0.2,conn_type)
#connection_list.append(conn)
#
#
#pre,post = neurons[1,'i'],neurons[0,'i']
#conn = Connection(pre,post,0.01*4*0.2,conn_type)
#connection_list.append(conn)
#pre,post = neurons[0,'i'],neurons[1,'i']
#conn = Connection(pre,post,0.01*4*0.2,conn_type)
#connection_list.append(conn)

Plat = platform(nneurons_list,connection_list,Net_params)
# initialize point neurons and connections
Plat.init_allneurons()
Plat.init_allconns()

#Plat.VEmat[1,0] = 0.96;
#Plat.VImat[2,0] = 0.60;
#
#for nn in Plat.neurons_list:
#    nn.update_Vpoint()

# Initializing a sequence of time parameters
xEbin_ra,xIbin_ra  = np.zeros((10010,2)),np.zeros((10010,2))
VEavgbin_ra,VIavgbin_ra  = np.zeros((10010,2)),np.zeros((10010,2))
VEstdbin_ra,VIstdbin_ra  = np.zeros((10010,2)),np.zeros((10010,2))
vbins = int(np.floor(1.0/(0.015/0.80)))
rEpre,rIpre={},{}
rEpos,rIpos={},{}
rEorg,rIorg={},{}
rErec,rIrec={},{}
LE,LI=0,0
dtmax = time_params['dt']
TMAX = 10000.0;tbinsize=1.0;dtbin_ij=0
t_sum=0; dt_cur=0; T_rem=TMAX; dt_rem=tbinsize;
# Initializing a sequence of voltage parameters
NmEY = np.ones((1,num_pop))*2.5*1.2*nenum_pop
NmIY = np.ones((1,num_pop))*2.0*1.32*ninum_pop
DIY = np.ones((1,num_pop))*0.013/0.85
DEY = np.ones((1,num_pop))*0.013/0.85

# Initializing a sequence of voltage parameters
NmEY = np.ones((1,num_pop))*3.11*0.96*nenum_pop
NmIY = np.ones((1,num_pop))*2.30*1.02*ninum_pop
DIY = np.ones((1,num_pop))*0.013145/0.90
DEY = np.ones((1,num_pop))*0.01316/0.90

# Initializing a sequence of voltage parameters
NmEY2 = np.ones((1,num_pop))*3.11*0.78*nenum_pop
DEY2 = np.ones((1,num_pop))*0.01316/0.74
NmIY2 = np.ones((1,num_pop))*2.30*0.74*ninum_pop
DIY2 = np.ones((1,num_pop))*0.013145/0.74

NmEY1 = np.ones((1,num_pop))*3.11*0.82*nenum_pop
DEY1 = np.ones((1,num_pop))*0.01316/0.72
NmIY1 = np.ones((1,num_pop))*2.30*0.82*ninum_pop
DIY1 = np.ones((1,num_pop))*0.013145/0.72

# # Initializing a sequence of voltage parameters
# NmEY2 = np.ones((1,num_pop))*3.11*0.74*nenum_pop
# DEY2 = np.ones((1,num_pop))*0.01316/0.74
# NmIY2 = np.ones((1,num_pop))*2.30*0.74*ninum_pop
# DIY2 = np.ones((1,num_pop))*0.013145/0.74

VT = 1.0
RE,REj = np.zeros((1,num_pop)),np.zeros((1,num_pop),dtype=int)
RI,RIj = np.zeros((1,num_pop)),np.zeros((1,num_pop),dtype=int)
for i in range(num_pop):
    RE[0,i]  = -np.log(np.random.rand())/NmEY[0,i]
    REj[0,i] = 0+min(nenum_pop-1,max(0,np.floor(nenum_pop*np.random.rand())))
    RI[0,i]  = -np.log(np.random.rand())/NmIY[0,i] 
    RIj[0,i] = 0+min(ninum_pop-1,max(0,np.floor(ninum_pop*np.random.rand())))

# start updating and evoluting
while t_sum<TMAX:
    if t_sum<00:
        NmEY= NmEY1
        DEY = DEY1
        NmIY= NmIY1
        DIY = DIY1
    else:
        NmEY = NmEY2
        DEY  = DEY2
        NmIY = NmIY2
        DIY  = DIY2
    min_RI,idx_RI = min(RI[0,:]),np.argmin(RI[0,:])
    min_RE,idx_RE = min(RE[0,:]),np.argmin(RE[0,:])
    if min_RI<min_RE:
        min_RT = min_RI
        idx_RT = idx_RI
    elif min_RI>=min_RE:
        min_RT = min_RE
        idx_RT = idx_RE
    if np.mod(dtbin_ij,100)==0 and (t_sum-dtbin_ij)<=dtmax:
        print('Time: ',t_sum)
        idxE,idxI = 0,0
        for nn in Plat.neurons_list:
            if nn.eitype == 'e':
                print('E pop ',idxE,' Inmda:')
                for key,val in nn.accum_Islow_dicts.items():
                    print(key,':',val/1.0)
                idxE+=1
            elif nn.eitype == 'i':
                print('I pop ',idxE,' Inmda:')
                for key,val in nn.accum_Islow_dicts.items():
                    print(key,':',val/1.0)
                idxI+=1            
    # if (np.mod(dtbin_ij,99)==1) and ((dtbin_ij+1)*tbinsize-t_sum<1.1*dtmax)and ((dtbin_ij+1)*tbinsize-t_sum >0):
    #      for i in range(num_pop):
    #          idshown = np.floor(dtbin_ij/2000)
    #          idstart = np.int(idshown*2000)
    #          if idstart == dtbin_ij:
    #              idstart = idstart -1               
    #          idend   = min((idshown+1)*2000,4000)
    #          ttt = np.arange(idstart,dtbin_ij) * 1.0
    #          plt.figure(10)
    #          plt.subplot(num_pop,1,int(i)+1)
    #          plt.plot(ttt,xEbin_ra[idstart:dtbin_ij,i],'r')
    #          plt.xlim([ttt[0],ttt[0]+2000])
    #          plt.ylim([0,nenum_pop])
    #          plt.pause(0.1)
    #          plt.figure(11)
    #          plt.subplot(num_pop,1,int(i)+1)
    #          plt.plot(ttt,xIbin_ra[idstart:dtbin_ij,i],'b')
    #          plt.xlim([ttt[0],ttt[0]+2000])
    #          # plt.xlim([0,int(self.tf)])
    #          plt.ylim([0,ninum_pop])
    #          plt.pause(0.1)
    if (T_rem<=min_RT)and(T_rem<=dtmax):
#        print('condition 1')
        dt_cur = T_rem
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        if LE>0 or LI>0:
            for idx in range(num_pop):
                LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                LEp,LIp = int(LEp),int(LIp)
                VE_rechist,_ = np.histogram(np.squeeze(Plat.VEmat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                VI_rechist,_ = np.histogram(np.squeeze(Plat.VImat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)            
                try:
                    rErec[LE,LI,LEp,LIp,idx]=np.append(rErec[LE,LI,LEp,LIp,idx],np.reshape(VE_rechist,(vbins,1)),1)
                    rIrec[LE,LI,LEp,LIp,idx]=np.append(rIrec[LE,LI,LEp,LIp,idx],np.reshape(VI_rechist,(vbins,1)),1)

                except:
                    rErec[LE,LI,LEp,LIp,idx]=np.reshape(VE_rechist,(vbins,1))
                    rIrec[LE,LI,LEp,LIp,idx]=np.reshape(VI_rechist,(vbins,1)) 
        LE,LI=0,0
        Plat.update_Islow(dtmax)
        '''
        '''
        max_VE = max(np.reshape(Plat.VEmat,(num_pop*nenum_pop)))
        max_VI = max(np.reshape(Plat.VImat,(num_pop*ninum_pop)))
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        if (max_VE>VT) or (max_VI>VT):
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
            if(t_sum>1000.0):
                LE,LI = np.size(np.squeeze(E_fired)),np.size(np.squeeze(I_fired))
                # calculate hist
                for idx in range(num_pop):
                    # LE/LI per pop
                    LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                    LEp,LIp = int(LEp),int(LIp)
                    # pre hist
                    VE_orghist,_ = np.histogram(np.squeeze(VE_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_orghist,_ = np.histogram(np.squeeze(VI_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_prehist,_ = np.histogram(np.squeeze(VE_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_prehist,_ = np.histogram(np.squeeze(VI_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_poshist,_ = np.histogram(np.squeeze(VE_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_poshist,_ = np.histogram(np.squeeze(VI_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    try:
                        rEorg[LE,LI,LEp,LIp,idx]=np.append(rEorg[LE,LI,LEp,LIp,idx],np.reshape(VE_orghist,(vbins,1)),1)
                        rIorg[LE,LI,LEp,LIp,idx]=np.append(rIorg[LE,LI,LEp,LIp,idx],np.reshape(VI_orghist,(vbins,1)),1)
                        rEpre[LE,LI,LEp,LIp,idx]=np.append(rEpre[LE,LI,LEp,LIp,idx],np.reshape(VE_prehist,(vbins,1)),1)
                        rIpre[LE,LI,LEp,LIp,idx]=np.append(rIpre[LE,LI,LEp,LIp,idx],np.reshape(VI_prehist,(vbins,1)),1)
                        rEpos[LE,LI,LEp,LIp,idx]=np.append(rEpos[LE,LI,LEp,LIp,idx],np.reshape(VE_poshist,(vbins,1)),1)
                        rIpos[LE,LI,LEp,LIp,idx]=np.append(rIpos[LE,LI,LEp,LIp,idx],np.reshape(VI_poshist,(vbins,1)),1)
                    except:
                        rEorg[LE,LI,LEp,LIp,idx]=np.reshape(VE_orghist,(vbins,1))
                        rIorg[LE,LI,LEp,LIp,idx]=np.reshape(VI_orghist,(vbins,1))
                        rEpre[LE,LI,LEp,LIp,idx]=np.reshape(VE_prehist,(vbins,1))
                        rIpre[LE,LI,LEp,LIp,idx]=np.reshape(VI_prehist,(vbins,1))
                        rEpos[LE,LI,LEp,LIp,idx]=np.reshape(VE_poshist,(vbins,1))
                        rIpos[LE,LI,LEp,LIp,idx]=np.reshape(VI_poshist,(vbins,1))                        
        else:
            Plat.update_vain()
        '''
        Plat.update_vain() # 21/04       
        '''         
        # update firing recording
        idxE,idxI = 0,0
        for nn in Plat.neurons_list:
            if nn.eitype == 'e':
                xEbin_ra[dtbin_ij,idxE] += nn.total_firing_rate
                VEavgbin_ra[dtbin_ij,idxE] = VEavgbin_ra[dtbin_ij,idxE] + dt_cur*np.mean(nn.vol); 
                VEstdbin_ra[dtbin_ij,idxE] = VEstdbin_ra[dtbin_ij,idxE] + dt_cur*np.std(nn.vol); 
                idxE+=1
            elif nn.eitype == 'i':
                xIbin_ra[dtbin_ij,idxI] += nn.total_firing_rate
                VIavgbin_ra[dtbin_ij,idxI] = VIavgbin_ra[dtbin_ij,idxI] + dt_cur*np.mean(nn.vol); 
                VIstdbin_ra[dtbin_ij,idxI] = VIstdbin_ra[dtbin_ij,idxI] + dt_cur*np.std(nn.vol); 
                idxI+=1
    elif (dtmax<=min_RT)and(dtmax<=T_rem):
#        print('condition 2')
        dt_cur = dtmax
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dtmax)
        if LE>0 or LI>0:
            for idx in range(num_pop):
                LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                LEp,LIp = int(LEp),int(LIp)
                VE_rechist,_ = np.histogram(np.squeeze(Plat.VEmat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                VI_rechist,_ = np.histogram(np.squeeze(Plat.VImat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)            
                try:
                    rErec[LE,LI,LEp,LIp,idx]=np.append(rErec[LE,LI,LEp,LIp,idx],np.reshape(VE_rechist,(vbins,1)),1)
                    rIrec[LE,LI,LEp,LIp,idx]=np.append(rIrec[LE,LI,LEp,LIp,idx],np.reshape(VI_rechist,(vbins,1)),1)

                except:
                    rErec[LE,LI,LEp,LIp,idx]=np.reshape(VE_rechist,(vbins,1))
                    rIrec[LE,LI,LEp,LIp,idx]=np.reshape(VI_rechist,(vbins,1)) 
        LE,LI=0,0
        '''
        '''
        max_VE = max(np.reshape(Plat.VEmat,(num_pop*nenum_pop)))
        max_VI = max(np.reshape(Plat.VImat,(num_pop*ninum_pop)))
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        if (max_VE>VT) or (max_VI>VT):
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
            if(t_sum>1000.0):
                LE,LI = np.size(np.squeeze(E_fired)),np.size(np.squeeze(I_fired))
                # calculate hist
                for idx in range(num_pop):
                    # LE/LI per pop
                    LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                    LEp,LIp = int(LEp),int(LIp)
                    # pre hist
                    VE_orghist,_ = np.histogram(np.squeeze(VE_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_orghist,_ = np.histogram(np.squeeze(VI_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_prehist,_ = np.histogram(np.squeeze(VE_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_prehist,_ = np.histogram(np.squeeze(VI_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_poshist,_ = np.histogram(np.squeeze(VE_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_poshist,_ = np.histogram(np.squeeze(VI_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    try:
                        rEorg[LE,LI,LEp,LIp,idx]=np.append(rEorg[LE,LI,LEp,LIp,idx],np.reshape(VE_orghist,(vbins,1)),1)
                        rIorg[LE,LI,LEp,LIp,idx]=np.append(rIorg[LE,LI,LEp,LIp,idx],np.reshape(VI_orghist,(vbins,1)),1)
                        rEpre[LE,LI,LEp,LIp,idx]=np.append(rEpre[LE,LI,LEp,LIp,idx],np.reshape(VE_prehist,(vbins,1)),1)
                        rIpre[LE,LI,LEp,LIp,idx]=np.append(rIpre[LE,LI,LEp,LIp,idx],np.reshape(VI_prehist,(vbins,1)),1)
                        rEpos[LE,LI,LEp,LIp,idx]=np.append(rEpos[LE,LI,LEp,LIp,idx],np.reshape(VE_poshist,(vbins,1)),1)
                        rIpos[LE,LI,LEp,LIp,idx]=np.append(rIpos[LE,LI,LEp,LIp,idx],np.reshape(VI_poshist,(vbins,1)),1)
                    except:
                        rEorg[LE,LI,LEp,LIp,idx]=np.reshape(VE_orghist,(vbins,1))
                        rIorg[LE,LI,LEp,LIp,idx]=np.reshape(VI_orghist,(vbins,1))
                        rEpre[LE,LI,LEp,LIp,idx]=np.reshape(VE_prehist,(vbins,1))
                        rIpre[LE,LI,LEp,LIp,idx]=np.reshape(VI_prehist,(vbins,1))
                        rEpos[LE,LI,LEp,LIp,idx]=np.reshape(VE_poshist,(vbins,1))
                        rIpos[LE,LI,LEp,LIp,idx]=np.reshape(VI_poshist,(vbins,1))                        
        else:
            Plat.update_vain()
        '''
        Plat.update_vain()
        '''
        # update firing recording
        idxE,idxI = 0,0
        for nn in Plat.neurons_list:
            if nn.eitype == 'e':
                xEbin_ra[dtbin_ij,idxE] += nn.total_firing_rate
                VEavgbin_ra[dtbin_ij,idxE] = VEavgbin_ra[dtbin_ij,idxE] + dt_cur*np.mean(nn.vol); 
                VEstdbin_ra[dtbin_ij,idxE] = VEstdbin_ra[dtbin_ij,idxE] + dt_cur*np.std(nn.vol); 
                idxE+=1
            elif nn.eitype == 'i':
                xIbin_ra[dtbin_ij,idxI] += nn.total_firing_rate
                VIavgbin_ra[dtbin_ij,idxI] = VIavgbin_ra[dtbin_ij,idxI] + dt_cur*np.mean(nn.vol); 
                VIstdbin_ra[dtbin_ij,idxI] = VIstdbin_ra[dtbin_ij,idxI] + dt_cur*np.std(nn.vol); 
                idxI+=1
#        print(Plat.VEmat[:,0])
    # trigger an Inhibitory neuron
    elif ((min_RI<min_RE)and(min_RI<dtmax)and(min_RI<T_rem)):
#        print('condition 3')
        dt_cur = min_RI
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur 
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dt_cur)
        if LE>0 or LI>0:
            for idx in range(num_pop):
                LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                LEp,LIp = int(LEp),int(LIp)
                VE_rechist,_ = np.histogram(np.squeeze(Plat.VEmat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                VI_rechist,_ = np.histogram(np.squeeze(Plat.VImat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)            
                try:
                    rErec[LE,LI,LEp,LIp,idx]=np.append(rErec[LE,LI,LEp,LIp,idx],np.reshape(VE_rechist,(vbins,1)),1)
                    rIrec[LE,LI,LEp,LIp,idx]=np.append(rIrec[LE,LI,LEp,LIp,idx],np.reshape(VI_rechist,(vbins,1)),1)

                except:
                    rErec[LE,LI,LEp,LIp,idx]=np.reshape(VE_rechist,(vbins,1))
                    rIrec[LE,LI,LEp,LIp,idx]=np.reshape(VI_rechist,(vbins,1)) 
        LE,LI=0,0
        # refresh VImat[RIj[0,idx_RI],idx_RI]
        Plat.VImat[RIj[0,idx_RI],idx_RI] += DIY[0,idx_RI]
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        '''
        '''
        max_VE = max(np.reshape(Plat.VEmat,(num_pop*nenum_pop)))
        max_VI = max(np.reshape(Plat.VImat,(num_pop*ninum_pop)))
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        if (Plat.VImat[RIj[0,idx_RI],idx_RI]>=VT) or (max_VE>VT) or (max_VI>VT):
        # '''
        # if (Plat.VImat[RIj[0,idx_RI],idx_RI]>=VT) :
        # '''
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
            if(t_sum>1000.0):
                LE,LI = np.size(np.squeeze(E_fired)),np.size(np.squeeze(I_fired))
                # calculate hist
                for idx in range(num_pop):
                    # LE/LI per pop
                    LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                    LEp,LIp = int(LEp),int(LIp)
                    # pre hist
                    VE_orghist,_ = np.histogram(np.squeeze(VE_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_orghist,_ = np.histogram(np.squeeze(VI_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_prehist,_ = np.histogram(np.squeeze(VE_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_prehist,_ = np.histogram(np.squeeze(VI_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_poshist,_ = np.histogram(np.squeeze(VE_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_poshist,_ = np.histogram(np.squeeze(VI_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    try:
                        rEorg[LE,LI,LEp,LIp,idx]=np.append(rEorg[LE,LI,LEp,LIp,idx],np.reshape(VE_orghist,(vbins,1)),1)
                        rIorg[LE,LI,LEp,LIp,idx]=np.append(rIorg[LE,LI,LEp,LIp,idx],np.reshape(VI_orghist,(vbins,1)),1)
                        rEpre[LE,LI,LEp,LIp,idx]=np.append(rEpre[LE,LI,LEp,LIp,idx],np.reshape(VE_prehist,(vbins,1)),1)
                        rIpre[LE,LI,LEp,LIp,idx]=np.append(rIpre[LE,LI,LEp,LIp,idx],np.reshape(VI_prehist,(vbins,1)),1)
                        rEpos[LE,LI,LEp,LIp,idx]=np.append(rEpos[LE,LI,LEp,LIp,idx],np.reshape(VE_poshist,(vbins,1)),1)
                        rIpos[LE,LI,LEp,LIp,idx]=np.append(rIpos[LE,LI,LEp,LIp,idx],np.reshape(VI_poshist,(vbins,1)),1)
                    except:
                        rEorg[LE,LI,LEp,LIp,idx]=np.reshape(VE_orghist,(vbins,1))
                        rIorg[LE,LI,LEp,LIp,idx]=np.reshape(VI_orghist,(vbins,1))
                        rEpre[LE,LI,LEp,LIp,idx]=np.reshape(VE_prehist,(vbins,1))
                        rIpre[LE,LI,LEp,LIp,idx]=np.reshape(VI_prehist,(vbins,1))
                        rEpos[LE,LI,LEp,LIp,idx]=np.reshape(VE_poshist,(vbins,1))
                        rIpos[LE,LI,LEp,LIp,idx]=np.reshape(VI_poshist,(vbins,1))    
        else:
            Plat.update_vain()
#        print(Plat.VImat[:,0])
            
        # update firing recording
        idxE,idxI = 0,0
        for nn in Plat.neurons_list:
            if nn.eitype == 'e':
                xEbin_ra[dtbin_ij,idxE] += nn.total_firing_rate
                VEavgbin_ra[dtbin_ij,idxE] = VEavgbin_ra[dtbin_ij,idxE] + dt_cur*np.mean(nn.vol); 
                VEstdbin_ra[dtbin_ij,idxE] = VEstdbin_ra[dtbin_ij,idxE] + dt_cur*np.std(nn.vol); 
                idxE+=1
            elif nn.eitype == 'i':
                xIbin_ra[dtbin_ij,idxI] += nn.total_firing_rate
                VIavgbin_ra[dtbin_ij,idxI] = VIavgbin_ra[dtbin_ij,idxI] + dt_cur*np.mean(nn.vol); 
                VIstdbin_ra[dtbin_ij,idxI] = VIstdbin_ra[dtbin_ij,idxI] + dt_cur*np.std(nn.vol); 
                idxI+=1
        RI[0,idx_RI] = -np.log(np.random.rand())/NmIY[0,idx_RI]
        RIj[0,idx_RI] = 0+min(ninum_pop-1,max(0,np.floor(ninum_pop*np.random.rand())))
    elif ((min_RE<min_RI)and(min_RE<dtmax)and(min_RE<T_rem)):
#        print('condition 4')
        dt_cur = min_RE
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur 
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dt_cur) 
        if LE>0 or LI>0:
            for idx in range(num_pop):
                LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                LEp,LIp = int(LEp),int(LIp)
                VE_rechist,_ = np.histogram(np.squeeze(Plat.VEmat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                VI_rechist,_ = np.histogram(np.squeeze(Plat.VImat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)            
                try:
                    rErec[LE,LI,LEp,LIp,idx]=np.append(rErec[LE,LI,LEp,LIp,idx],np.reshape(VE_rechist,(vbins,1)),1)
                    rIrec[LE,LI,LEp,LIp,idx]=np.append(rIrec[LE,LI,LEp,LIp,idx],np.reshape(VI_rechist,(vbins,1)),1)

                except:
                    rErec[LE,LI,LEp,LIp,idx]=np.reshape(VE_rechist,(vbins,1))
                    rIrec[LE,LI,LEp,LIp,idx]=np.reshape(VI_rechist,(vbins,1)) 
        LE,LI=0,0
        Plat.VEmat[REj[0,idx_RE],idx_RE] += DEY[0,idx_RE]
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        '''
        '''
        max_VE = max(np.reshape(Plat.VEmat,(num_pop*nenum_pop)))
        max_VI = max(np.reshape(Plat.VImat,(num_pop*ninum_pop)))
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        if (Plat.VEmat[REj[0,idx_RE],idx_RE]>=VT) or (max_VE>VT) or (max_VI>VT):
        # '''
        # if (Plat.VEmat[REj[0,idx_RE],idx_RE]>=VT):
        # '''
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
            if(t_sum>1000.0):
                LE,LI = np.size(np.squeeze(E_fired)),np.size(np.squeeze(I_fired))
                # calculate hist
                for idx in range(num_pop):
                    # LE/LI per pop
                    LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                    LEp,LIp = int(LEp),int(LIp)
                    # pre hist
                    VE_orghist,_ = np.histogram(np.squeeze(VE_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_orghist,_ = np.histogram(np.squeeze(VI_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_prehist,_ = np.histogram(np.squeeze(VE_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_prehist,_ = np.histogram(np.squeeze(VI_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_poshist,_ = np.histogram(np.squeeze(VE_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_poshist,_ = np.histogram(np.squeeze(VI_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    try:
                        rEorg[LE,LI,LEp,LIp,idx]=np.append(rEorg[LE,LI,LEp,LIp,idx],np.reshape(VE_orghist,(vbins,1)),1)
                        rIorg[LE,LI,LEp,LIp,idx]=np.append(rIorg[LE,LI,LEp,LIp,idx],np.reshape(VI_orghist,(vbins,1)),1)
                        rEpre[LE,LI,LEp,LIp,idx]=np.append(rEpre[LE,LI,LEp,LIp,idx],np.reshape(VE_prehist,(vbins,1)),1)
                        rIpre[LE,LI,LEp,LIp,idx]=np.append(rIpre[LE,LI,LEp,LIp,idx],np.reshape(VI_prehist,(vbins,1)),1)
                        rEpos[LE,LI,LEp,LIp,idx]=np.append(rEpos[LE,LI,LEp,LIp,idx],np.reshape(VE_poshist,(vbins,1)),1)
                        rIpos[LE,LI,LEp,LIp,idx]=np.append(rIpos[LE,LI,LEp,LIp,idx],np.reshape(VI_poshist,(vbins,1)),1)
                    except:
                        rEorg[LE,LI,LEp,LIp,idx]=np.reshape(VE_orghist,(vbins,1))
                        rIorg[LE,LI,LEp,LIp,idx]=np.reshape(VI_orghist,(vbins,1))
                        rEpre[LE,LI,LEp,LIp,idx]=np.reshape(VE_prehist,(vbins,1))
                        rIpre[LE,LI,LEp,LIp,idx]=np.reshape(VI_prehist,(vbins,1))
                        rEpos[LE,LI,LEp,LIp,idx]=np.reshape(VE_poshist,(vbins,1))
                        rIpos[LE,LI,LEp,LIp,idx]=np.reshape(VI_poshist,(vbins,1))  
        else:
            Plat.update_vain()
        # update firing recording
        idxE,idxI = 0,0
        for nn in Plat.neurons_list:
            if nn.eitype == 'e':
                xEbin_ra[dtbin_ij,idxE] += nn.total_firing_rate
                VEavgbin_ra[dtbin_ij,idxE] = VEavgbin_ra[dtbin_ij,idxE] + dt_cur*np.mean(nn.vol); 
                VEstdbin_ra[dtbin_ij,idxE] = VEstdbin_ra[dtbin_ij,idxE] + dt_cur*np.std(nn.vol); 
                idxE+=1
            elif nn.eitype == 'i':
                xIbin_ra[dtbin_ij,idxI] += nn.total_firing_rate
                VIavgbin_ra[dtbin_ij,idxI] = VIavgbin_ra[dtbin_ij,idxI] + dt_cur*np.mean(nn.vol); 
                VIstdbin_ra[dtbin_ij,idxI] = VIstdbin_ra[dtbin_ij,idxI] + dt_cur*np.std(nn.vol); 
                idxI+=1
#        print(Plat.VEmat[:,1])
        RE[0,idx_RE] = -np.log(np.random.rand())/NmEY[0,idx_RE]
        REj[0,idx_RE] = 0+min(nenum_pop-1,max(0,np.floor(nenum_pop*np.random.rand())))
    elif ((min_RE==min_RI)and(min_RE<dtmax)and(min_RE<T_rem)):
#        print('condition 5')
        dt_cur = min_RE
        RE = RE-dt_cur*np.ones_like(RE)
        RI = RI-dt_cur*np.ones_like(RI)
        t_sum += dt_cur; T_rem = TMAX-dt_cur; dt_rem -= dt_cur 
        if dt_rem<0: 
            dt_rem += tbinsize; dtbin_ij += 1; 
        Plat.update_Islow(dt_cur) 
        if LE>0 or LI>0:
            for idx in range(num_pop):
                LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                LEp,LIp = int(LEp),int(LIp)
                VE_rechist,_ = np.histogram(np.squeeze(Plat.VEmat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                VI_rechist,_ = np.histogram(np.squeeze(Plat.VImat[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)            
                try:
                    rErec[LE,LI,LEp,LIp,idx]=np.append(rErec[LE,LI,LEp,LIp,idx],np.reshape(VE_rechist,(vbins,1)),1)
                    rIrec[LE,LI,LEp,LIp,idx]=np.append(rIrec[LE,LI,LEp,LIp,idx],np.reshape(VI_rechist,(vbins,1)),1)

                except:
                    rErec[LE,LI,LEp,LIp,idx]=np.reshape(VE_rechist,(vbins,1))
                    rIrec[LE,LI,LEp,LIp,idx]=np.reshape(VI_rechist,(vbins,1)) 
        LE,LI=0,0
        # refresh VImat[RIj[0,idx_RI],idx_RI]
        Plat.VEmat[REj[0,idx_RE],idx_RE] += DEY[0,idx_RE]
        Plat.VImat[RIj[0,idx_RI],idx_RI] += DIY[0,idx_RI]
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        '''
        '''
        max_VE = max(np.reshape(Plat.VEmat,(num_pop*nenum_pop)))
        max_VI = max(np.reshape(Plat.VImat,(num_pop*ninum_pop)))
        VE_org,VI_org = Plat.VEmat,Plat.VImat
        if (Plat.VEmat[REj[0,idx_RE],idx_RE]>=VT)or(Plat.VImat[RIj[0,idx_RI],idx_RI]>=VT) or(max_VE>VT) or (max_VI>VT):
        # '''
        # if (Plat.VEmat[REj[0,idx_RE],idx_RE]>=VT)or(Plat.VImat[RIj[0,idx_RI],idx_RI]>=VT) :
        # '''
            E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos = Plat.update_MFE()
            if(t_sum>1000.0):
                LE,LI = np.size(np.squeeze(E_fired)),np.size(np.squeeze(I_fired))
                # calculate hist
                for idx in range(num_pop):
                    # LE/LI per pop
                    LEp,LIp = np.sum(np.squeeze(Plat.E_fired_num[idx,:])),np.sum(np.squeeze(Plat.I_fired_num[idx,:]))
                    LEp,LIp = int(LEp),int(LIp)
                    # pre hist
                    VE_orghist,_ = np.histogram(np.squeeze(VE_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_orghist,_ = np.histogram(np.squeeze(VI_org[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_prehist,_ = np.histogram(np.squeeze(VE_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_prehist,_ = np.histogram(np.squeeze(VI_pre[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VE_poshist,_ = np.histogram(np.squeeze(VE_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    VI_poshist,_ = np.histogram(np.squeeze(VI_pos[:,idx]),bins=vbins,range=(-1.0,1.0),normed=True,density=True)
                    try:
                        rEorg[LE,LI,LEp,LIp,idx]=np.append(rEorg[LE,LI,LEp,LIp,idx],np.reshape(VE_orghist,(vbins,1)),1)
                        rIorg[LE,LI,LEp,LIp,idx]=np.append(rIorg[LE,LI,LEp,LIp,idx],np.reshape(VI_orghist,(vbins,1)),1)
                        rEpre[LE,LI,LEp,LIp,idx]=np.append(rEpre[LE,LI,LEp,LIp,idx],np.reshape(VE_prehist,(vbins,1)),1)
                        rIpre[LE,LI,LEp,LIp,idx]=np.append(rIpre[LE,LI,LEp,LIp,idx],np.reshape(VI_prehist,(vbins,1)),1)
                        rEpos[LE,LI,LEp,LIp,idx]=np.append(rEpos[LE,LI,LEp,LIp,idx],np.reshape(VE_poshist,(vbins,1)),1)
                        rIpos[LE,LI,LEp,LIp,idx]=np.append(rIpos[LE,LI,LEp,LIp,idx],np.reshape(VI_poshist,(vbins,1)),1)
                    except:
                        rEorg[LE,LI,LEp,LIp,idx]=np.reshape(VE_orghist,(vbins,1))
                        rIorg[LE,LI,LEp,LIp,idx]=np.reshape(VI_orghist,(vbins,1))
                        rEpre[LE,LI,LEp,LIp,idx]=np.reshape(VE_prehist,(vbins,1))
                        rIpre[LE,LI,LEp,LIp,idx]=np.reshape(VI_prehist,(vbins,1))
                        rEpos[LE,LI,LEp,LIp,idx]=np.reshape(VE_poshist,(vbins,1))
                        rIpos[LE,LI,LEp,LIp,idx]=np.reshape(VI_poshist,(vbins,1))    
        else:
            Plat.update_vain()

        # update firing recording
        idxE,idxI = 0,0
        for nn in Plat.neurons_list:
            if nn.eitype == 'e':
                xEbin_ra[dtbin_ij,idxE] += nn.total_firing_rate
                VEavgbin_ra[dtbin_ij,idxE] = VEavgbin_ra[dtbin_ij,idxE] + dt_cur*np.mean(nn.vol); 
                VEstdbin_ra[dtbin_ij,idxE] = VEstdbin_ra[dtbin_ij,idxE] + dt_cur*np.std(nn.vol); 
                idxE+=1
            elif nn.eitype == 'i':
                xIbin_ra[dtbin_ij,idxI] += nn.total_firing_rate
                VIavgbin_ra[dtbin_ij,idxI] = VIavgbin_ra[dtbin_ij,idxI] + dt_cur*np.mean(nn.vol); 
                VIstdbin_ra[dtbin_ij,idxI] = VIstdbin_ra[dtbin_ij,idxI] + dt_cur*np.std(nn.vol); 
                idxI+=1
        RE[0,idx_RE] = -np.log(np.random.rand())/NmEY[0,idx_RE]
        REj[0,idx_RE] = 0+min(nenum_pop-1,max(0,np.floor(nenum_pop*np.random.rand())))
        
        RI[0,idx_RI] = -np.log(np.random.rand())/NmIY[0,idx_RI]
        RIj[0,idx_RI] = 0+min(ninum_pop-1,max(0,np.floor(ninum_pop*np.random.rand())))
def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  
window_size = 500
window_smooth = np.ones(int(window_size ))/float(window_size )    
mEbin_ra = np.zeros_like(xEbin_ra)
mIbin_ra = np.zeros_like(xIbin_ra)
xEbin_ra_ = np.zeros_like(xEbin_ra)
xIbin_ra_ = np.zeros_like(xIbin_ra)
for i in range(2):
    cpE = np.zeros((10010,2))
    cpE[:,0] = xEbin_ra[:,i]-nenum_pop/10.0;cpE[:,1] = 0.0
    xEbin_ra_[:,i] = np.max(cpE,axis = 1)
    mEbin_ra[:,i] = np.convolve(xEbin_ra_[:,i],window_smooth,'same')
    cpI = np.zeros((10010,2))
    cpI[:,0] = xIbin_ra[:,i]-ninum_pop/10.0;cpI[:,1] = 0.0
    xIbin_ra_[:,i] = np.max(cpI,axis = 1)
    mIbin_ra[:,i] = np.convolve(xIbin_ra_[:,i],window_smooth,'same')

# plt.figure(21)        
# plt.subplot(212)
# plt.plot(xEbin_ra[:,1])
# plt.ylim([0,nenum_pop])
# plt.subplot(211)
# plt.plot(xEbin_ra[:,0])
# plt.ylim([0,nenum_pop])
# plt.figure(22)        
# plt.subplot(212)
# plt.plot(mEbin_ra[:,1])
# plt.ylim([0,2])
# plt.subplot(211)
# plt.plot(mEbin_ra[:,0])
# plt.ylim([0,2])

import time
import scipy.io as scio
ISOTIMEFORMAT='%Y%m%d%H%M%S'
fileparamname = str(time.strftime(ISOTIMEFORMAT)) + '_paramsLI.mat'
scio.savemat(fileparamname,
             {'DEE': DEE, 'DEI': DEI, 'DIE': DIE, 'DII': DII, 'LEE': LEE, 'LIE': LIE, 'NmIY': NmIY, 'NmEY': NmEY,
              'DIY': DIY, 'DEY': DEY})
filename = str(time.strftime(ISOTIMEFORMAT)) + '_LI.mat'
scio.savemat(filename,
            {'xEbin_ra': xEbin_ra, 'xIbin_ra': xIbin_ra,'VEavgbin_ra':VEavgbin_ra,'VIavgbin_ra':VIavgbin_ra,
             'VEstdbin_ra':VEstdbin_ra,'VIstdbin_ra':VIstdbin_ra})
# filenamenp = str(time.strftime(ISOTIMEFORMAT)) + 'struct_LI.npz'
# np.savez(filenamenp, rEorg=rEorg, rEpre= rEpre, rEpos = rEpos, rIorg=rIorg, rIpre= rIpre, rIpos = rIpos,)
'''
'''