from connectiondistributioncollection import ConnectionDistributionCollection
import time
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import scipy.io as scio 


class Simulation(object):
    """
    Parameters:
    list :
        All sub-population (cluster)
        All connection (cluster)
        [type of both is 'List', which is changable variable, and could be changed]
        
    generate after initiate(by hand)
        connection_distribution
        connection_distribution_list
        [the differences between connection, connection_distribution and connection_distribution_list are
        connection ---> the component of 'connection_list', record all information and related information and object,like source and others
        connection_distribution --> this variable is a preparation variable for further processing, each 'connection' could generate a 
        class 'connecton_distribution' and then, using weight,syn,prob, could calculate flux_matrix and threshold
        each 'connection_distribution' item is defined by 'weight''syn ''prob', items with identical symbol will be classified to the same
        distribution
        connection_distribution_list --> this is a 'basket', store all unique connections(definition of unique: unique symbol
        'weight','syn','prob' no matter the target/source population)
    """
    def __init__(self,population_list,connection_list,Net_settings,Cell_type_num,DEE,DIE,DEI,DII,LEE,LIE,LEI,LII,tau_d,verbose=True):
        
        self.verbose = verbose
        self.population_list = population_list
        self.connection_list = [c for c in connection_list if c.nsyn!=0.0]
        self.Net_settings    = Net_settings
        tfinal = Net_settings['Final_time']
        dt     = Net_settings['dt']
        self.dt,self.tau_m = dt,20.0
        self.ntt = int(tfinal/dt)
        ''' >>>>>>>>>>>>>>>>> MFE >>>>>>>>>>>>>>>>>>>>>>> '''
        # >>>>>>>>>>>>>>> Re-sampling Voltage Distribution >>>>>>>>>>
        self.VE,self.VI = None,None
        self.Vedges,self.Vbins = None,None
        self.NE,self.NI = Cell_type_num['e'],Cell_type_num['i']
        self.MFE_num  = 0
        self.MFE_flag = 0
        # >>>>>>>>>>>>>> Inherit Instant Synaptic Input from Main File, Calculating Instant Jump >>>>>>>>>>>
        self.DEE,self.DIE,self.DEI,self.DII = None,None,None,None
        self.DEE = DEE.copy()
        self.DEI = DEI.copy()
        self.DIE = DIE.copy()
        self.DII = DII.copy()
        
        self.LEE = LEE.copy()
        self.LIE = LIE.copy()
        self.LEI = LEI.copy()
        self.LII = LII.copy()
        self.tau_d = tau_d        
        self.mfeflag = 1
    
    def initialize(self,t0=0.0):
        """
        The Most Important Issue in This File is, Both Populations and Connections Are within One Shared Plantform Called Simulation, Thus all Personal Variables and Settings Could be Effective Connectted and Have Easier Access to
        """
        self.iteration_max = self.ntt+100
        iteration_max = self.iteration_max  
        self.tbin_tmp = 0 # initial
        self.tbinsize = 1.0
        dtperbin = int(self.tbinsize/self.dt)
        self.dtperbin = dtperbin

        # >>>>>>>>>>> Different Time Scale >>>>>>>>>>>>>>
        # Record per dt, Record per dtbin
        iteration_bin = int(iteration_max/dtperbin)
        NPATCH = self.Net_settings['hyp_num']
        NE,NI  = self.NE,self.NI
        self.VE,self.VI = np.zeros((NE,NPATCH)),np.zeros((NI,NPATCH))
        # DTBIN_RECORD_FLAG
        self.tbin_ra = np.zeros((iteration_max,1))
        # each dt !!!
        '''
        # each dt
        self.mE_ra   = np.zeros((iteration_max,NPATCH))
        self.mI_ra   = np.zeros((iteration_max,NPATCH))
        self.NMDAE_ra = np.zeros((iteration_max,NPATCH))
        self.NMDAI_ra = np.zeros((iteration_max,NPATCH))
        '''
        # >>> >>> >>> Remain Each dt >>> >>> >>>
        self.P_MFE_ra = np.zeros((iteration_max,NPATCH))

        # each tbin
        self.mEbin_ra = np.zeros((iteration_bin,NPATCH))
        self.mIbin_ra = np.zeros_like(self.mEbin_ra)
        self.xEbin_ra = np.zeros_like(self.mEbin_ra)
        self.xIbin_ra = np.zeros_like(self.xEbin_ra)
        ''' long-range '''
        self.NMDAEbin_ra  = np.zeros_like(self.xEbin_ra)
        self.NMDAIbin_ra  = np.zeros_like(self.xIbin_ra)
        self.tau_NMDA     = np.zeros((2,1))
        
        ''' >>>>>>> Possible MFE and inhence Effective MFE >>>>>>>>'''
        # >>>>>>>>>> Possible MFE(probability value) and Index >>>>>>>>>>>
        self.P_MFEbin_ra = np.zeros_like(self.xIbin_ra)
        # >>>>>>>>>> Possible MFE Time (dt)>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.idx_MFE_ra  = np.zeros((iteration_max,1))
        # >>>>>>>>>>>>>> Effective/Successful MFE(VALUE), Index and Time >>>>>>>>
        self.P_MFE_eff   = np.zeros((iteration_max,1))
        # >>>>>>>>>>>>>> Index and Time >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.idx_MFE_eff = np.zeros((iteration_max,2))        
        #  >>>>>>>>>>>> Normal Voltage Distribution >>>>>>>>>>>>>>>>>>>
        self.rEbin_ra = np.zeros((NPATCH,2000,iteration_bin))
        self.rIbin_ra = np.zeros_like(self.rEbin_ra)
        # FOR DATA ANALYSIS
#        self.rEbinp_ra = np.zeros((NPATCH,2000,5000))
#        self.rIbinp_ra = np.zeros_like(self.rEbinp_ra)
        self.VEbinp_ra = np.zeros((NPATCH,652,12000))  # + time + LE/LI
        self.VIbinp_ra = np.zeros_like(self.VEbinp_ra)
        self.rhovc     = 0

        self.VEavgbin_ra = np.zeros_like(self.P_MFEbin_ra)
        self.VIavgbin_ra = np.zeros_like(self.VEavgbin_ra)
        self.VEstdbin_ra = np.zeros_like(self.VIavgbin_ra)
        self.VIstdbin_ra = np.zeros_like(self.VEstdbin_ra)

        self.VEmubin_ra = np.zeros_like(self.VIavgbin_ra)
        self.VImubin_ra = np.zeros_like(self.VEstdbin_ra)
        self.VEsigbin_ra = np.zeros_like(self.VIavgbin_ra)
        self.VIsigbin_ra = np.zeros_like(self.VEstdbin_ra)
        
        # each dt !!!
        '''
        self.VEavg_ra = np.zeros((iteration_max,NPATCH))
        self.VIavg_ra = np.zeros_like(self.VEavg_ra)
        self.VEstd_ra = np.zeros_like(self.VIavg_ra)
        self.VIstd_ra = np.zeros_like(self.VEstd_ra)
        '''
        # >>>>>>>>>>>>>>>> Instant Recording Rhov >>>>>>>>>>>>
        self.rE,self.rI  = None,None
        self.NPATCH = NPATCH        
        # why _ra for each dt
        self.LE_ra = np.zeros((iteration_max,NPATCH))
        self.LI_ra = np.zeros_like(self.LE_ra)

        # >>>>>>>>>>>>>>>>>>>> Extract Instant Synaptic Connections from self.structure, Prepare for MFE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        DEE,DIE,DEI,DII = self.DEE,self.DIE,self.DEI,self.DII
        
        vT = 1.0
        dv = self.Net_settings['dv']

        # >>>>>>>>>>>>>>> Vedges --> Node = (max-min)/dv + 1, Vbins --> Bin-central Value >>>>>>>>>
        self.Vedges = util.get_v_edges(-1.0,1.0,dv)
        self.Vbins = 0.5*(self.Vedges[0:-1] + self.Vedges[1:]) 
        Vedges = self.Vedges 
        Vbins  = self.Vbins
        idx_vT = len(Vedges)-1 # or len(Vbins)
        idx_kickE,idx_kickI = np.zeros((NPATCH,NPATCH),dtype=int),np.zeros((NPATCH,NPATCH),dtype=int)
        for it in range(self.NPATCH):
            for js in range(self.NPATCH):
                value_kickE = vT - DEE[it,js]
                value_kickI = vT - DIE[it,js]
                Ind_k1  = np.where(Vedges>value_kickE)
                IndI_k1 = np.where(Vedges>value_kickI) 
                if np.shape(Ind_k1)[1]>0:
                    idx_kickE[it,js]  = Ind_k1[0][0]
                else:
                    idx_kickE[it,js]  = idx_vT
                if np.shape(IndI_k1)[1]>0:
                    idx_kickI[it,js]  = IndI_k1[0][0]
                else:
                    idx_kickI[it,js]  = idx_vT
        
        self.idx_kickE,self.idx_kickI,self.idx_vT = idx_kickE,idx_kickI,idx_vT
        self.MFE_pevent = np.zeros(self.NPATCH)
        self.p_single   = np.zeros(self.NPATCH)
        self.rE = np.zeros((len(self.Vbins),self.NPATCH))
        self.rI = np.zeros_like(self.rE)
        print('kick!>>',idx_kickE)
        print('kick!>>',idx_kickI)
        
        # >>> An connection_distribution_list (store unique connection(defined by weight,syn,prob)) >>>
        self.connection_distribution_collection = ConnectionDistributionCollection() # this is 
        self.t = t0   # zero

        # >>>>> Number of Internal Populations, Excitatory and Inhibitory Populations Included >>>>
        numCGPatch = self.Net_settings['nmax'] * 2 
        '''
        # >>>>>>>>>>>>>>>>> The Most Improtant Issue, Populations as well as Connections Should be 
        '''
        for subpop in self.population_list:
            subpop.simulation = self    # .simulation = self(self is what we called 'simulation')
        for connpair in self.connection_list:
            connpair.simulation = self
        ''' simulation is a platform, both populations and connections could have access to it '''
            
        # >>>>>>>>>>>>> Preparation Finished, Initialize Populations and Connections >>>>>>       
        for p in self.population_list:
            p.initialize()      # 2   
        
        for c in self.connection_list:
            c.initialize()      # 1

    def update(self,t0,dt,tf):

        # >>>>>>>>>>>>>>>>>>> Initialize >>>>>>>>>>>>>>>>>>>>>>>>>>
        self.dt = dt
        self.tf = tf   
        self.initialize(t0)

        # >>>>>>>>>>>>>>>>> Start Iteration >>>>>>>>>>>>>>>>>>>>
        start_time = time.time()
        self.initialize_time_period = time.time()-start_time
        counter = 0
        # >>>>>>>>>>>>>>>>> Basic Network Structure >>>>>>>>>>>>
        Vbins,Vedges,NPATCH = self.Vbins,self.Vedges,self.NPATCH
        numCGPatch = self.Net_settings['nmax']*2
        print('Number of both Excitatory and Inhibitory populations(double of NPATCH): ',self.Net_settings['nmax']*2)
        LEE,LIE,LEI,LII = self.LEE,self.LIE,self.LEI,self.LII

        while self.t < self.tf:
            # >>>>>>>>>>>>>>>> Time Evolution >>>>>>>>>>>>>>>>>
            self.t += self.dt
            # >>>>>>>>>>>>>>>>> tbin_tmp for Recording >>>>>>>>>>>
            self.tbin_tmp = int(np.floor(self.t/self.tbinsize))
            # >>>>>>>>>>>>>>>>> If We Want Change Connection-strength(period of time), Comment Code below Could be Appropriate
            '''
            for c in self.connection_list:
                if (self.t > 100.0) & (flagS <4):                    
                    print('current time',self.t)
                    if (c.conn_type == 'LongRange') & (c.ei_pop_post == 'e'):
                        flagS += 1
                        c.weights /= 1.2
                        c.weights *= 2.66#2.85#2.6#2.1#1.8
                        print('change e to e Long range, flag',flagS,' value:',c.weights)
                    if (c.conn_type == 'LongRange') & (c.ei_pop_post == 'i'):
                        flagS += 1
                        c.weights /= 1.2
                        c.weights *= 1.2#3.60#3.6
                        print('change e to i Long range, flag',flagS,' value:',c.weights)
                if (self.t > 100.0) & (flagE <4):                    
                    print('current time',self.t)
                    if (c.conn_type == 'LongRange') & (c.ei_pop_post == 'e'):
                        flagE += 1
                        c.weights /= 2.66#2.85
                        c.weights *= 1.20#2.6#2.1#1.8
                        print('change e to e Long range, flag',flagE,' value:',c.weights)
                    if (c.conn_type == 'LongRange') & (c.ei_pop_post == 'i'):
                        flagE += 1
                        c.weights /= 1.2
                        c.weights *= 1.2#3.60#3.6
                        print('change e to i Long range, flag',flagE,' value:',c.weights)
            '''
            ind_rec,idxE,idxI = 0,0,0  
            for p in self.population_list:
                p.USUALorMFE = 1
                ind_rec += 1
                '''
                Recording at first, before p.update(),
                rE and rI purely from(after) MFE should be recorded in rE/I(bin)_ra, rather
                than RvE from Moment
                '''
                # before Moment iteration
                if(ind_rec>numCGPatch): # means internal-population, not external-population
                    if p.ei_pop == 'e': 
                        # >>>>>> Per dtbin >>>>>>>>>>
                        self.VEavgbin_ra[self.tbin_tmp,idxE] += p.v1*dt
                        self.VEstdbin_ra[self.tbin_tmp,idxE] += np.sqrt(p.v2-p.v1**2)*dt
                        self.VEmubin_ra[self.tbin_tmp,idxE]  += p.total_fp_vslave*dt
                        self.VEsigbin_ra[self.tbin_tmp,idxE] += p.total_fp_sigv*dt
#                        self.rEbinp_ra[idxE,:,rhovc] = p.curr_rhov
                        
                        idxE +=1
                    else:
                        # >>>>>> Per dt >>>>>>>>>>>>>
                        ''' 
                        self.VIavg_ra[counter,idxI] = p.v1
                        self.VIstd_ra[counter,idxI] = np.sqrt(p.v2-p.v1**2)
                        '''
                        # >>>>>> Per dtbin >>>>>>>>>>
                        self.VIavgbin_ra[self.tbin_tmp,idxI] += p.v1*dt
                        self.VIstdbin_ra[self.tbin_tmp,idxI] += np.sqrt(p.v2-p.v1**2)*dt
                        self.VImubin_ra[self.tbin_tmp,idxI] += p.total_fp_vslave*dt
                        self.VIsigbin_ra[self.tbin_tmp,idxI] += p.total_fp_sigv*dt
#                        self.rIbinp_ra[idxI,:,rhovc] = p.curr_rhov
                        
                        idxI +=1
                        
                p.update()
            # add in 02/04/2020 refresh c.curr_Delay_firing rate, for p.firing_rate = 0.0
            for c in self.connection_list:
                c.update()                   
                '''
                when using USUALorMFE==1
                updating rhov as well as firing rate
                
                next, should record firing rate mE/I in mE/I(bin)_ra
                [but not rE/I(bin)_ra]
                
                and also, RvE/I were extracted out from p-list, which were used
                to calculate MFE probability                   
                
                '''
            ind_rec,idxE,idxI = 0,0,0                
            for p in self.population_list:
                ind_rec += 1
                if(ind_rec>numCGPatch):
                    if p.ei_pop == 'e':
                        # >>>>>> Only Excitatory Population Could Trigger MFE
                        self.rE[:,idxE] = p.curr_rhov
                        # >>>>>>>>> Per dt >>>>>>>>>>>>
                        ''' 
                        self.mE_ra[counter,idxE]    = p.curr_firing_rate
                        self.NMDAE_ra[counter,idxE] = p.acm_NMDA
                        '''
                        self.p_single[idxE]         = p.curr_firing_rate * self.dt * p.NumCell
                        idxE += 1
                    else:
                        # >>>>>>>>> Per dt >>>>>>>>>>>>
                        ''' 
                        self.mI_ra[counter,idxI]    = p.curr_firing_rate    
                        self.NMDAI_ra[counter,idxI] = p.acm_NMDA 
                        '''
                        self.rI[:,idxI] = p.curr_rhov    
                        idxI += 1
                
            # >>>>>>>>>>>>>>>>>>>>>>>>> Probability to Generate MFE >>>>>>>>>>>>>>.
            NE,NI = self.NE,self.NI
            NPATCH = self.NPATCH
            h = self.Vedges[1]- self.Vedges[0]
            local_pevent = 1.0            
            for isource in range(self.NPATCH):
                local_pevent = 1.0
                for jt in range(self.NPATCH):
                    kickE,kickI = self.idx_kickE[jt,isource],self.idx_kickI[jt,isource]
                    idx_vT   = self.idx_vT
                    ''' >>>>>>>>>>>> Excitatory Target Populations to Be Triggered >>>> '''
                    trhov    = np.squeeze(self.rE[:,jt])
                    if isource!=jt:
                        Nup = NE 
                    else:
                        Nup = NE                    
                    prob_event = (1.0 - np.sum(np.squeeze(trhov[kickE:idx_vT])) * h) ** Nup 
                    local_pevent *= prob_event    
                    ''' >>>>>>>>>>>> Inhibitory Target Populations to Be Triggered >>>> '''
                    trhov    = np.squeeze(self.rI[:,jt])
                    Nup = NI
                    prob_event = (1.0 - np.sum(np.squeeze(trhov[kickI:idx_vT])) * h) ** Nup 
                    local_pevent *= prob_event
                self.MFE_pevent[isource] = self.p_single[isource] * (1-local_pevent)  
                
            ''' >>>>> Prob MFE and Real MFE >>>>>>>>>>>> '''
            self.P_MFE_ra[counter,:] = self.MFE_pevent[:]
            if self.mfeflag == 1:
                MFE_pevent_max = max(self.MFE_pevent)   # choose the maximum-value from all possible 
            else:
                MFE_pevent_max = max(self.MFE_pevent)
            # >>>>>>>>> Per dt >>>>>>>>>>>>
            self.P_MFEbin_ra[self.tbin_tmp,0] +=  MFE_pevent_max * dt
            idx_pevent_max = np.argwhere(self.MFE_pevent == np.amax(self.MFE_pevent))#np.argmax(self.MFE_pevent)
            # print('idx_',idx_pevent_max)
            if len(idx_pevent_max)>1:
                np.random.shuffle(np.squeeze(idx_pevent_max))
            idx_pevent_max = idx_pevent_max[0]
            self.idx_MFE_ra[counter] = idx_pevent_max
            self.MFE_flag = 0
            local_pevent_d = np.random.random()
#            print('>>>>>>>>>>>>>>>>>check MFE:%.5f '% self.MFE_pevent[isource])
            if local_pevent_d < MFE_pevent_max:#-1:#
                # >>>>>>>>>>>>>>> Recording N-serie
                self.MFE_flag = 1
                self.MFE_num += 1
                self.idx_MFE_eff[self.MFE_num,0] = idx_pevent_max
                self.idx_MFE_eff[self.MFE_num,1] = self.t
                self.P_MFE_eff[self.MFE_num]     = MFE_pevent_max
                # >>>>>> Voltage Distribution Sampling >>>>>>>>>>
                VE = np.zeros((self.NE,self.NPATCH))
                VI = np.zeros((self.NI,self.NPATCH))
                for i in range(self.NPATCH):
                    V_sample = util.getsamples(np.squeeze(self.Vedges[1:]),np.squeeze(self.rE[:,i]),self.NE)
                    VE[:,i]  = np.squeeze(V_sample)
                    V_sample = util.getsamples(np.squeeze(self.Vedges[1:]),np.squeeze(self.rI[:,i]),self.NI)
                    VI[:,i]  = np.squeeze(V_sample)

                # >>>>>> 2 Neurons Triggered >>>>>>>>>>>>>>>>>>>>>>>
                VE[0:2,idx_pevent_max] = 1.0
                DEEu = self.DEE.copy()
                DIEu = self.DIE.copy()
                DEIu = self.DEI.copy()
                DIIu = self.DII.copy()
                
                pop_idx_E = np.zeros((self.NE,self.NPATCH),dtype=int)
                pop_idx_I = np.zeros((self.NI,self.NPATCH),dtype=int)
                for i in range(self.NPATCH):
                    pop_idx_E[:,i] = i
                    pop_idx_I[:,i] = i
                    
                (E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos) = util.getMFE_ifdyn(0,VE,VI,DEEu,DEIu,DIEu,DIIu,0,0,pop_idx_E,pop_idx_I)
                self.mfeflag = 0
                '''
                # wrong E_fired_num arrangement!!!!!!!
                
                E_fired_num,I_fired_num = np.zeros((NE*NPATCH,1)),np.zeros((NI*NPATCH,1))
                E_fired_num[E_fired] = 1
                I_fired_num[I_fired] = 1
                E_fired_num = np.reshape(E_fired_num,(NE,NPATCH))
                I_fired_num = np.reshape(I_fired_num,(NI,NPATCH))
                E_ind_num = np.sum(E_fired_num,axis = 0)
                I_ind_num = np.sum(I_fired_num,axis = 0) 
                '''
             
                E_fired_num,I_fired_num = np.zeros((NE*NPATCH,1)),np.zeros((NI*NPATCH,1))
                E_fired_num[E_fired] = 1
                I_fired_num[I_fired] = 1
                E_fired_num = np.reshape(E_fired_num,(NPATCH,NE))
                E_fired_num = E_fired_num.T
                I_fired_num = np.reshape(I_fired_num,(NPATCH,NI))
                I_fired_num = I_fired_num.T
                E_ind_num = np.sum(E_fired_num,axis = 0)
                self.VEbinp_ra[:,-1,self.rhovc] = E_ind_num
                I_ind_num = np.sum(I_fired_num,axis = 0) 
                self.VIbinp_ra[:,-1,self.rhovc] = I_ind_num
                self.rhovc += 1 
           
                for i in range(NPATCH):
                    self.LE_ra[self.MFE_num,i] = E_ind_num[i]
                    print('E-fired: ',E_ind_num[i])
                    self.LI_ra[self.MFE_num,i] = I_ind_num[i]
                VEpos,VIpos,VEpre,VIpre = VE_pos.copy(),VI_pos.copy(),VE_pre.copy(),VI_pre.copy()
                Vedges = self.Vedges
                Vbins  = self.Vbins
                h = Vbins[2]-Vbins[1]
                rE,rI = np.zeros((len(Vbins),NPATCH)),np.zeros((len(Vbins),NPATCH))

                # >>>>>>>>>>>>>>> Re-sample Rhov after MFE >>>>>>>>>>>>
                for i in range(NPATCH):
                    VEposu = np.squeeze(VEpos[:,i])
                    rE_tmp,Vedge = np.histogram(VEposu, Vedges)
                    # >>>>>>>>>> New Rhov-E >>>>>>>>>>>>>>>
                    rE[:,i] = rE_tmp/(self.NE * h)
                    self.rE[:,i] = rE[:,i]
                    
                    VIposu = np.squeeze(VIpos[:,i])
                    rI_tmp,Vedge = np.histogram(VIposu, Vedges)
                    # >>>>>>>>>> New Rhov-I >>>>>>>>>>>>>>>>
                    rI[:,i] = rI_tmp/(self.NI *h) 
                    self.rI[:,i] = rI[:,i]

                # Refresh Network and Populations' Characters after MFE-event >>>>>
                # 1st Refresh Fring-rate,NMDA,HNMDA,Rhov
                idxE,idxI = 0,0
                ind_rec   = 0
                for p in self.population_list:
                    ind_rec +=1
                    if(ind_rec>numCGPatch):
                        if p.ei_pop == 'e':
                            p.rhov[:] = rE[:,idxE]
                            extract_NMDA = 0.0
                            for inmda in range(NPATCH):
                                if (inmda == idxE):
                                    extract_NMDA = extract_NMDA
                                else:
                                    extract_NMDA += E_ind_num[inmda] * LEE[idxE,inmda]
                                    extract_NMDA -= I_ind_num[inmda] * LEI[idxE,inmda]
                            p.acm_NMDA = p.acm_NMDA  + extract_NMDA*self.tau_m/self.tau_d
                            idxE += 1
                        else:
                            ''' rhovI[i]=0'''
                            p.rhov[:] = rI[:,idxI]
                            # ACM_NMDA 
                            extract_NMDA = 0.0
                            for inmda in range(NPATCH):
                                if (inmda == idxI):
                                    extract_NMDA = extract_NMDA
                                else:
                                    extract_NMDA += E_ind_num[inmda] * LIE[idxI,inmda]
                                    extract_NMDA -= I_ind_num[inmda] * LII[idxI,inmda]
                            p.acm_NMDA  = p.acm_NMDA  + extract_NMDA*self.tau_m/self.tau_d
                            idxI += 1

                # 2nd, Refresh Moments
                V1 = Vbins
                V2 = V1*Vbins
                V3 = V2*Vbins
                V4 = V3*Vbins   
                h = Vbins[2] - Vbins[1]
                vbarE = np.zeros(NPATCH)
                wbarE,vbar3E,vbar4E = np.zeros_like(vbarE),np.zeros_like(vbarE),np.zeros_like(vbarE)
                vbarI = np.zeros(NPATCH)
                wbarI,vbar3I,vbar4I = np.zeros_like(vbarI),np.zeros_like(vbarI),np.zeros_like(vbarI)
                for i in range(NPATCH):
                    rE_tmp = np.squeeze(rE[:,i])
                    rI_tmp = np.squeeze(rI[:,i])
                    vbarE[i] = np.sum(V1*rE_tmp ) * h
                    wbarE[i] = np.sum(V2*rE_tmp ) * h
                    vbar3E[i] = np.sum(V3*rE_tmp ) * h
                    vbar4E[i] = np.sum(V4*rE_tmp ) * h
                    
                    vbarI[i] = np.sum(V1*rI_tmp ) * h
                    wbarI[i] = np.sum(V2*rI_tmp ) * h
                    vbar3I[i] = np.sum(V3*rI_tmp ) * h
                    vbar4I[i] = np.sum(V4*rI_tmp ) * h


                idxE,idxI,ind_rec = 0,0,0
                '''
                set v1,v2,v3,v4 from outter
                USUALorMFE == 0,only change
                    VE/Is,DE/I (total.....), notice that before update, mE/I = 0,
                    p.firing_rate = 0
                    also,La0/1 equal to zero
                    
                    recalculate VE/Is DE/I
                    rhov_EQ(cause VE/Is and DE/I change)
                    
                    do not change rhov E/I
                    rhov have resampled after MFE, but didn't give it to p.rhov
                    so, give it to p-list, and do not change anymore (USUALorMFE==0)
                    firing rate keep 0 (USUALorMFE)                    
                    
                '''
                for p in self.population_list:
                    ind_rec +=1
                    if(ind_rec>numCGPatch):
                        if p.ei_pop == 'e':
                            ''' 
                            # >>>> Previous Info for Testing >>>
                            '''
#                                print('before MFE, Exc cell: ')
#                                print('firing rate: ',p.curr_firing_rate)
#                                print('moment: v1 %.5f'%(p.v1))
#                                print('Lag-parame: ',p.La0)
                            ''' >>>> Firing-rate was Forced to Be Zero >>>'''
                            p.firing_rate = 0.0
#                                ''' ALL RESET '''
#                                p.acm_HNMDA   = 0.0
#                                p.acm_NMDA    = 0.0
                            ''' v1,v2,v3,v4 '''
                            p.v1,p.v2,p.v3,p.v4 = vbarE[idxE],wbarE[idxE],vbar3E[idxE],vbar4E[idxE]
                            ''' >>>> lambda was Forced to Be Zero >>>'''
                            md_La = np.transpose([1,p.v1,p.v2])
                            p.La0,p.La1 = np.zeros_like(md_La),np.zeros_like(md_La)
                            idxE += 1
                        else:
                            ''' >>>> Firing-rate was Forced to Be Zero >>>'''
                            p.firing_rate = 0.0
                            p.v1,p.v2,p.v3,p.v4 = vbarI[idxI],wbarI[idxI],vbar3I[idxI],vbar4I[idxI]
                            ''' >>>> lambda was Forced to Be Zero >>>'''
                            md_La = np.transpose([1,p.v1,p.v2])
                            p.La0,p.La1 = np.zeros_like(md_La),np.zeros_like(md_La)
                            idxI += 1
                # add in 02/04/2020 refresh c.curr_Delay_firing rate, for p.firing_rate = 0.0
                for c in self.connection_list:
                    c.update()                        
                # >>>>>>>>>>>>>>>>> Refresh,But Not Recalculate Firing-rate >>>>>>>
                ind_rec   = 0
                for p in self.population_list:
                    ind_rec +=1
                    if(ind_rec>numCGPatch):
                        p.USUALorMFE = 0
                        p.update()
                '''
                # >> This Refresh(Re-update) is necessary, we shouldn't directly use VEpos/VIpos as rhov, because in VEpos/VIpos, We Only Consider Instant Synaptic Inputs, afterhence, We Refresh HNMDA and NMDA, So Refresh RhovEq and Vs Ds is necessary >>
                '''

            for c in self.connection_list:
                c.update()
            counter +=1
            
            # >>>>>>>>>>>>>>>> Recording >>>>>>>>>>>>>>>>>>>>>>>>>>>.
            self.tbin_ra[counter] = np.floor(self.t/self.tbinsize)
            tbin = int(np.floor(self.t/self.tbinsize))
            ind_rec,idxE,idxI   = 0,0,0
            for p in self.population_list:
                ind_rec +=1
                if(counter>-1):
                    if(ind_rec>numCGPatch):
                        if p.ei_pop == 'e':
                            self.tau_NMDA[0,0] = p.tau_r
                            self.tau_NMDA[1,0] = p.tau_d
                            self.mEbin_ra[tbin,idxE] += (1-self.MFE_flag) * p.curr_firing_rate * NE * dt + self.MFE_flag * self.LE_ra[self.MFE_num,idxE]
                            self.xEbin_ra[tbin,idxE] += (1-self.MFE_flag) * util.psample(p.curr_firing_rate * NE * dt) + self.MFE_flag * self.LE_ra[self.MFE_num,idxE]
                            self.rEbin_ra[idxE,:,self.tbin_tmp] += (1-self.MFE_flag) * p.curr_rhov * dt + self.MFE_flag * self.rE[:,idxE] * dt
                            ''' Long range '''
                            self.NMDAEbin_ra[tbin,idxE] += p.acm_NMDA * NE * dt
                                                    
                            idxE += 1
                        else:
                            self.mIbin_ra[tbin,idxI] += (1-self.MFE_flag) * p.curr_firing_rate * NI * dt + self.MFE_flag * self.LI_ra[self.MFE_num,idxI]
                            self.xIbin_ra[tbin,idxI] += (1-self.MFE_flag) * util.psample(p.curr_firing_rate * NI * dt) + self.MFE_flag * self.LI_ra[self.MFE_num,idxI]
                            self.rIbin_ra[idxI,:,self.tbin_tmp] += (1-self.MFE_flag) * p.curr_rhov * dt + self.MFE_flag * self.rI[:,idxI] * dt
                            ''' Long range '''
                            self.NMDAIbin_ra[tbin,idxI] += p.acm_NMDA * NI * dt
                            
                            idxI += 1
                            
            ''' visualizing '''
            if np.mod(counter,500) < 1:
                if np.mod(counter,500) == 0:
                    print("t_sum: ",counter * self.dt)
                for i in range(NPATCH):
                    idshown = np.floor(tbin/2000)
                    idstart = np.int(idshown * 2000)
                    if idstart == tbin:
                        idstart = idstart -1
                        
                    idend   = min((idshown+1)*2000,4000)
                    print('Excitatory pop %d :%.4f'%(i,self.mEbin_ra[tbin,i]))
                    print('Inhibitory pop %d :%.4f'%(i,self.mIbin_ra[tbin,i]))
                    ttt = np.arange(idstart,tbin) * 1.0
                    plt.figure(10)
                    plt.subplot(NPATCH,1,int(i)+1)
                    plt.plot(ttt,self.mEbin_ra[idstart:tbin,i],'r')
                    plt.xlim([ttt[0],ttt[0]+2000])
                    plt.ylim([0,NE])
                    plt.pause(0.1)
                    plt.figure(11)
                    plt.subplot(NPATCH,1,int(i)+1)
                    plt.plot(ttt,self.mIbin_ra[idstart:tbin,i],'b')
                    plt.xlim([ttt[0],ttt[0]+2000])
                    # plt.xlim([0,int(self.tf)])
                    plt.ylim([0,NI])
                    plt.pause(0.1)
                    
#            if np.mod(counter,4999) == 0:
#                icounter    = np.ceil(counter/4999)
#                intic,ip,ic = int(icounter),int((icounter-1)*500),int((icounter)*500)
#                
#                timeformat = '%Y%m%d%H'           
#                filename=str(time.strftime(timeformat)) + str(intic) +'.mat'
#                scio.savemat(filename,{'mEbin_ra':self.mEbin_ra[ip:ic,:],'mIbin_ra':self.mIbin_ra[ip:ic,:],'xEbin_ra':self.xEbin_ra[ip:ic,:],'xIbin_ra':self.xIbin_ra[ip:ic,:],\
#                                       'VEavgbin_ra':self.VEavgbin_ra[ip:ic,:],'VIavgbin_ra':self.VIavgbin_ra[ip:ic,:],'VEstdbin_ra':self.VEstdbin_ra[ip:ic,:],'VIstdbin_ra':self.VIstdbin_ra[ip:ic,:],\
#                                       'rEbin_ra':self.rEbin_ra[:,:,ip:ic],'rIbin_ra':self.rIbin_ra[:,:,ip:ic],'P_MFEbin_ra':self.P_MFEbin_ra[ip:ic,:]}) 
#                # filelrname=str(time.strftime(timeformat)) + str(intic) + '_LR.mat'
#                # scio.savemat(filelrname, {'NMDAEbin_ra':self.NMDAEbin_ra[ip:ic,:],'NMDAIbin_ra':self.NMDAIbin_ra[ip:ic,:],\
#                #                           'HNMDAEbin_ra':self.HNMDAEbin_ra[ip:ic,:],'HNMDAI_ra':self.HNMDAIbin_ra[ip:ic,:],\
#                #                           'tau_NMDA':self.tau_NMDA})
#            if np.mod(counter,10000)==0:
#                indexc = np.floor(counter/10000)
#                timeformat = '%Y%m%d%H'           
#                filename= 'rhovp.mat'
#                scio.savemat(filename, {'rEbinp_ra': self.rEbinp_ra, 'rIbinp_ra': self.rIbinp_ra,'VEbinp_ra':self.VEbinp_ra,'VIbinp_ra':self.VIbinp_ra})
#
#                           
        return self.mEbin_ra,self.mIbin_ra,self.rEbin_ra,self.rIbin_ra,self.P_MFEbin_ra,self.NMDAEbin_ra,self.NMDAIbin_ra,self.VEavgbin_ra,self.VIavgbin_ra,self.VEstdbin_ra,self.VIstdbin_ra,\
        self.VEmubin_ra,self.VImubin_ra,self.VEsigbin_ra,self.VIsigbin_ra