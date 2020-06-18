"""
Module contains maybe the most important part in our Moments Algorithm
Internal populations and their structures and dynamics!
This module combine utilities function, and describe dynamics of recurrent(internal) populations

This version is used for Pytorch and have been test on Dell
Last modified 2018/04/06
"""

import numpy as np
import utilities as util
from scipy.optimize import minimize



"""
09/08/2018 version
edited by SYX

internal population, neurons within primary visual cortex, which could receive recurrent synaptic inputs from 
other neurons/internal-populations in primary visual cortex, which we called recurrent connections for internal neuron population
"""   
class RecurrentPopulation(object):
    """
    Parameters:
    v_min: minimum voltage(default = -1.0)
    v_th/max: maximum/threshold 
    dv   : voltage domain discritization 
    record: flag(True/False)
    curr_firing_rate: firing rate of the corresponding recurrent population
    update_mode: str'approx' or 'exact'(default=')

    ADDING :
    hyp_idx : hyper-column index for this moment
    ei_pop  : excitatory or inhibitory population
    NumCell : number of cells in this population
    
    """
    
    def __init__(self,tau_m = 20.0,dt = 0.1,v_min = -1.0,v_max = 1.0,dv = 1e-3,record = True,
                firing_rate = 0.0,update_method = 'exact',approx_order = None,tol = 1e-12,norm = np.inf,
                hyp_idx = 0,ei_pop = 'e', NumCell = 0, **kwargs):
        # >>>>>>>>>>>>>>>>> transmit parameters >>>>>>>>>>>>>>>>>
        self.dt    = dt
        (self.v_min,self.v_max) = (v_min,v_max)
        self.dv = dv
        self.record = record
        self.firing_rate = 0.0 # firing_rate
        self.update_method = update_method
        self.approx_order = approx_order
        self.tol = tol
        self.norm = norm
        
        self.type = 'Recurrent'
        self.hyp_idx = hyp_idx
        self.ei_pop  = ei_pop
        self.tau_m = tau_m
#        if self.ei_pop == 'e':
#            self.tau_m = 20.0
#        else:
#            self.tau_m = 10.0
        self.NumCell = NumCell
        self.MFE_num = 0
        self.USUALorMFE = 1
        # additional parameters
        self.metadata = kwargs
        
        # >>>>>>>>>>>>> Resetting Voltage Bins and Voltage Edges, Using None, Further Initiation Are Needed >>>>>>>>>>
        self.edges = None
        self.rhov = np.zeros([20000])
        self.firing_rate_record = None # used for recording corresponding spike train
        self.local_pevent = 0.0
        self.t_record = None # time series
        
        #  >>>>>>>>>>>>>>>>>>>>>>>>> Simulation Platform >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.simulation = None

        # if we use active release NMDA synaptic signal
        # once the sender(pre) population generated firing rate
        # it had ability to automatically release NMDA-type slow conductance
        # so it naturally has this property(without self.weights)

        # >>> There Are Two Different Method to Calculate NMDA And Hnmda, Here Is the First One, I Called It'Active NMDA'.
        # Here, Once This Population Emitted a Spike，the Capability to Give LR-input Would Increase thus self.hnmda changed, later on self.inmda changed
        # But This Method Has a Disadvantage, If the Connectivity-strength Changed, Post-population Integrate Corresponding LR-inputs, Will Change Instantanously. >>
        self.hnmda,self.inmda = 0.0,0.0
        # >>> Here is the Second Method, Here I Use 'Passive NMDA', Passive Means that We Integrate Firing-rate of Pre-population as Normal (Weight and Nsyn), Rather Than Inmda and Hnmda,
        # And We Convolve Alpha Function in Post-population, Thus There Could be a Buffer for Weight-changing
        self.acm_NMDA, self.acm_HNMDA = 0,0

        self.v1 = 0.0
        self.v2 = 0.0
        self.v3 = 0.0
        self.v4 = 0.0

        self.La0 = None
        self.fin = None
        
        self.total_fp_vslave = 0.0
        self.total_fp_sigv   = 0.0

        # >>>>>>>>>>>>> Time Constants for Long-range Connections >>>>>>>>>>>>>
        self.tau_r = 2.0
        self.tau_d = 128.0
        
    def initialize(self):
        """
        initialize some specific parameters and variables by hand
        with 
            1)voltage-edge/bin
            2)connection dictionary
            3)all about recorder
        """
        self.initialize_edges()
        self.initialize_prob()
        self.initialize_fpmusigv_dict()
        self.initialize_MFE()

        
    """
    Code below is designed for some basic matrix or elements which might be initialized
    at the beginning and maintained unchanged during the whole data analysis, but if U 
    need update some connections or strurtures, U could still start the 'Update' function 
    to regenerate a new structure(connections)
    """   
    def initialize_edges(self):
        # >>> Initialize discreted voltage bins
        self.edges = util.get_v_edges(self.v_min,self.v_max,self.dv)
    def initialize_fpmusigv_dict(self):
        self.total_fpmu_dict  = {}
        self.total_fpsig_dict = {}
        self.total_Inmda_dict = {}
        # >>> 1st mean-value (mu), 2nd std-value (sig), 3rd LR-input (Inmda)
        for c in self.source_connection_list:
            if (c.conn_type =='ShortRange'):
                # >>>try & except structure, avoid NaN and Inf >>>                
                try:
                    curr_mu = self.total_fpmu_dict.setdefault(c.connection_distribution,0)
                    curr_sigv = self.total_fpsig_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                    
                    curr_mu = self.total_fpmu_dict.setdefault(c.connection_distribution,0)
                    curr_sigv = self.total_fpsig_dict.setdefault(c.connection_distribution,0)
                self.total_fpmu_dict[c.connection_distribution] = curr_mu + c.curr_delayed_firing_rate * c.nsyn * c.weights # c.curr_firing_rate * c.nsyn * c.weights
                self.total_fpsig_dict[c.connection_distribution] = curr_sigv + c.curr_delayed_firing_rate * c.nsyn * (c.weights**2) # c.curr_firing_rate * c.nsyn * (c.weights**2)
            else:
                try:
                    curr_nmda_i = self.total_fpmu_dict.setdefault(c.connection_distribution,0)
                except:
                    c.initialize_connection_distribution()
                self.total_Inmda_dict[c.connection_distribution] = 0.0 + c.curr_Inmda * c.nsyn * c.weights
                # >>> No contribution to sigma(standard deviation)
            
        # >>>>>>>>>>>>>>>>>>>>>> Summation >>>>>>>>>>>>>>>>>>>>>>>
        self.total_fp_vslave = 0.0
        self.total_Inmda     = 0.0
        for key,val in self.total_fpmu_dict.items():            
            try:
                self.total_fp_vslave += val
            except: 
                key.initialize()              
                self.total_fp_vslave += val
        self.total_fp_vslave  = self.total_fp_vslave * self.tau_m
        # >>>>>>>>>>>>>>>>>>>> Notice /gL >>>>>>>>>>>>>>>>>>>>>>
        # >>> Inmda <-- Hnmda <-- m(firing-rate), All these attribute to firing-rate, similar to vslave, multipling self.tau_m >>>
        for key,val in self.total_Inmda_dict.items():
            try:
                self.total_Inmda += val
            except:
                key.initialize()
                self.total_Inmda += val
        self.total_Inmda = self.total_Inmda * self.tau_m              
        # >>>>>>>>>>>>>>>>>>>>>> Summation >>>>>>>>>>>>>>>>>>>>>>>
        self.total_fp_sigv = 0.0
        for key,val in self.total_fpsig_dict.items():
            try:
                self.total_fp_sigv += val
            except:
                key.initialize()
                self.total_fp_sigv += val
        self.total_fp_sigv  = self.total_fp_sigv * self.tau_m
    def initialize_prob(self):
        # >>> Voltage distribution, length of rhov equals to length of Voltage-bin
        self.rhov = np.zeros_like(self.edges[:-1])
        # >>> Transfer voltage-edge into voltage-bin >>>
        vedges = self.edges
        h = vedges[2]-vedges[1]
        vedges = 0.5*(vedges[0:-1] + vedges[1:])
        
        var1   = np.power((5/3.0/250.0),2)
        source = np.exp(-np.square(vedges-0.0)/var1/2.0)/np.sqrt(2.0*np.pi*var1)
        source = source/(h*np.sum(np.squeeze(source)))
        
        self.rhov = source
        rhovs = self.rhov
        # >>> Calculate Voltage-moment fourth-order >>>
        v1 = h*np.sum(vedges*rhovs)
        v2 = h*np.sum(np.square(vedges)*rhovs)
        v3 = h*np.sum(np.power(vedges,3.0)*rhovs)
        v4 = h*np.sum(np.power(vedges,4.0)*rhovs)

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4
        gamma   = [1,v1,v2,v3,v4]
        fi      = np.transpose(gamma)
        F       = fi[1:3]
        La0     = np.transpose(gamma[0:3])
        N       = 3#len(La0)  #3
        fin     = np.zeros((len(vedges),N))
        fin[:,0] = np.ones_like(fin[:,0])
        for n in range(1,N):
            fin[:,n] = vedges*fin[:,n-1]
        # >>> Saving lambda >>>
        self.La0 = La0
        self.fin = fin
        # >>> Saving initial parameters & variables for RESETTING >>> 
        self.source    = source 
        self.initLa0   = La0
        self.initgamma = gamma

    def initialize_MFE(self):
        # >>> Transfer Voltage-edge into Voltage-bn >>>
        vedges = self.edges
        h      = vedges[2] - vedges[1]
        vedges = 0.5 * (vedges[0:-1] + vedges[1:])
        rhov   = self.rhov
        NumCell = self.NumCell
        self.V_sample = util.getsamples(vedges,rhov,NumCell)
                
        local_pevent = 1.0
        NE,NI = 0,0
        if self.ei_pop =='e':
            for c in self.target_connection_list:
                if (c.conn_type =='ShortRange'):
                    # if already initialize connection_distribution or not
                    try:
                        idx_kick = c.idx_kick
                        idx_vT   = c.idx_vT
                        trhov    = c.trhov
                        nsyn_post = c.nsyn_post
                        prob_event = (1.0 - np.sum(np.squeeze(trhov[idx_kick:idx_vT])) * h) ** nsyn_post 
                        local_pevent *= prob_event
                        
                    except:
                        c.initialize_connection_distribution()
                        
                        idx_kick = c.idx_kick
                        idx_vT   = c.idx_vT
                        trhov    = c.trhov
                        nsyn_post = c.nsyn_post
                        prob_event = (1.0 - np.sum(trhov[idx_kick:idx_vT]) * h) ** nsyn_post 
                        local_pevent *= prob_event
                        
                else:
                    local_pevent = 1.0
            NE = self.NumCell
            local_pevent = NE *  self.curr_firing_rate * (1-local_pevent)  
            self.local_pevent = local_pevent
        else:
            self.local_pevent = local_pevent

    # >>> Method 1, Update capability to release HNMDA and NMDA >>>>>>>>
    def update_NMDA_midvar_syncurr(self):
        ownfr = self.curr_firing_rate
        # parameters
        deltat = self.dt
        trise  = self.tau_r
        tdamp  = self.tau_d

        tr   = deltat/trise
        etr  = np.exp(-tr)
        td   = deltat/tdamp
        etd  = np.exp(-td)
        cst  = 1.0/(tdamp - trise)*(etd - etr) # trise/(tdamp - trise)*(etd - etr)

        self.inmda = self.inmda * etd + self.hnmda * cst
        self.hnmda = self.hnmda * etr + ownfr #* self.dt  * 1.0 # * trise

    def update_total_fpmu_dict(self):
        # >>> Reset to zero, both vslave(instant firing rate) and long-range NMDA >>>
        for c in self.source_connection_list:
            self.total_fpmu_dict[c.connection_distribution] = 0.0
            if (c.conn_type == 'LongRange'):
                self.total_Inmda_dict[c.connection_distribution] = 0.0
                # >>> Refreshing >>>
            """
            no matter short-/long-range connections should at first reset to zeros~~~
            in case not to be distubed by previous result
            """
        # >>> Done! Clear up all synaptic inputs >>>        
        # >>> Time and other important parameters >>>
        deltat = self.dt
        tdamp  = self.tau_d
        td   = deltat/tdamp
        etd  = np.exp(-td)
        # >>> Extract integrated HNMDA and NMDA >>> 
        accumulated_NMDA  = self.acm_NMDA
        tt_NMDA           = 0
        for c in self.source_connection_list:  
            if(c.conn_type == 'ShortRange'):
                self.total_fpmu_dict[c.connection_distribution]  += c.curr_delayed_firing_rate * c.nsyn * c.weights
            else:
                self.total_Inmda_dict[c.connection_distribution] += c.curr_Inmda * c.nsyn * c.weights
                tt_NMDA += c.curr_delayed_firing_rate * c.nsyn * c.weights 
                
        # >>> Accumulated HNMDA & NMDA, multipling tau_m >>>
        accumulated_NMDA  = accumulated_NMDA*etd+tt_NMDA*etd/tdamp*self.tau_m*self.dt
        # >>> Saving & recording
        self.acm_NMDA  = accumulated_NMDA        
        # >>> multipling tau_m, and then adding accumulated INMDA >>>
        self.total_fp_vslave = 0.0
        for key,val in self.total_fpmu_dict.items():           
            try:
                self.total_fp_vslave += val
            except:
                key.initialize()
                self.total_fp_vslave += val                
#        if self.simulation.t<20:
#            self.total_fp_vslave +=0.5
            
        self.total_fp_vslave = self.total_fp_vslave * self.tau_m + self.acm_NMDA
        # >>> Active integration >>>
        self.total_Inmda = 0.0
        for key,val in self.total_Inmda_dict.items():
            try:
                self.total_Inmda += val
            except:
                key.initialize()
                self.total_Inmda += val
        self.total_Inmda = self.total_Inmda  * self.tau_m        
    
    def update_total_fpsig_dict(self):
        """
        >>> Update sigma(variance) for fokker-planck equation
        """
        for c in self.source_connection_list:
            self.total_fpsig_dict[c.connection_distribution] = 0.0
        for c in self.source_connection_list:
            if(c.conn_type == 'ShortRange'):
                self.total_fpsig_dict[c.connection_distribution] += c.curr_delayed_firing_rate * (c.weights**2) * c.nsyn #1.0 * (self.testw**2)

        # >>>>>>>>>>>>>> Calculate summation >>>>>>>>>>>>>>
        self.total_fp_sigv = 0.0
        for key,val in self.total_fpsig_dict.items():
            try:
                self.total_fp_sigv += val
            except:
                key.initialize()
                self.total_fp_sigv += val
                
        if self.simulation.t<00.0:
            self.total_fp_sigv +=0.5
        self.total_fp_sigv = self.total_fp_sigv * self.tau_m
   
    def update_fp_moment4(self):
        v1 = self.v1.copy()
        v2 = self.v2.copy()
        v3 = self.v3.copy()
        v4 = self.v4.copy()

        fr = self.curr_firing_rate
        vs = self.total_fp_vslave
        ds = self.total_fp_sigv

        dtgL = self.dt / self.tau_m
        gL   = 1.0/self.tau_m

        v1n = v1 + dtgL*(-fr/gL - (v1-vs))
        v2n = v2 + dtgL*(-fr/gL - 2.0*(v2-vs*v1-0.5*ds))
        v3n = v3 + dtgL*(-fr/gL - 3.0*(v3-vs*v2-ds*v1))
        v4n = v4 + dtgL*(-fr/gL - 4.0*(v4-vs*v3-1.5*ds*v2))

        self.v1 = v1n
        self.v2 = v2n
        self.v3 = v3n
        self.v4 = v4n   
        #print('inter moment: v1 %.5f %.5f'%(self.v1,self.v2))

    def update_rhoEQ(self):
        vs = self.total_fp_vslave
        ds = self.total_fp_sigv
        fin  = self.fin
        gL   = 1.0/self.tau_m

        h = self.edges[2]-self.edges[1]
        vedges = self.edges
        vedges = 0.5*(vedges[0:-1] + vedges[1:])
        rhoEQ,sum_rhoEQ = util.rho_EQ(vs,ds,vedges)
        
        self.sum_rhoEQ  = sum_rhoEQ 
        self.rhoEQ      = rhoEQ
        
    def update_MFE(self):
        vedges = self.edges
        h      = vedges[2] - vedges[1]
        vedges = 0.5 * (vedges[0:-1] + vedges[1:])
        rhov   = self.rhov
        NumCell = self.NumCell
        self.V_sample = util.getsamples(vedges,rhov,NumCell)
        
        local_pevent = 1.0
        NE,NI = 0,0
        if self.ei_pop =='e':
            #print('inter target:')
            for c in self.target_connection_list:
                if (c.conn_type =='ShortRange'):
                    #print(c.idx_kick)
                    idx_kick = c.idx_kick
                    idx_vT   = c.idx_vT
                    trhov     = c.curr_trhov
                    nsyn_post = c.nsyn_post
                    prob_event = (1.0 - np.sum(np.squeeze(trhov[idx_kick:idx_vT])) * h) ** nsyn_post                    
                    local_pevent *= prob_event                        
                else:
                    local_pevent *= 1.0
            NE = self.NumCell
            local_pevent = NE *  self.curr_firing_rate * (1-local_pevent) * self.dt 
            self.local_pevent = local_pevent
        else:
            self.local_pevent = 0.0
        
#        ''' MFE happened or not '''
#        local_pevent_d = np.random.random()
#        if (local_pevent_d < local_pevent):
#            self.MFE_flag = 1
#            self.MFE_num += 1
#            ''' SELF '''
#            VE = []
#            VI = []
#            VE = self.V_sample
#            VE[0:2] = 1.0
#            for c in self.target_connection_list:
#                if (c.conn_type == 'ShortRange'):
#                    if(c.ei_pop_post=='e'):
#                        VE = np.append(VE,c.V_sample_post)
#                    else:
#                        VI = np.append(VI,c.V_sample_post)      
            
    def update(self):
        if self.USUALorMFE:
            self.update_NMDA_midvar_syncurr()
            self.update_total_fpsig_dict()
            self.update_total_fpmu_dict()
            self.update_fp_moment4()
            self.update_rhoEQ()
        else:
            self.update_total_fpsig_dict()
            self.update_total_fpmu_dict()
            self.update_rhoEQ()

        vs   = self.total_fp_vslave
        ds   = self.total_fp_sigv
        La0  = self.La0
        fin  = self.fin
        gL   = 1.0/self.tau_m

        self.La1  = La0
        h = self.edges[2]-self.edges[1]
        vedges = self.edges
        # >>> Transfer Voltage-edge into Voltage-bin >>>
        vedges = 0.5*(vedges[0:-1] + vedges[1:])
        rhoEQ  = self.rhoEQ
        
        gamma = [1,self.v1,self.v2,self.v3,self.v4]
        fi    = np.transpose(gamma)
        F     = fi[1:3]
        (tmu,tx,tPEq,tfin,tgamma) = (F,vedges,self.rhoEQ,fin,1.0)
        a0    = La0
        res   = minimize(util.optfun,a0,args=(tmu,tx,tPEq,tfin,tgamma))
        La1   = res.x
        La0   = La1#np.real(La1)
        self.La0 = La0
        self.La1 = La1
        
        if self.USUALorMFE:
            rhov   = rhoEQ*np.exp(np.dot(np.squeeze(fin[:,:]),La0),dtype=np.float64)
            # >>> Normalization >>>
            rhov        = rhov/(h*np.sum(rhov))
            sum_rhoEQ   = self.sum_rhoEQ
            # >>> Here, we calculate firing-rate and other characters, under normal occasion >>>
            # >>> MFE didn't recalculate firing rate based on this equation, only base on LE >>>          
            firing_rate  = gL*np.sqrt(ds)*np.exp(np.sum(La1))/sum_rhoEQ/2.0

            # >>> Warning and Error dealing >>>
            if np.isnan(firing_rate)|np.isnan(np.sum(rhov))|np.isinf(firing_rate)|np.isinf(np.sum(rhov))|\
            np.isneginf(firing_rate)|np.isneginf(np.sum(rhov)):
                print('time: ',self.simulation.t,'fr: ',firing_rate,' ds: ', ds,' sumla1:', np.sum(La1),' sumrho: ', sum_rhoEQ)
                # >>> Firing-rate resets to zero, long-range NMDA & HNMDA reset as well >>>
                firing_rate = 0.0
                self.acm_NMDA  = 0.0
                self.La1 = self.initLa0
                self.La0 = self.initLa0
                rhov     = self.source
                gamma    = self.initgamma
                
                sum_rhoEQ = (h)*np.sum(rhov)
                rhov      = rhov/sum_rhoEQ
                rhoEQ     = rhov
                
                      
            # >>> Update characters not in MFE
            self.rhov = rhov
            self.rhoEQ = rhoEQ
            self.sum_rhoEQ = sum_rhoEQ
            self.v1,self.v2,self.v3,self.v4 = gamma[1],gamma[2],gamma[3],gamma[4]
            # before MFE mE = 0
            # after MFE didn't change and mE = 0 still
            self.firing_rate = firing_rate   


    @property
    def source_connection_list(self):
        return [c for c in self.simulation.connection_list if c.post_population == self]
    
    @property
    def target_connection_list(self):
        return [c for c in self.simulation.connection_list if c.pre_population == self]
        
    @property
    def curr_firing_rate(self):
        return self.firing_rate
    
    @property
    def curr_rhov(self):
        return self.rhov

    @property
    def curr_Inmda(self):
        return self.inmda
    
    @property
    def curr_V_sample(self):
        return self.V_sample

       

        
