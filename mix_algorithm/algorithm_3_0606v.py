import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import scipy.integrate as spi
import bisect
from scipy.optimize import minimize
from scipy import special
import time
import scipy.special as sm
import matplotlib.pylab as plt
import itertools

# input parameters and variables (rho_o,rho_ln_o,SEE,SIE,SEI,SII,tf..,N_neuron_fire_first,Seqs_Add_buffer,Total_Add_scaler[index,type],    Seqs_subt_buffer[index,type]=[])
def cal_CDF(vedges,rho,rho_ln):
    NPATCH,lenbins=np.size(rho)
    assert(lenbins==len(vedges)-1)
    h = vedges[1]-vedges[0]
    cdf_o,cdf_ln_o=np.zeros_like(rho),np.zeros_like(rho_ln)
    for idx in range(NPATCH):
        for ii in range(lenbins):
            cdf_o[idx,ii]=np.sum(rho[idx,:ii])*h
            cdf_ln_o[idx,ii]=np.sum(rho_ln[idx,:ii])*h

    return cdf_o,cdf_ln_o



def algrithm_3(NPATCH,NE,NI,vbins,vedges,vT,vmin,rho_o,rho_ln_o, SEE,SIE,SEI,SII,tfEE,tfIE,tfEI,tfII,N_neuron_fire_first,Seqs_Add_buffer,Total_Add_scaler,Seqs_subt_buffer,):
    '''
    Parameters:
    Variables:
    Ndarray:

    '''    
    # initial signal
    NPATCH_i=np.size(rho_o)[0]
    nedges,nbins=len(vedges),len(vbins)
    assert(NPATCH_i==NPATCH)
    h = vbins[1]-vbins[0]
    idxtrigger=np.where(N_neuron_fire_first>0)
    Seqs_spikeE=[idxtrigger,idxtrigger]
    Seqs_spikeEI=[idxtrigger,idxtrigger]
    # let's get started! First, the N_neuron_fire_first E spikes contribute to the foremost trigger-signal, which initiates the consequential E/I spikes.
    for ispks in range(N_neuron_fire_first):
        for i in range(NPATCH):
            Seqs_Add_buffer[i,'E'].append(SEE[i,idxtrigger])
            Total_Add_scaler[i,'E']+=SEE[i,idxtrigger]
            # Seqs_subt_buffer[i,'E'].append(SEI[i,idxtrigger])
            Seqs_Add_buffer[i,'I'].append(SIE[i,idxtrigger])
            Total_Add_scaler[i,'I']+=SIE[i,idxtrigger]
            # Seqs_subt_buffer[i,'I'].append(SII[i,idxtrigger])
    # tell which population fires first.
    # calculate the cumulative distribution function (literally)
    tmpE,tmpI=np.zeros((NPATCH,len(vedges))),np.zeros((NPATCH,len(vedges))) # len(vedges)=len(vbins)+1
    flux_vt_E,flux_vt_I=np.zeros(2),np.zeros(2)

    # entry the algorithm
    cdf_o,cdf_ln_o=cal_CDF(vedges,rho_o,rho_ln_o)
    # calculate the cut-off points
    potential_E,potential_I=np.zeros(2),np.zeros(2)
    iE_cut, iI_cut=(nedges+1)*np.ones(2),(nedges+1)*np.ones(2)
    for i in range(NPATCH):
        potential_E[i],potential_I[i] = flux_vt_E[i]*NE,flux_vt_I[i]*NI
        # for E neurons
        for ispks in range(len(Seqs_Add_buffer[i,'E'])):
            potential_E[i]+=Seqs_Add_buffer[i,'E'][ispks]*NE
            if potential_E[i]>1.0: # then this is the cut off, an E spike occurs
                iE_cut[i]=ispks-1
                # we should calculate portion
                partial_i = (potential_E[i]-1.0)/NE/Seqs_Add_buffer[i,'E'][ispks]
                iE_cut[i]+=partial_i
                break
        # for I neurons
        for ispks in range(len(Seqs_Add_buffer[i,'I'])):
            potential_I[i]+=Seqs_Add_buffer[i,'I'][ispks]*NI
        if potential_I[i]>1.0: # then this is the cut off, an I spike occurs
            iI_cut[i]=ispks-1
            # we should calculate portion
            partial_i = (potential_I[i]-1.0)/NI/Seqs_Add_buffer[i,'I'][ispks]
            iI_cut[i]+=partial_i
            break       
        '''
        we got iE_cut and iI_cut for how many bins (percentage) can ensure an E/I spike.
        less i_cut corresponds to closer relationship, and will also fire in advanced.
        '''

    # for i in range(NPATCH):
    #     potential_E[i],potential_I[i] = flux_vt_E[i]*NE,flux_vt_I[i]*NI
    #     # for E neurons
    #     potential_E[i] += np.interp(Total_Add_scaler[i,'E'],vbins,cdf_o[i,:],left=0.0,right=cdf_o[i,-1])*NE
    #     # for I neurons
    #     potential_I[i] += np.interp(Total_Add_scaler[i,'I'],vbins,cdf_ln_o[i,:],left=0.0,right=cdf_ln_o[i,-1])*NI
    #     # find out the closest E and I neurons in each individual populations
    #     # corresponds to idx_E idx_I max_E max_I
    #     iE_set=np.where((cdf_o[i,:]+flux_vt_E[i])*NE>=1)
    #     iI_set=np.where((cdf_ln_o[i,:]+flux_vt_I[i])*NI>=1)
    #     if not(np.empty(iE_set)):
    #         iE_cut[i]=iE_set[0]
    #     if not(np.empty(iI_set)):
    #         iI_cut[i]=iI_set[0]
    # if (max(potential_E)<1.0) & (max(potential_I)<1.0): # this is case 0, no E and I spikes
    if (min(iE_cut)>nedges) & (min(iI_cut)>nedges): # this is case 0, no E and I spikes
        # only voltage change
        for ilen in range(len(Seqs_spikeE)):
            isender = Seqs_spikeE[ilen] # E neuron, excite others 
            for itar in range(NPATCH): # targe             
                # structure fraction of SQE * transfer matrix * rho
                fracE = Seqs_Add_buffer[itar,'E'][ilen]/SEE[itar,isender]
                fracI = Seqs_Add_buffer[itar,'I'][ilen]/SIE[itar,isender] # why here we use fractional results?
                # there should be cases that the portion left in buffer can not cover the whole transfer matrix
                print('they are supposed to be 1: E fraction--',fracE,'; I fraction--',fracI)
                # E neurons
                rho_new=np.dot(fracE*tfEE[itar,isender,:,:],np.reshape(rho_o[itar,:],(nbins,1)))
                # no flux through v-threshold
                # check diff
                prob_missing= np.abs(np.sum(rho_new)-np.sum(rho_o[itar,:]))
                flux_vt_E[itar]+=prob_missing
                rho_o[itar,:]=rho_new
                # I neurons
                rho_ln_new=np.dot(fracI*tfIE[itar,isender,:,:],np.reshape(rho_ln_o[itar,:],(nbins,1)))
                prob_missing=np.abs(np.sum(rho_ln_new)-np.sum(rho_ln_o[itar,:]))
                flux_vt_I[itar]+=prob_missing
                rho_ln_o[itar,:]=rho_ln_new
    # elif (max(potential_E)>=1.0)&(max(potential_I)<1.0): # this is case 1, there is an E spike and no I spike
    elif (min(iE_cut)<nedges)&(min(iI_cut)>nedges): # this is case 1, there is an E spike 
        # there should be increment in both E and I populations
        # find the cut-off value for E spikes  refer to iE_cut
        cut_E, idx_cut_E = np.min(iE_cut),np.argmin(iE_cut) 
        # not this, if SEE_IDEN AND SEE_OPPO
        # check
        deltavE = (cut_E+1) * h
        if deltavE*NE>potential_E[idx_cut_E]:
            print('potential E is not correct')
        # calculate how many 'Quantumi_E' 
        quantumv=0.0
        for iq in range(len(Seqs_Add_buffer[idx_cut_E,'E'])):
            isender=Seqs_spikeE[iq]
            if(quantumv+Seqs_Add_buffer[idx_cut_E,'E'][iq]>deltavE):
                partial_i=(deltavE-quantumv)/Seqs_Add_buffer[idx_cut_E,'E'][iq]
                for itar in range(NPATCH):
                    # for E neuron
                    # rho_o[itar,:]
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                    # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                    # for I neuron
                    # rho_ln_o[itar,:]
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                    # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)
                end_portion = partial_i
                integer_sub=(iq-1)+1
                break
            else:
                # change Seqs_Add_buffer[idx_cut_E,'E'][iq]
                partial_i = 1.0
                for itar in range(NPATCH):
                    # for E neuron
                    # for E neuron
                    # rho_o[itar,:]
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                    # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                    # for I neuron
                    # rho_ln_o[itar,:]
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                    # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)

            # 1st, delete some blocks in the buffer
            for idx in range(NPATCH):
                # buffer deduction
                Seqs_Add_buffer[idx,'E'][:integer_sub]=[]
                Seqs_Add_buffer[idx,'E'][0]*=(1-end_portion)
                Seqs_Add_buffer[idx,'I'][:integer_sub]=[]
                Seqs_Add_buffer[idx,'I'][0]*=(1-end_portion)
                # Also check the buffer with corresponding total 
                diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
            # adding the extra excitation caused by the newest E spike
            isender=idx_cut_E
            # also add the newest sender
            print('flux across the threshold, for E population that releasing an E spike:', flux_vt_E[idx_cut_E])
            flux_vt_E[idx_cut_E]=0.0
            # subtract E spikes (elevation) from Seqs_spikeE
            Seqs_spikeE[:integer_sub]=[]
            Seqs_spikeE.append(idx_cut_E)
            Seqs_spikeEI.append(idx_cut_E)
            for itar in range(NPATCH):
                Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                Total_Add_scaler[itar,'E']+=SEE[itar,isender]
                Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                Total_Add_scaler[itar,'I']+=SIE[itar,isender]
    elif (max(potential_E)<1.0)&(max(potential_I)>=1.0): # this is case 2, there is an I spike and no E spike
        # there should be increment in both E and I populations
        # find the cut-off value for E spikes  refer to iE_cut
        cut_I, idx_cut_I = np.min(iI_cut),np.argmin(iI_cut)
        # check
        deltavI = (cut_I+1) * h
        if deltavI*NI>potential_I[idx_cut_I]:
            print('potential I is not correct')
        # calculate how many 'Quantumi_E' 
        quantumv=0.0
        for iq in range(len(Seqs_Add_buffer[idx_cut_I,'I'])):
            isender=Seqs_spikeE[iq]
            if(quantumv+Seqs_Add_buffer[idx_cut_I,'I'][iq]>deltavI):
                partial_i=(deltavI-quantumv)/Seqs_Add_buffer[idx_cut_I,'I'][iq]
                for itar in range(NPATCH):
                    # for E neuron
                    # rho_o[itar,:]
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                    # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                    # for I neuron
                    # rho_ln_o[itar,:]
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                    # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)
                end_portion = partial_i
                integer_sub=(iq-1)+1
                break
            else:
                # change Seqs_Add_buffer[idx_cut_E,'E'][iq]
                partial_i = 1.0
                for itar in range(NPATCH):
                    # for E neuron
                    # for E neuron
                    # rho_o[itar,:]
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                    # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                    # for I neuron
                    # rho_ln_o[itar,:]
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                    # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)

            # 1st, delete some blocks in the buffer
            for idx in range(NPATCH):
                # buffer deduction
                Seqs_Add_buffer[idx,'E'][:integer_sub]=[] # entire vbin
                Seqs_Add_buffer[idx,'E'][0]*=(1-end_portion) # portion
                Seqs_Add_buffer[idx,'I'][:integer_sub]=[]
                Seqs_Add_buffer[idx,'I'][0]*=(1-end_portion)
                # Also check the buffer with corresponding total 
                diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
            # adding the extra excitation caused by the newest E spike
            isender=idx_cut_I
            # also add the newest sender
            print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
            flux_vt_I[idx_cut_I]=0.0
            # Seqs_spikeE.append(idx_cut_E)
            Seqs_spikeEI.append(-idx_cut_I)
            for itar in range(NPATCH):
                # for E neuron
                rho_o_new=np.dot(np.squeeze(tfEI[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                rho_o_new[-1]+=prob_missing
                rho_o[itar,:]=rho_o_new

                # for I neuron
                rho_ln_new=np.dot(np.squeeze(tfII[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                rho_ln_new[-1]+=prob_missing
                rho_ln_o[itar,:]=rho_ln_new
                # Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                # Total_Add_scaler[itar,'E']+=SEE[itar,isender]
                # Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                # Total_Add_scaler[itar,'I']+=SIE[itar,isender]
    elif (max(potential_E)>=1.0)&(max(potential_I)>=1.0): # this is case 2, there is an I spike and no E spike
        # there should be increment in both E and I populations
        # find the cut-off value for E spikes  refer to iE_cut
        cut_I, idx_cut_I = np.min(iI_cut),np.argmin(iI_cut)
        cut_E, idx_cut_E = np.min(iE_cut),np.argmin(iE_cut)
        # check
        deltavI,deltaE = (cut_I+1) * h,(cut_E+1) * h
        if deltavI*NI>potential_I[idx_cut_I]:
            print('potential I is not correct')
        if deltavE*NI>potential_E[idx_cut_E]:
            print('potential E is not correst')
        # check whether the E and I neuron will fire first
        if cut_E <= cut_I: # which means E neuron will fire first 
            # calculate how many 'Quantumi_E' 
            quantumv=0.0
            for iq in range(len(Seqs_Add_buffer[idx_cut_E,'E'])):
                isender=Seqs_spikeE[iq]
                if(quantumv+Seqs_Add_buffer[idx_cut_E,'E'][iq]>deltavE):
                    partial_i=(deltavE-quantumv)/Seqs_Add_buffer[idx_cut_E,'E'][iq]
                    for itar in range(NPATCH):
                        # for E neuron
                        # rho_o[itar,:]
                        rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=rho_o_new
                        # delete these portion
                        Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                        # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                        # for I neuron
                        # rho_ln_o[itar,:]
                        rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=rho_ln_new
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                        # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)
                    end_portion = partial_i
                    integer_sub=(iq-1)+1
                    break
                else:
                    # change Seqs_Add_buffer[idx_cut_E,'E'][iq]
                    partial_i = 1.0
                    for itar in range(NPATCH):
                        # for E neuron
                        # for E neuron
                        # rho_o[itar,:]
                        rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=rho_o_new
                        # delete these portion
                        Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                        # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                        # for I neuron
                        # rho_ln_o[itar,:]
                        rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=rho_ln_new
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                        # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)

                # 1st, delete some blocks in the buffer
                for idx in range(NPATCH):
                    # buffer deduction
                    Seqs_Add_buffer[idx,'E'][:integer_sub]=[]
                    Seqs_Add_buffer[idx,'E'][0]*=(1-end_portion)
                    Seqs_Add_buffer[idx,'I'][:integer_sub]=[]
                    Seqs_Add_buffer[idx,'I'][0]*=(1-end_portion)
                    # Also check the buffer with corresponding total 
                    diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                    diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler[idx,'I'])
                    print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                # adding the extra excitation caused by the newest E spike
                isender=idx_cut_E
                # also add the newest sender
                print('flux across the threshold, for E population that releasing an E spike:', flux_vt_E[idx_cut_E])
                flux_vt_E[idx_cut_E]=0.0
                # subtract E spikes (elevation) from Seqs_spikeE
                Seqs_spikeE[:integer_sub]=[]
                Seqs_spikeE.append(idx_cut_E)
                Seqs_spikeEI.append(idx_cut_E)
                for itar in range(NPATCH):
                    Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                    Total_Add_scaler[itar,'E']+=SEE[itar,isender]
                    Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                    Total_Add_scaler[itar,'I']+=SIE[itar,isender]
        elif cut_I<cut_E:
            # calculate how many 'Quantumi_E' 
            quantumv=0.0
            for iq in range(len(Seqs_Add_buffer[idx_cut_I,'I'])):
                isender=Seqs_spikeE[iq]
                if(quantumv+Seqs_Add_buffer[idx_cut_I,'I'][iq]>deltavI):
                    partial_i=(deltavI-quantumv)/Seqs_Add_buffer[idx_cut_I,'I'][iq]
                    for itar in range(NPATCH):
                        # for E neuron
                        # rho_o[itar,:]
                        rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=rho_o_new
                        # delete these portion
                        Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                        # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                        # for I neuron
                        # rho_ln_o[itar,:]
                        rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=rho_ln_new
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                        # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)
                    end_portion = partial_i
                    integer_sub=(iq-1)+1
                    break
                else:
                    # change Seqs_Add_buffer[idx_cut_E,'E'][iq]
                    partial_i = 1.0
                    for itar in range(NPATCH):
                        # for E neuron
                        # for E neuron
                        # rho_o[itar,:]
                        rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=rho_o_new
                        # delete these portion
                        Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq]*partial_i # first, subtract this portion
                        # Seqs_Add_buffer[itar,'E'][iq]*=(1-partial_i) # then, change the buffer
                        # for I neuron
                        # rho_ln_o[itar,:]
                        rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=rho_ln_new
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]*partial_i
                        # Seqs_Add_buffer[itar,'I'][iq]*=(1-partial_i)

                # 1st, delete some blocks in the buffer
                for idx in range(NPATCH):
                    # buffer deduction
                    Seqs_Add_buffer[idx,'E'][:integer_sub]=[] # entire vbin
                    Seqs_Add_buffer[idx,'E'][0]*=(1-end_portion) # portion
                    Seqs_Add_buffer[idx,'I'][:integer_sub]=[]
                    Seqs_Add_buffer[idx,'I'][0]*=(1-end_portion)
                    # Also check the buffer with corresponding total 
                    diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                    diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler[idx,'I'])
                    print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                # adding the extra excitation caused by the newest E spike
                isender=idx_cut_I
                # also add the newest sender
                print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
                flux_vt_I[idx_cut_I]=0.0
                # Seqs_spikeE.append(idx_cut_E)
                Seqs_spikeEI.append(-idx_cut_I)
                for itar in range(NPATCH):
                    # for E neuron
                    rho_o_new=np.dot(np.squeeze(tfEI[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    rho_o_new[-1]+=prob_missing
                    rho_o[itar,:]=rho_o_new

                    # for I neuron
                    rho_ln_new=np.dot(np.squeeze(tfII[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    rho_ln_new[-1]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new

            


     



