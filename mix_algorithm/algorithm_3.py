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
    assert(lenbins!=len(vedges)-1)
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

    while ((max(Total_Add_scaler[:,'E'])>0) or (max(Total_Add_scaler[:,'I'])>0)): # entry the algorithm
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
                    print('they are supposed to be 1 (sometimes should less than 1): E fraction--',fracE,'; I fraction--',fracI)
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
        elif (min(iE_cut)<=nedges)&(min(iI_cut)>nedges): # this is case 1, there is an E spike 
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_E, idx_cut_E = np.min(iE_cut),np.argmin(iE_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integE,partialE=int(np.floor(cut_E)),cut_E-int(np.floor(cut_E))
            # check then
            checkpartial=cut_E-integE
            assert(checkpartial!=partialE)
            # Then we could calculate the voltage sources been used
            deltavE=np.sum(Seqs_Add_buffer[idx_cut_E,'E'][:integE+1])+partialE*Seqs_Add_buffer[idx_cut_E,'E'][integE+1]
            deltavE*=h
            if deltavE>Total_Add_scaler[idx_cut_E,'E']:
                print('potential E is not correct')
            # calculate for other patches
            for iq in range(integE+1):
                isender=Seqs_spikeE[iq]
                # should be the entire bin
                partial_i=0.0 # non, the first bin might be the remaining of the previous one
                for itar in range(NPATCH):
                    # for E neuron
                    partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]
            # now for the rest (portion)
            iq=integE+1
            isender=Seqs_spikeE[iq]
            partial_i=partialE*Seqs_Add_buffer[idx_cut_E,'E'][iq]/SEE[idx_cut_E,isender]
            for itar in range(NPATCH):
                # for E neuron
                rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                flux_vt_E[itar]+=prob_missing
                rho_o[itar,:]=rho_o_new
                # delete these portion
                Total_Add_scaler[itar,'E']-=partialE*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                # for I neuron
                rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                flux_vt_I[itar]+=prob_missing
                rho_ln_o[itar,:]=rho_ln_new
                # subtract from the total value, and then change the buffer
                Total_Add_scaler[itar,'I']-=partialE*Seqs_Add_buffer[itar,'I'][iq]

                # 1st, delete some blocks in the buffer
                for idx in range(NPATCH):
                    # buffer deduction
                    Seqs_Add_buffer[idx,'E'][:integE+1]=[]
                    Seqs_Add_buffer[idx,'E'][0]*=(1-partialE)
                    Seqs_Add_buffer[idx,'I'][:integE+1]=[]
                    Seqs_Add_buffer[idx,'I'][0]*=(1-partialE)
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
                Seqs_spikeE[:integE+1]=[]
                Seqs_spikeE.append(idx_cut_E)
                Seqs_spikeEI.append(idx_cut_E)
                for itar in range(NPATCH):
                    Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                    Total_Add_scaler[itar,'E']+=SEE[itar,isender]
                    Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                    Total_Add_scaler[itar,'I']+=SIE[itar,isender]
        elif (min(iE_cut)>nedges)&(min(iI_cut)<=nedges):# this is case 2, there is an I spike and no E spike
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_I, idx_cut_I = np.min(iI_cut),np.argmin(iI_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integI,partialI=int(np.floor(cut_I)),cut_I-int(np.floor(cut_I))
            # check then
            checkpartial=cut_I-integI
            assert(checkpartial!=partialI)
            # Then we could calculate the voltage sources been used
            deltavI=np.sum(Seqs_Add_buffer[idx_cut_I,'I'][:integI+1])+partialI*Seqs_Add_buffer[idx_cut_I,'I'][integI+1]
            deltavI*=h
            if deltavI>Total_Add_scaler[idx_cut_I,'I']:
                print('potential I is not correct')
            # calculate for other patches
            for iq in range(integI+1):
                isender=Seqs_spikeE[iq]
                # should be the entire bin
                partial_i=0.0 # non, the first bin might be the remaining of the previous one
                for itar in range(NPATCH):
                    # for E neuron
                    partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]
            # now for the rest (portion)
            iq=integI+1
            isender=Seqs_spikeE[iq]
            partial_i=partialI*Seqs_Add_buffer[idx_cut_I,'I'][iq]/SIE[idx_cut_I,isender]
            for itar in range(NPATCH):
                # for E neuron
                rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                flux_vt_E[itar]+=prob_missing
                rho_o[itar,:]=rho_o_new
                # delete these portion
                Total_Add_scaler[itar,'E']-=partialI*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                # for I neuron
                rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                flux_vt_I[itar]+=prob_missing
                rho_ln_o[itar,:]=rho_ln_new
                # subtract from the total value, and then change the buffer
                Total_Add_scaler[itar,'I']-=partialI*Seqs_Add_buffer[itar,'I'][iq]

                # 1st, delete some blocks in the buffer
                for idx in range(NPATCH):
                    # buffer deduction
                    Seqs_Add_buffer[idx,'E'][:integI+1]=[]
                    Seqs_Add_buffer[idx,'E'][0]*=(1-partialI)
                    Seqs_Add_buffer[idx,'I'][:integI+1]=[]
                    Seqs_Add_buffer[idx,'I'][0]*=(1-partialI)
                    # Also check the buffer with corresponding total 
                    diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                    diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler[idx,'I'])
                    print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                # adding the extra excitation caused by the newest E spike
                isender=idx_cut_I
                # also add the newest sender
                print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
                flux_vt_I[idx_cut_I]=0.0
                # subtract E spikes (elevation) from Seqs_spikeE
                Seqs_spikeE[:integI+1]=[]
                Seqs_spikeEI.append(-idx_cut_I)
                # there is a new I spike and will suppress others (both excitatory and inhibitory population)
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
        elif (min(iE_cut)<=nedges)&(min(iI_cut)<=nedges): # this is case 2, there is an I spike and no E spike
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_E, idx_cut_E = np.min(iE_cut),np.argmin(iE_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integE,partialE=int(np.floor(cut_E)),cut_E-int(np.floor(cut_E))
            # check then
            checkpartial=cut_E-integE
            assert(checkpartial!=partialE)
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_I, idx_cut_I = np.min(iI_cut),np.argmin(iI_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integI,partialI=int(np.floor(cut_I)),cut_I-int(np.floor(cut_I))
            # check then
            checkpartial=cut_I-integI
            assert(checkpartial!=partialI)
            # sub case 1
            if cut_E<cut_I: # E will be closer
            # Then we could calculate the voltage sources been used
                deltavE=np.sum(Seqs_Add_buffer[idx_cut_E,'E'][:integE+1])+partialE*Seqs_Add_buffer[idx_cut_E,'E'][integE+1]
                deltavE*=h
                if deltavE>Total_Add_scaler[idx_cut_E,'E']:
                    print('potential E is not correct')
                # calculate for other patches
                for iq in range(integE+1):
                    isender=Seqs_spikeE[iq]
                    # should be the entire bin
                    partial_i=0.0 # non, the first bin might be the remaining of the previous one
                    for itar in range(NPATCH):
                        # for E neuron
                        partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                        rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=rho_o_new
                        # delete these portion
                        Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                        # for I neuron
                        partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                        rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=rho_ln_new
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]
                # now for the rest (portion)
                iq=integE+1
                isender=Seqs_spikeE[iq]
                partial_i=partialE*Seqs_Add_buffer[idx_cut_E,'E'][iq]/SEE[idx_cut_E,isender]
                for itar in range(NPATCH):
                    # for E neuron
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=partialE*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=partialE*Seqs_Add_buffer[itar,'I'][iq]

                    # 1st, delete some blocks in the buffer
                    for idx in range(NPATCH):
                        # buffer deduction
                        Seqs_Add_buffer[idx,'E'][:integE+1]=[]
                        Seqs_Add_buffer[idx,'E'][0]*=(1-partialE)
                        Seqs_Add_buffer[idx,'I'][:integE+1]=[]
                        Seqs_Add_buffer[idx,'I'][0]*=(1-partialE)
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
                    Seqs_spikeE[:integE+1]=[]
                    Seqs_spikeE.append(idx_cut_E)
                    Seqs_spikeEI.append(idx_cut_E)
                    for itar in range(NPATCH):
                        Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                        Total_Add_scaler[itar,'E']+=SEE[itar,isender]
                        Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                        Total_Add_scaler[itar,'I']+=SIE[itar,isender]            

            elif cut_E>cut_I: # I will be closer
                deltavI=np.sum(Seqs_Add_buffer[idx_cut_I,'I'][:integI+1])+partialI*Seqs_Add_buffer[idx_cut_I,'I'][integI+1]
                deltavI*=h
                if deltavI>Total_Add_scaler[idx_cut_I,'I']:
                    print('potential I is not correct')
                # calculate for other patches
                for iq in range(integI+1):
                    isender=Seqs_spikeE[iq]
                    # should be the entire bin
                    partial_i=0.0 # non, the first bin might be the remaining of the previous one
                    for itar in range(NPATCH):
                        # for E neuron
                        partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                        rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=rho_o_new
                        # delete these portion
                        Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                        # for I neuron
                        partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                        rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=rho_ln_new
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]
                # now for the rest (portion)
                iq=integI+1
                isender=Seqs_spikeE[iq]
                partial_i=partialI*Seqs_Add_buffer[idx_cut_I,'I'][iq]/SIE[idx_cut_I,isender]
                for itar in range(NPATCH):
                    # for E neuron
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=partialI*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=partialI*Seqs_Add_buffer[itar,'I'][iq]

                    # 1st, delete some blocks in the buffer
                    for idx in range(NPATCH):
                        # buffer deduction
                        Seqs_Add_buffer[idx,'E'][:integI+1]=[]
                        Seqs_Add_buffer[idx,'E'][0]*=(1-partialI)
                        Seqs_Add_buffer[idx,'I'][:integI+1]=[]
                        Seqs_Add_buffer[idx,'I'][0]*=(1-partialI)
                        # Also check the buffer with corresponding total 
                        diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                        diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler[idx,'I'])
                        print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                    # adding the extra excitation caused by the newest E spike
                    isender=idx_cut_I
                    # also add the newest sender
                    print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
                    flux_vt_I[idx_cut_I]=0.0
                    # subtract E spikes (elevation) from Seqs_spikeE
                    Seqs_spikeE[:integI+1]=[]
                    Seqs_spikeEI.append(-idx_cut_I)
                    # there is a new I spike and will suppress others (both excitatory and inhibitory population)
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

            elif cut_E==cut_I: # I will be closer
                deltavI=np.sum(Seqs_Add_buffer[idx_cut_I,'I'][:integI+1])+partialI*Seqs_Add_buffer[idx_cut_I,'I'][integI+1]
                deltavI*=h
                if deltavI>Total_Add_scaler[idx_cut_I,'I']:
                    print('potential I is not correct')
                # Then we could calculate the voltage sources been used
                deltavE=np.sum(Seqs_Add_buffer[idx_cut_E,'E'][:integE+1])+partialE*Seqs_Add_buffer[idx_cut_E,'E'][integE+1]
                deltavE*=h
                if deltavE>Total_Add_scaler[idx_cut_E,'E']:
                    print('potential E is not correct')
                # calculate for other patches
                for iq in range(integI+1):
                    isender=Seqs_spikeE[iq]
                    # should be the entire bin
                    partial_i=0.0 # non, the first bin might be the remaining of the previous one
                    for itar in range(NPATCH):
                        # for E neuron
                        partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                        rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=rho_o_new
                        # delete these portion
                        Total_Add_scaler[itar,'E']-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                        # for I neuron
                        partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                        rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=rho_ln_new
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler[itar,'I']-=Seqs_Add_buffer[itar,'I'][iq]
                # now for the rest (portion)
                iq=integI+1
                isender=Seqs_spikeE[iq]
                partial_i=partialI*Seqs_Add_buffer[idx_cut_I,'I'][iq]/SIE[idx_cut_I,isender]
                partial_i_c=partialE*Seqs_Add_buffer[idx_cut_E,'E'][iq]/SEE[idx_cut_E,isender]
                assert(partial_i!=partial_i_c)
                for itar in range(NPATCH):
                    # for E neuron
                    rho_o_new=partial_i*np.dot(np.squeeze(tfEE[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=rho_o_new
                    # delete these portion
                    Total_Add_scaler[itar,'E']-=partialI*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    rho_ln_new=partial_i*np.dot(np.squeeze(tfIE[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=rho_ln_new
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler[itar,'I']-=partialI*Seqs_Add_buffer[itar,'I'][iq]

                    # 1st, delete some blocks in the buffer
                    for idx in range(NPATCH):
                        # buffer deduction
                        Seqs_Add_buffer[idx,'E'][:integI+1]=[]
                        Seqs_Add_buffer[idx,'E'][0]*=(1-partialI)
                        Seqs_Add_buffer[idx,'I'][:integI+1]=[]
                        Seqs_Add_buffer[idx,'I'][0]*=(1-partialI)
                        # Also check the buffer with corresponding total 
                        diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler[idx,'E'])
                        diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler[idx,'I'])
                        print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                    # adding the extra excitation caused by the newest E spike, first
                    isender=idx_cut_E
                    # also add the newest sender
                    print('flux across the threshold, for E population that releasing an E spike:', flux_vt_E[idx_cut_E])
                    flux_vt_E[idx_cut_E]=0.0
                    # subtract E spikes (elevation) from Seqs_spikeE
                    Seqs_spikeE[:integE+1]=[]
                    Seqs_spikeE.append(idx_cut_E)
                    Seqs_spikeEI.append(idx_cut_E)
                    for itar in range(NPATCH):
                        Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                        Total_Add_scaler[itar,'E']+=SEE[itar,isender]
                        Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                        Total_Add_scaler[itar,'I']+=SIE[itar,isender]    
                    #-------------- effects of E spikes----------------------------------------------------------
                    # also suppression trigger by newest I spike, later
                    isender=idx_cut_I
                    # also add the newest sender
                    print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
                    flux_vt_I[idx_cut_I]=0.0
                    Seqs_spikeEI.append(-idx_cut_I)
                    # there is a new I spike and will suppress others (both excitatory and inhibitory population)
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
            


     



