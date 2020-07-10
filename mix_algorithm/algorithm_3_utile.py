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
    NPATCH,lenbins=np.shape(rho)
    lenedges=len(vedges)
    assert(lenbins==len(vedges)-1)
    h = vedges[1]-vedges[0]
    cdf_o,cdf_ln_o=np.zeros((NPATCH,lenedges)),np.zeros((NPATCH,lenedges))
    for idx in range(NPATCH):
        for ii in range(lenedges):
            cdf_o[idx,ii]=np.sum(rho[idx,:ii])*h
            cdf_ln_o[idx,ii]=np.sum(rho_ln[idx,:ii])*h

    return cdf_o,cdf_ln_o

'''
The function calculates the population voltage distribution after inhibitory suppression 
'''
def EI_matrix_transfer_fix(weight,vedge,):
    nedge=len(vedge)
    v=vedge.copy()
    h=v[1]-v[0]
    # weight should be negative (for suppression)
    v_post=v-(weight)
    overlapping=np.zeros((nedge-1,nedge-1))
    # v_post is the ref
    # transfer from v to v_post
    for i in range(nedge-1): # v_post[i,i+1]
        cutL,cutR=0,0
        while (cutL<nedge-1) and (v[cutL]<v_post[i]):
            cutL+=1
        if cutL>0:
            cutL-=1
        cutR=cutL
        while (cutR<nedge-1) and (v[cutR]<v_post[i+1]):
            cutR+=1
        if cutR>=nedge:
            cutR=nedge-1
        # determine each individual overlap
        # 'interval probability' in v_post[i,i+1] should be fully allocated to v-bins in v(standard)
        vstd_start,vstd_end = v_post[i],v_post[i+1]
        for j in range(cutL,cutR): # including cutL, excluding cutR
            # v[j,j+1] how much in percentage does v_post[i,i+1] allocate in this voltage bin?
            if v[j+1]<=vstd_start:
                continue
            elif (v[j]<=vstd_start) and (v[j+1]<=vstd_end):
                part=v[j+1]-vstd_start
                percentage=part/h/1.0
                overlapping[j][i]=percentage
            elif (v[j]<=vstd_start) and (v[j+1]>vstd_end):
                overlapping[j][i]=1.0
            elif (v[j]>vstd_start) and (v[j]<=vstd_end) and (v[j+1]<=vstd_end):
                overlapping[j][i] = h/h
            elif (v[j]>vstd_start) and (v[j]<=vstd_end) and (v[j+1]>vstd_end):
                overlapping[j][i] = (vstd_end-v[j])/h
            elif (v[j]>vstd_end):
                continue
        # overlapping=overlapping.T
    # # if the elements are negative, there should be retreat _flux and they are supposed to be allocated in the resting state(the minimum)
    # if weight<0.0:
    #     retreat_flux = 1.0-overlapping.sum(axis=0)
    #     overlapping[-1,:]+=retreat_flux
    return overlapping

def algrithm_3(NPATCH,NE,NI,vbins,vedges,vT,vmin,rho_o,rho_ln_o, SEE,SIE,SEI,SII,tfEE,tfIE,tfEI,tfII,N_neuron_fire_first,):
    '''
    Parameters:
    Variables:
    Ndarray:

    '''    
    # Ndarray
    Seqs_Add_buffer={}
    PATCH_IDX=np.arange(NPATCH)
    for index, type in itertools.product(PATCH_IDX,['E','I']):
        Seqs_Add_buffer[index,type]=[]
    Total_Add_scaler_E=np.zeros(NPATCH)
    Total_Add_scaler_I=np.zeros(NPATCH)
    LE,LI=np.zeros(2),np.zeros(2)
    # initial signal
    NPATCH_i=np.shape(rho_o)[0]
    nedges,nbins=len(vedges),len(vbins)
    assert(NPATCH_i==NPATCH)
    h = vbins[1]-vbins[0]
    vrights=vedges[1:]
    idxtrigger=np.where(N_neuron_fire_first>0)
    Seqs_spikeE=[idxtrigger,idxtrigger]
    Seqs_spikeEI=[idxtrigger,idxtrigger]
    # let's get started! First, the N_neuron_fire_first E spikes contribute to the foremost trigger-signal, which initiates the consequential E/I spikes.
    for ispks in range(len(Seqs_spikeE)):
        for i in range(NPATCH):
            Seqs_Add_buffer[i,'E'].append(SEE[i,idxtrigger])
            Total_Add_scaler_E[i]+=SEE[i,idxtrigger]
            Seqs_Add_buffer[i,'I'].append(SIE[i,idxtrigger])
            Total_Add_scaler_I[i]+=SIE[i,idxtrigger]  # =Because there is no current here.
    # tell which population fires first.
    # calculate the cumulative distribution function (literally)
    cdf_o,cdf_ln_o=cal_CDF(vedges,rho_o,rho_ln_o)
    cdf_o/=cdf_o[-1]
    cdf_ln_o/=cdf_ln_o[-1]
    flux_vt_E,flux_vt_I=np.zeros(2),np.zeros(2)
    while ((max(Total_Add_scaler_E)>0) or (max(Total_Add_scaler_I)>0)): # entry the algorithm       
        # calculate the cut-off points
        # print('67 Total exc:',(max(Total_Add_scaler_E)) , (max(Total_Add_scaler_I)))
        print('68 flux_EI:',flux_vt_E,flux_vt_I)
        potential_E,potential_I=np.zeros(2),np.zeros(2)
        iE_cut, iI_cut=(nedges+1)*np.ones(2),(nedges+1)*np.ones(2)
        for i in range(NPATCH):
            # remaining voltage bin and probability flux
            flux_remain_E=(1.0/NE-flux_vt_E[i]) # next, find where it was
            # print('72:',flux_remain_E)
            # print('73:',max(cdf_o[i,:]))
            cutoff_E=np.where(np.squeeze(cdf_o[i,:])>=flux_remain_E)
            if len(cutoff_E[0])>0:# the remaining is enough
                # cut_off_E=cut_off_E[0][0]
                # print('79 cut off for E:',cut_off_E)
                # then iteratively calculate and finally chech the answer
                # potential_E[i],potential_I[i] = flux_vt_E[i]*NE,flux_vt_I[i]*NI
                vol_r,cdf_r,vol_ln_r,cdf_ln_r=0.0,0.0,0.0,0.0
                pre_idx_interp,idx_interp=0,0
                # for E neurons
                for ispks in range(len(Seqs_Add_buffer[i,'E'])):
                    # then should be Seqs_Add_buffer[i,'E'] vol bin
                    pre_idx_interp=int(np.floor(vol_r/h)) # pre vol bins
                    vol_r+=Seqs_Add_buffer[i,'E'][ispks]
                    idx_interp=int(np.floor(vol_r/h))  # post vol bins
                    cdf_r=np.interp(vol_r,vedges,cdf_o[i,:],left=0.0,right=cdf_o[i,-1])
                    potential_E[i]=cdf_r*NE+flux_vt_E[i]*NE
                    # print('90 potential E spikes:',Seqs_Add_buffer[i,'E'][ispks],potential_E[i])
                    # check
                    pre_vol,post_vol=np.sum(Seqs_Add_buffer[i,'E'][:ispks]),np.sum(Seqs_Add_buffer[i,'E'][:ispks+1])
                    if potential_E[i]>=1.0: # then this is the cut off, an E spike occurs
                        # then, should be somewhere between pre vol bin idx and post vol bin idx, find it 
                        for ibetween in range(pre_idx_interp,idx_interp+1,1):
                            if ibetween>(nedges-1):
                                ibetween-=1
                            cdf_between = cdf_o[i,ibetween]
                            if cdf_between>=flux_remain_E:
                                cut_off_E=ibetween
                                break
                        # ------------- index in vedges - for E ---------
                        iE_cut[i]=ispks-1 # index in buffer
                        # we should calculate portion
                        # print('97 left accurate right:',np.sum(Seqs_Add_buffer[i,'E'][:ispks]),vbins[cut_off_E],np.sum(Seqs_Add_buffer[i,'E'][:ispks+1]))
                        partial_i = vedges[cut_off_E]-np.sum(Seqs_Add_buffer[i,'E'][:ispks])
                        partial_i/=Seqs_Add_buffer[i,'E'][ispks]
                        iE_cut[i]+=partial_i
                        # print('149, iE_cut-',iE_cut[i],'; partial_i-',partial_i)
                        break
            # remaining voltage bin and probability flux
            flux_remain_I=(1.0/NI-flux_vt_I[i]) # next, find where it was
            # print('98:...',flux_remain_I)
            cutoff_I=np.where(np.squeeze(cdf_ln_o[i,:])>=flux_remain_I)
            if len(cutoff_I[0])>0:# the remaining is enough
                # cut_off_I=cut_off_I[0][0]
                # then iteratively calculate and finally chech the answer
                # potential_E[i],potential_I[i] = flux_vt_E[i]*NE,flux_vt_I[i]*NI
                vol_r,cdf_r,vol_ln_r,cdf_ln_r=0.0,0.0,0.0,0.0
                pre_idx_ln_interp,idx_ln_interp=0,0
                # for I neurons
                for ispks in range(len(Seqs_Add_buffer[i,'I'])):
                    pre_idx_ln_interp=int(np.floor(vol_ln_r/h)) # pre vol bins
                    vol_ln_r+=Seqs_Add_buffer[i,'I'][ispks]
                    idx_ln_interp=int(np.floor(vol_ln_r/h))  # post vol bins
                    cdf_ln_r=np.interp(vol_ln_r,vedges,cdf_ln_o[i,:],left=0.0,right=cdf_ln_o[i,-1])
                    potential_I[i]=cdf_ln_r*NI+flux_vt_I[i]*NI
                    if potential_I[i]>=1.0: # then this is the cut off, an I spike occurs
                        # then, should be somewhere between pre vol bin idx and post vol bin idx, find it 
                        for ibetween in range(pre_idx_ln_interp,idx_ln_interp+1,1):
                            if ibetween>(nedges-1):
                                ibetween-=1
                            cdf_ln_between = cdf_ln_o[i,ibetween]
                            if cdf_ln_between>=flux_remain_I:
                                cut_off_I=ibetween
                                break
                        iI_cut[i]=ispks-1
                        # print('119 left accurate right:',np.sum(Seqs_Add_buffer[i,'I'][:ispks]),vbins[cut_off_I],np.sum(Seqs_Add_buffer[i,'I'][:ispks+1]))
                        # we should calculate portion
                        partial_i = vedges[cut_off_I]-np.sum(Seqs_Add_buffer[i,'I'][:ispks])
                        partial_i/=Seqs_Add_buffer[i,'I'][ispks]
                        iI_cut[i]+=partial_i
                        # print('121 buffer:',vbins[cut_off_I],Seqs_Add_buffer[i,'I'][ispks])
                        break   
            '''
            we got iE_cut and iI_cut for how many bins (percentage) can ensure an E/I spike.
            less i_cut corresponds to closer relationship, and will also fire in advanced.
            '''
        '''
        '''
        # print('127, i buffer:',Seqs_Add_buffer[1,'I'])
        # print('128, cut off:',iE_cut,iI_cut)
        
        # if (max(potential_E)<1.0) & (max(potential_I)<1.0): # this is case 0, no E and I spikes
        if (min(iE_cut)>nedges) & (min(iI_cut)>nedges): # this is case 0, no E and I spikes
            print('137 Case 0, no more spikes! E',LE,';I',LI)
            # only voltage change
            for ilen in range(len(Seqs_spikeE)):
                isender = Seqs_spikeE[ilen] # E neuron, excite others 
                for itar in range(NPATCH): # targe             
                    # structure fraction of SQE * transfer matrix * rho
                    fracE = Seqs_Add_buffer[itar,'E'][ilen]/SEE[itar,isender]
                    fracI = Seqs_Add_buffer[itar,'I'][ilen]/SIE[itar,isender] # why here we use fractional results?
                    # there should be cases that the portion left in buffer can not cover the whole transfer matrix
                    # print('they are supposed to be 1 (sometimes should less than 1): E fraction--',fracE,'; I fraction--',fracI)
                    # E neurons
                    if fracE<1.0:
                        # recalculate
                        tfE = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'E'][ilen]),vedges)
                    else:
                        tfE=tfEE[itar,isender,:,:]
                    rho_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                    # no flux through v-threshold
                    # check diff
                    prob_missing= np.abs(np.sum(rho_new)-np.sum(rho_o[itar,:]))
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=np.squeeze(rho_new)
                    Total_Add_scaler_E[itar]-=Seqs_Add_buffer[itar,'E'][ilen]
                    # I neurons
                    if fracI<1.0:
                        # recalculate
                        tfI = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'I'][ilen]),vedges)
                    else:
                        tfI=tfIE[itar,isender,:,:]
                    rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=np.abs(np.sum(rho_ln_new)-np.sum(rho_ln_o[itar,:]))
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                    Total_Add_scaler_I[itar]-=Seqs_Add_buffer[itar,'I'][ilen]
            Total_Add_scaler_E,Total_Add_scaler_I=np.zeros(NPATCH),np.zeros(NPATCH)
        # elif (max(potential_E)>=1.0)&(max(potential_I)<1.0): # this is case 1, there is an E spike and no I spike
        elif (min(iE_cut)<=nedges)&(min(iI_cut)>nedges): # this is case 1, there is an E spike 
            
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_E, idx_cut_E = np.min(iE_cut),np.argmin(iE_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integE,partialE=int(np.floor(cut_E)),cut_E-int(np.floor(cut_E))
            # check then
            checkpartial=cut_E-integE
            assert(checkpartial==partialE)
            # Then we could calculate the voltage sources been used
            # print('235, ',Seqs_Add_buffer[0,'E'],Seqs_Add_buffer[1,'E'],'; integE-',integE+1,'partialE-,',partialE)
            deltavE=np.sum(Seqs_Add_buffer[idx_cut_E,'E'][:integE+1])+partialE*Seqs_Add_buffer[idx_cut_E,'E'][integE+1]
            deltavE*=h
            if deltavE>Total_Add_scaler_E[idx_cut_E]:
                print('potential E is not correct')
            # calculate for other patches
            for iq in range(integE+1):
                isender=Seqs_spikeE[iq]
                # should be the entire bin
                partial_i=0.0 # non, the first bin might be the remaining of the previous one
                for itar in range(NPATCH):
                    # for E neuron
                    partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                    if partial_i<1.0:
                        # recalculate
                        tfE = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'E'][iq]),vedges)
                    else:
                        tfE=tfEE[itar,isender,:,:]
                    rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=np.squeeze(rho_o_new)
                    # delete these portion
                    Total_Add_scaler_E[itar]-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                    if partial_i<1.0:
                        # recalculate
                        tfI = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'I'][iq]),vedges)
                    else:
                        tfI=tfIE[itar,isender,:,:]
                    rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler_I[itar]-=Seqs_Add_buffer[itar,'I'][iq]
            # now for the rest (portion)
            iq=integE+1
            isender=Seqs_spikeE[iq]
            partial_i=partialE*Seqs_Add_buffer[idx_cut_E,'E'][iq]/SEE[idx_cut_E,isender]
            for itar in range(NPATCH):
                # for E neuron
                tfE = EI_matrix_transfer_fix(np.squeeze(partialE*Seqs_Add_buffer[itar,'E'][iq]),vedges)
                rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                flux_vt_E[itar]+=prob_missing
                rho_o[itar,:]=np.squeeze(rho_o_new)
                # delete these portion
                Total_Add_scaler_E[itar]-=partialE*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                # for I neuron
                tfI = EI_matrix_transfer_fix(np.squeeze(partialE*Seqs_Add_buffer[itar,'I'][iq]),vedges)
                rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                flux_vt_I[itar]+=prob_missing
                rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                # subtract from the total value, and then change the buffer
                Total_Add_scaler_I[itar]-=partialE*Seqs_Add_buffer[itar,'I'][iq]

            # 1st, delete some blocks in the buffer
            for idx in range(NPATCH):
                # buffer deduction
                Seqs_Add_buffer[idx,'E'][:integE+1]=[]
                Seqs_Add_buffer[idx,'E'][0]*=(1-partialE)
                Seqs_Add_buffer[idx,'I'][:integE+1]=[]
                Seqs_Add_buffer[idx,'I'][0]*=(1-partialE)
                # Also check the buffer with corresponding total 
                diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler_E[idx])
                diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler_I[idx])
                # print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
            # adding the extra excitation caused by the newest E spike
            isender=idx_cut_E
            # also add the newest sender
            # print('flux across the threshold, for E population that releasing an E spike:', flux_vt_E[idx_cut_E])
            flux_vt_E[idx_cut_E]=0.0
            # subtract E spikes (elevation) from Seqs_spikeE
            Seqs_spikeE[:integE+1]=[]
            Seqs_spikeE.append(idx_cut_E)
            Seqs_spikeEI.append(idx_cut_E)
            for itar in range(NPATCH):
                Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                Total_Add_scaler_E[itar]+=SEE[itar,isender]
                Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                Total_Add_scaler_I[itar]+=SIE[itar,isender]
            LE[idx_cut_E]+=1
            print('338 Case 1, Excitatory spikes! E',LE,';I',LI)
        elif (min(iE_cut)>nedges)&(min(iI_cut)<=nedges):# this is case 2, there is an I spike and no E spike
            print('251 Case 2, Inhibitory spikes! E',LE,';I',LI)
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_I, idx_cut_I = np.min(iI_cut),np.argmin(iI_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integI,partialI=int(np.floor(cut_I)),cut_I-int(np.floor(cut_I))
            # check then
            checkpartial=cut_I-integI
            assert(checkpartial==partialI)
            # Then we could calculate the voltage sources been used
            deltavI=np.sum(Seqs_Add_buffer[idx_cut_I,'I'][:integI+1])+partialI*Seqs_Add_buffer[idx_cut_I,'I'][integI+1]
            deltavI*=h
            if deltavI>Total_Add_scaler_I[idx_cut_I]:
                print('potential I is not correct')
            # calculate for other patches
            for iq in range(integI+1):
                isender=Seqs_spikeE[iq]
                # should be the entire bin
                partial_i=0.0 # non, the first bin might be the remaining of the previous one
                for itar in range(NPATCH):
                    # for E neuron
                    partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                    if partial_i<1.0:
                        # recalculate
                        tfE = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'E'][iq]),vedges)
                    else:
                        tfE=tfEE[itar,isender,:,:]
                    rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=np.squeeze(rho_o_new)
                    # delete these portion
                    Total_Add_scaler_E[itar]-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                    if partial_i<1.0:
                        # recalculate
                        tfI = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'I'][iq]),vedges)
                    else:
                        tfI=tfIE[itar,isender,:,:]
                    rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler_I[itar]-=Seqs_Add_buffer[itar,'I'][iq]

            # now for the rest (portion)
            iq=integI+1
            isender=Seqs_spikeE[iq]
            partial_i=partialI*Seqs_Add_buffer[idx_cut_I,'I'][iq]/SIE[idx_cut_I,isender]
            for itar in range(NPATCH):
                # for E neuron
                tfE = EI_matrix_transfer_fix(np.squeeze(partialI*Seqs_Add_buffer[itar,'E'][iq]),vedges)
                rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                flux_vt_E[itar]+=prob_missing
                rho_o[itar,:]=np.squeeze(rho_o_new)
                # delete these portion
                Total_Add_scaler_E[itar]-=partialI*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                # for I neuron
                tfI = EI_matrix_transfer_fix(np.squeeze(partialI*Seqs_Add_buffer[itar,'I'][iq]),vedges)
                rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                flux_vt_I[itar]+=prob_missing
                rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                # subtract from the total value, and then change the buffer
                Total_Add_scaler_I[itar]-=partialI*Seqs_Add_buffer[itar,'I'][iq]

            # 1st, delete some blocks in the buffer
            for idx in range(NPATCH):
                # buffer deduction
                Seqs_Add_buffer[idx,'E'][:integI+1]=[]
                Seqs_Add_buffer[idx,'E'][0]*=(1-partialI)
                Seqs_Add_buffer[idx,'I'][:integI+1]=[]
                Seqs_Add_buffer[idx,'I'][0]*=(1-partialI)
                # Also check the buffer with corresponding total 
                diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler_E[idx])
                diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler_I[idx])
                # print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
            # adding the extra excitation caused by the newest E spike
            isender=idx_cut_I
            # also add the newest sender
            # print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
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
                rho_o[itar,:]=np.squeeze(rho_o_new)

                # for I neuron
                rho_ln_new=np.dot(np.squeeze(tfII[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                rho_ln_new[-1]+=prob_missing
                rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                # Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                # Total_Add_scaler[itar,'E']+=SEE[itar,isender]
                # Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                # Total_Add_scaler[itar,'I']+=SIE[itar,isender]
            LI[idx_cut_I]+=1
        elif (min(iE_cut)<=nedges)&(min(iI_cut)<=nedges): # this is case 2, there is an I spike and no E spike
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_E, idx_cut_E = np.min(iE_cut),np.argmin(iE_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integE,partialE=int(np.floor(cut_E)),cut_E-int(np.floor(cut_E))
            # check then
            checkpartial=cut_E-integE
            assert(checkpartial==partialE)
            # there should be increment in both E and I populations
            # find the cut-off value for E spikes  refer to iE_cut
            cut_I, idx_cut_I = np.min(iI_cut),np.argmin(iI_cut) 
            # not this, if SEE_IDEN AND SEE_OPPO
            integI,partialI=int(np.floor(cut_I)),cut_I-int(np.floor(cut_I))
            # check then
            checkpartial=cut_I-integI
            assert(checkpartial==partialI)
            # sub case 1
            if cut_E<cut_I: # E will be closer
                
                # Then we could calculate the voltage sources been used
                # print('447, size-',np.shape(Seqs_Add_buffer[idx_cut_E,'E']),'; integE-',integE,integE+1)
                deltavE=np.sum(Seqs_Add_buffer[idx_cut_E,'E'][:integE+1])+partialE*Seqs_Add_buffer[idx_cut_E,'E'][integE+1]
                deltavE*=h
                if deltavE>Total_Add_scaler_E[idx_cut_E]:
                    print('potential E is not correct')
                # calculate for other patches
                for iq in range(integE+1):
                    isender=Seqs_spikeE[iq]
                    # should be the entire bin
                    partial_i=0.0 # non, the first bin might be the remaining of the previous one
                    for itar in range(NPATCH):
                        # for E neuron
                        partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                        if partial_i<1.0:
                            # recalculate
                            tfE = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'E'][iq]),vedges)
                        else:
                            tfE=tfEE[itar,isender,:,:]
                        rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=np.squeeze(rho_o_new)
                        # delete these portion
                        Total_Add_scaler_E[itar]-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                        # for I neuron
                        partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                        if partial_i<1.0:
                            # recalculate
                            tfI = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'I'][iq]),vedges)
                        else:
                            tfI=tfIE[itar,isender,:,:]
                        rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler_I[itar]-=Seqs_Add_buffer[itar,'I'][iq]
                # now for the rest (portion)
                iq=integE+1
                isender=Seqs_spikeE[iq]
                partial_i=partialE*Seqs_Add_buffer[idx_cut_E,'E'][iq]/SEE[idx_cut_E,isender]
                for itar in range(NPATCH):
                    # for E neuron
                    # print('489-increment:',partialE*Seqs_Add_buffer[itar,'E'][iq])
                    tfE = EI_matrix_transfer_fix(np.squeeze(partialE*Seqs_Add_buffer[itar,'E'][iq]),vedges)
                    rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=np.squeeze(rho_o_new)
                    # delete these portion
                    Total_Add_scaler_E[itar]-=partialE*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    tfI = EI_matrix_transfer_fix(np.squeeze(partialE*Seqs_Add_buffer[itar,'I'][iq]),vedges)
                    rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler_I[itar]-=partialE*Seqs_Add_buffer[itar,'I'][iq]

                # 1st, delete some blocks in the buffer
                for idx in range(NPATCH):
                    # buffer deduction
                    Seqs_Add_buffer[idx,'E'][:integE+1]=[]
                    Seqs_Add_buffer[idx,'E'][0]*=(1-partialE)
                    Seqs_Add_buffer[idx,'I'][:integE+1]=[]
                    Seqs_Add_buffer[idx,'I'][0]*=(1-partialE)
                    # Also check the buffer with corresponding total 
                    diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler_E[idx])
                    diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler_I[idx])
                    # print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                # adding the extra excitation caused by the newest E spike
                isender=idx_cut_E
                # also add the newest sender
                # print('flux across the threshold, for E population that releasing an E spike:', flux_vt_E[idx_cut_E])
                flux_vt_E[idx_cut_E]=0.0
                # subtract E spikes (elevation) from Seqs_spikeE
                Seqs_spikeE[:integE+1]=[]
                Seqs_spikeE.append(idx_cut_E)
                Seqs_spikeEI.append(idx_cut_E)
                for itar in range(NPATCH):
                    Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                    Total_Add_scaler_E[itar]+=SEE[itar,isender]
                    Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                    Total_Add_scaler_I[itar]+=SIE[itar,isender]
                LE[idx_cut_E]+=1
                print('552 Case 3-1, E spikes! E',LE,';I',LI)
            elif cut_E>cut_I: # I will be closer
                
                # Then we could calculate the voltage sources been used
                deltavI=np.sum(Seqs_Add_buffer[idx_cut_I,'I'][:integI+1])+partialI*Seqs_Add_buffer[idx_cut_I,'I'][integI+1]
                deltavI*=h
                if deltavI>Total_Add_scaler_I[idx_cut_I]:
                    print('potential I is not correct')
                # calculate for other patches
                for iq in range(integI+1):
                    isender=Seqs_spikeE[iq]
                    # should be the entire bin
                    partial_i=0.0 # non, the first bin might be the remaining of the previous one
                    for itar in range(NPATCH):
                        # for E neuron
                        partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                        if partial_i<1.0:
                            # recalculate
                            tfE = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'E'][iq]),vedges)
                        else:
                            tfE=tfEE[itar,isender,:,:]
                        rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=np.squeeze(rho_o_new)
                        # delete these portion
                        Total_Add_scaler_E[itar]-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                        # for I neuron
                        partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                        if partial_i<1.0:
                            # recalculate
                            tfI = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'I'][iq]),vedges)
                        else:
                            tfI=tfIE[itar,isender,:,:]
                        rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler_I[itar]-=Seqs_Add_buffer[itar,'I'][iq]

                # now for the rest (portion)
                iq=integI+1
                isender=Seqs_spikeE[iq]
                partial_i=partialI*Seqs_Add_buffer[idx_cut_I,'I'][iq]/SIE[idx_cut_I,isender]
                for itar in range(NPATCH):
                    # for E neuron
                    tfE = EI_matrix_transfer_fix(np.squeeze(partialI*Seqs_Add_buffer[itar,'E'][iq]),vedges)
                    rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=np.squeeze(rho_o_new)
                    # delete these portion
                    Total_Add_scaler_E[itar]-=partialI*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    tfI = EI_matrix_transfer_fix(np.squeeze(partialI*Seqs_Add_buffer[itar,'I'][iq]),vedges)
                    rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler_I[itar]-=partialI*Seqs_Add_buffer[itar,'I'][iq]

                # 1st, delete some blocks in the buffer
                for idx in range(NPATCH):
                    # buffer deduction
                    Seqs_Add_buffer[idx,'E'][:integI+1]=[]
                    Seqs_Add_buffer[idx,'E'][0]*=(1-partialI)
                    Seqs_Add_buffer[idx,'I'][:integI+1]=[]
                    Seqs_Add_buffer[idx,'I'][0]*=(1-partialI)
                    # Also check the buffer with corresponding total 
                    diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler_E[idx])
                    diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler_I[idx])
                    # print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                # adding the extra excitation caused by the newest E spike
                isender=idx_cut_I
                # also add the newest sender
                # print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
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
                    rho_o[itar,:]=np.squeeze(rho_o_new)

                    # for I neuron
                    rho_ln_new=np.dot(np.squeeze(tfII[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    rho_ln_new[-1]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                LI[idx_cut_I]+=1
                print('648 Case 3-2, I spikes! E',LE,';I',LI)
            elif cut_E==cut_I: # I will be closer
                
                LE[idx_cut_E]+=1
                LI[idx_cut_I]+=1
                deltavI=np.sum(Seqs_Add_buffer[idx_cut_I,'I'][:integI+1])+partialI*Seqs_Add_buffer[idx_cut_I,'I'][integI+1]
                deltavI*=h
                if deltavI>Total_Add_scaler_I[idx_cut_I]:
                    print('potential I is not correct')
                # Then we could calculate the voltage sources been used
                deltavE=np.sum(Seqs_Add_buffer[idx_cut_E,'E'][:integE+1])+partialE*Seqs_Add_buffer[idx_cut_E,'E'][integE+1]
                deltavE*=h
                if deltavE>Total_Add_scaler_E[idx_cut_E]:
                    print('potential E is not correct')
                # calculate for other patches
                for iq in range(integI+1):
                    isender=Seqs_spikeE[iq]
                    # should be the entire bin
                    partial_i=0.0 # non, the first bin might be the remaining of the previous one
                    for itar in range(NPATCH):
                        # for E neuron
                        partial_i=Seqs_Add_buffer[itar,'E'][iq]/SEE[itar,isender]
                        if partial_i<1.0:
                            # recalculate
                            tfE = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'E'][iq]),vedges)
                        else:
                            tfE=tfEE[itar,isender,:,:]
                        rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                        prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                        flux_vt_E[itar]+=prob_missing
                        rho_o[itar,:]=np.squeeze(rho_o_new)
                        # delete these portion
                        Total_Add_scaler_E[itar]-=Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                        # for I neuron
                        partial_i=Seqs_Add_buffer[itar,'I'][iq]/SIE[itar,isender]
                        if partial_i<1.0:
                            # recalculate
                            tfI = EI_matrix_transfer_fix(np.squeeze(Seqs_Add_buffer[itar,'I'][iq]),vedges)
                        else:
                            tfI=tfIE[itar,isender,:,:]
                        rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                        prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                        flux_vt_I[itar]+=prob_missing
                        rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                        # subtract from the total value, and then change the buffer
                        Total_Add_scaler_I[itar]-=Seqs_Add_buffer[itar,'I'][iq]
                # now for the rest (portion)
                iq=integI+1
                isender=Seqs_spikeE[iq]
                partial_i=partialI*Seqs_Add_buffer[idx_cut_I,'I'][iq]/SIE[idx_cut_I,isender]
                partial_i_c=partialE*Seqs_Add_buffer[idx_cut_E,'E'][iq]/SEE[idx_cut_E,isender]
                assert(partial_i==partial_i_c)
                for itar in range(NPATCH):
                    # for E neuron
                    tfE = EI_matrix_transfer_fix(np.squeeze(partialE*Seqs_Add_buffer[itar,'E'][iq]),vedges)
                    rho_o_new=np.dot(tfE,np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    flux_vt_E[itar]+=prob_missing
                    rho_o[itar,:]=np.squeeze(rho_o_new)
                    # delete these portion
                    Total_Add_scaler_E[itar]-=partialE*Seqs_Add_buffer[itar,'E'][iq] # first, subtract this portion  # Total_Add_scaler[itar,'E']-=SEE[itar,isender]*partial_i 
                    # for I neuron
                    tfI = EI_matrix_transfer_fix(np.squeeze(partialE*Seqs_Add_buffer[itar,'I'][iq]),vedges)
                    rho_ln_new=np.dot(tfI,np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    flux_vt_I[itar]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new)
                    # subtract from the total value, and then change the buffer
                    Total_Add_scaler_I[itar]-=partialE*Seqs_Add_buffer[itar,'I'][iq]

                # 1st, delete some blocks in the buffer
                for idx in range(NPATCH):
                    # buffer deduction
                    Seqs_Add_buffer[idx,'E'][:integI+1]=[]
                    Seqs_Add_buffer[idx,'E'][0]*=(1-partialI)
                    Seqs_Add_buffer[idx,'I'][:integI+1]=[]
                    Seqs_Add_buffer[idx,'I'][0]*=(1-partialI)
                    # Also check the buffer with corresponding total 
                    diffE=np.abs(np.sum(Seqs_Add_buffer[idx,'E'])-Total_Add_scaler_E[idx])
                    diffI=np.abs(np.sum(Seqs_Add_buffer[idx,'I'])-Total_Add_scaler_I[idx])
                    # print('The difference between E and I are respective: E--',diffE,'; I--',diffI)
                # adding the extra excitation caused by the newest E spike, first
                isender=idx_cut_E
                # also add the newest sender
                # print('flux across the threshold, for E population that releasing an E spike:', flux_vt_E[idx_cut_E])
                flux_vt_E[idx_cut_E]=0.0
                # subtract E spikes (elevation) from Seqs_spikeE
                Seqs_spikeE[:integE+1]=[]
                Seqs_spikeE.append(idx_cut_E)
                Seqs_spikeEI.append(idx_cut_E)
                for itar in range(NPATCH):
                    Seqs_Add_buffer[itar,'E'].append(SEE[itar,isender])
                    Total_Add_scaler_E[itar]+=SEE[itar,isender]
                    Seqs_Add_buffer[itar,'I'].append(SIE[itar,isender])
                    Total_Add_scaler_I[itar]+=SIE[itar,isender]    
                #-------------- effects of E spikes----------------------------------------------------------
                # also suppression trigger by newest I spike, later
                isender=idx_cut_I
                # also add the newest sender
                # print('flux across the threshold, for I population that releasing an I spike:', flux_vt_I[idx_cut_I])
                flux_vt_I[idx_cut_I]=0.0
                Seqs_spikeEI.append(-idx_cut_I)
                # there is a new I spike and will suppress others (both excitatory and inhibitory population)
                for itar in range(NPATCH):
                    # for E neuron
                    rho_o_new=np.dot(np.squeeze(tfEI[itar,isender,:,:]),np.reshape(rho_o[itar,:],(nbins,1)))
                    prob_missing = (np.sum(rho_o[itar,:])-np.sum(rho_o_new))*h
                    rho_o_new[-1]+=prob_missing
                    rho_o[itar,:]=np.squeeze(rho_o_new)

                    # for I neuron
                    rho_ln_new=np.dot(np.squeeze(tfII[itar,isender,:,:]),np.reshape(rho_ln_o[itar,:],(nbins,1)))
                    prob_missing=(np.sum(rho_ln_o[itar,:])-np.sum(rho_ln_new))*h
                    rho_ln_new[-1]+=prob_missing
                    rho_ln_o[itar,:]=np.squeeze(rho_ln_new) 
                LE[idx_cut_E]+=1
                LI[idx_cut_I]+=1
                print('765 Case 3-3, E and I spikes! E',LE,';I',LI)
        cdf_o,cdf_ln_o=cal_CDF(vedges,rho_o,rho_ln_o)
        
    return (Seqs_spikeE,Seqs_spikeEI,LE,LI)


     



