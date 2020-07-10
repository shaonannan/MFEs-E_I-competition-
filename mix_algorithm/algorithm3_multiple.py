# This code is for finding out the transformations between multiple populations 
import numpy as np
from scipy import special
import itertools
import algorithm_3_utile as util
import matplotlib.pylab as plt

def generate_voltage_dist(moment,V):
    Vs,D=moment[0],moment[1]
    Rv = np.copy(V)
    (vT,vR) = (1.0,0.0)
    tmpg = np.greater(V,vR)
    indp = (np.where(tmpg))
    sqrtD  = np.sqrt(D)
    np.seterr(all='ignore')
    try:
        intovT  = special.dawsn((vT-Vs)/sqrtD)*np.exp((vT-Vs)**2/D)
        intovSD = special.dawsn(-Vs/sqrtD)*np.exp(Vs**2/D)
        Rv[indp[0][:]] = -special.dawsn((V[indp[0][:]]-Vs)/sqrtD)+np.exp(-(V[indp[0][:]]-Vs)**2/D)*intovT
        
        if(indp[0][0]>1):
            Rv[0:indp[0][0]] = np.exp(-np.square(V[0:indp[0][0]]-Vs)/D)*(-intovSD + intovT)
        tmpl = np.less(V,-2.0/3.0)
        indp = np.where(tmpl)
        Rv[indp[0][:]] = 0.0  
        sum_c = (V[2]-V[1])*np.sum(Rv)
        Rv = Rv/sum_c
    except:
        sum_c = (V[2]-V[1])*np.sum(Rv)
        Rv = Rv/sum_c        
    return (Rv,sum_c)

def cal_CDF(v,rho):
    h=v[1]-v[0]
    cdf=np.zeros_like(rho)
    for i in range(len(v)):
        cdf[i]=h*np.sum(rho[:i])
    # normalization
    cdf=cdf/cdf[-1]
    return cdf

#'''
#The function calculates the population voltage distribution after inhibitory suppression 
#'''
#def EI_matrix_transfer_fix(weight,vedge,):
#    nedge=len(vedge)
#    v=vedge.copy()
#    h=v[1]-v[0]
#    # weight should be negative (for suppression)
#    v_post=v-(weight)
#    overlapping=np.eye(nedge-1)
#    # v_post is the ref
#    # transfer from v to v_post
#    for i in range(nedge-1): # v_post[i,i+1]
#        cutL,cutR=0,0
#        while (cutL<nedge-1) and (v[cutL]<v_post[i]):
#            cutL+=1
#        if cutL>0:
#            cutL-=1
#        cutR=cutL
#        while (cutR<nedge-1) and (v[cutR]<v_post[i+1]):
#            cutR+=1
#        if cutR>=nedge:
#            cutR=nedge-1
#        # determine each individual overlap
#        # 'interval probability' in v_post[i,i+1] should be fully allocated to v-bins in v(standard)
#        vstd_start,vstd_end = v_post[i],v_post[i+1]
#        for j in range(cutL,cutR): # including cutL, excluding cutR
#            # v[j,j+1] how much in percentage does v_post[i,i+1] allocate in this voltage bin?
#            if v[j+1]<=vstd_start:
#                continue
#            elif (v[j]<=vstd_start) and (v[j+1]<=vstd_end):
#                part=v[j+1]-vstd_start
#                percentage=part/h/1.0
#                overlapping[j][i]=percentage
#            elif (v[j]<=vstd_start) and (v[j+1]>vstd_end):
#                overlapping[j][i]=1.0
#            elif (v[j]>vstd_start) and (v[j]<=vstd_end) and (v[j+1]<=vstd_end):
#                overlapping[j][i] = h/h
#            elif (v[j]>vstd_start) and (v[j]<=vstd_end) and (v[j+1]>vstd_end):
#                overlapping[j][i] = (vstd_end-v[j])/h
#            elif (v[j]>vstd_end):
#                continue
#        # overlapping=overlapping.T
#    # # if the elements are negative, there should be retreat _flux and they are supposed to be allocated in the resting state(the minimum)
#    # if weight<0.0:
#    #     retreat_flux = 1.0-overlapping.sum(axis=0)
#    #     overlapping[-1,:]+=retreat_flux
#    return overlapping

def flux_dynamical(transferM, w, lam, p=1):
    'Compute a flux matrix for voltage bins v, weight w, firing rate lam, and probability p.'
        
    # fire and discard:
    if w > 0:
        
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += transferM*lam*p

        # Threshold:
        flux_to_zero_vector = -A.sum(axis=0)
        # for curr_zero_ind in zero_bin_ind_list:
        #     A[curr_zero_ind,:] += flux_to_zero_vector/len(zero_bin_ind_list)
    else:
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += transferM*lam*p
        
        
        missing_flux = -A.sum(axis=0)
        A[0,:] += missing_flux
        
        flux_to_zero_vector = np.zeros_like(A.sum(axis=0))

    return flux_to_zero_vector, A

def fraction_overlap(a1, a2, b1, b2):
    '''Calculate the fractional overlap between range (a1,a2) and (b1,b2).
    
    Used to compute a reallocation of probability mass from one set of bins to
    another, assuming linear interpolation.
    '''
    if a1 >= b1:    # range of A starts after B starts
        if a2 <= b2:    
            return 1       # A is within B
        if a1 >= b2:
            return 0       # A is after B
        # overlap is from a1 to b2
        return (b2 - a1) / (a2 - a1)
    else:            # start of A is before start of B
        if a2 <= b1:
            return 0       # A is completely before B
        if a2 >= b2:
            # B is subsumed in A, but fraction relative to |A|
            return (b2 - b1) / (a2 - a1)
        # overlap is from b1 to a2
        return (a2 - b1) / (a2 - a1) 

def redistribute_probability_mass(A, B):
    '''Takes two 'edge' vectors and returns a 2D matrix mapping each 'bin' in B
    to overlapping bins in A. Assumes that A and B contain monotonically increasing edge values.
    '''
    
    mapping = np.zeros((len(A)-1, len(B)-1))
    newL = 0
    newR = newL
    
    # Matrix is mostly zeros -- concentrate on overlapping sections
    for L in range(len(A)-1):
        
        # Advance to the start of the overlap
        while newL < len(B) and B[newL] < A[L]:
            newL = newL + 1
        if newL > 0:
            newL = newL - 1
        newR = newL
        
        # Find end of overlap
        while newR < len(B) and B[newR] < A[L+1]:
            newR = newR + 1
        if newR >= len(B):
            newR = len(B) - 1

        # Calculate and store remapping weights
        for j in range(newL, newR):
            mapping[L][j] = fraction_overlap(A[L], A[L+1], B[j], B[j+1])

    return mapping

    
def flux_matrix(v, w, lam, p=1):
    'Compute a flux matrix for voltage bins v, weight w, firing rate lam, and probability p.'
        
    # Flow back into zero bin:
    if w > 0:
        
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += redistribute_probability_mass(v+w, v).T*lam*p

        # Threshold:
        flux_to_zero_vector = -A.sum(axis=0)
        # for curr_zero_ind in zero_bin_ind_list:
        #     A[curr_zero_ind,:] += flux_to_zero_vector/len(zero_bin_ind_list)
    else:
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += redistribute_probability_mass(v+w, v).T*lam*p
        
        
        missing_flux = -A.sum(axis=0)
        A[0,:] += missing_flux
        
        flux_to_zero_vector = np.zeros_like(A.sum(axis=0))

    return flux_to_zero_vector, A

'''
# Network architecture and essential parameters (for structure)
# NPATCH, NE/I, CG_PATCH represents the network architecture
# numpy arrays for iterative calculation
#   Seqs_Add_buffer: '1','E','1','I','2','E','2','I'
#   Total_Add_scaler:
#   Seqs_subt_buffer:
#   
# transfer matrix:
#   4 sets, SEE/SEI/SIE/SII
'''
NPATCH,NE,NI=2,128,128
PATCH_IDX = np.arange(NPATCH)
Seqs_Add_buffer,Total_Add_scaler,Seqs_subt_buffer={},{},{}
for index, type in itertools.product(PATCH_IDX,['E','I']):
    Seqs_Add_buffer[index,type]=[]
    Total_Add_scaler[index,type]=0.0
    Seqs_subt_buffer[index,type]=[]

# connectivity matrices
SEE=np.zeros((NPATCH,NPATCH))
SIE,SEI,SII=np.zeros_like(SEE),np.zeros_like(SEE),np.zeros_like(SEE)
SEE = np.zeros((NPATCH, NPATCH))
SEI = np.zeros((NPATCH, NPATCH))
SII = np.zeros((NPATCH, NPATCH))
SIE = np.zeros((NPATCH, NPATCH))
for ii in range(NPATCH):
    SEE[ii,ii]=0.012
    SIE[ii,ii]=0.010
    SEI[ii,ii]=0.010
    SII[ii,ii]=0.007
# SEE[0,1],SEE[1,0],SIE[0,1],SIE[1,0]=0.002,0.002,0.002,0.002
# give voltage edges and bins
nbins=2000
vedges=np.linspace(-1.0,1.0,nbins+1)
v_o=0.5*(vedges[0:-1] + vedges[1:]) 
vT = 1.0
# 1st step is to obtain initial voltage distribution for both E and I population
rho_o,rho_ln_o=np.zeros((NPATCH,nbins)),np.zeros((NPATCH,nbins))
for ii in range(NPATCH):
    rho_o[ii,:],_=generate_voltage_dist([0.98,0.02],v_o)
    rho_o[ii,:] = rho_o[ii,-1::-1]
    rho_ln_o[ii,:],_=generate_voltage_dist([0.90,0.05],v_o)
    rho_ln_o[ii,:] = rho_ln_o[ii,-1::-1]
# transfer to backward-order
v=vedges[-1]-v_o[-1::-1]
vedges_b=vedges[-1]-vedges[-1::-1]
h=v[1]-v[0]
# calculate transfer function
tfEE=np.zeros((NPATCH,NPATCH,nbins,nbins))
tfEI,tfIE,tfII=np.zeros_like(tfEE),np.zeros_like(tfEE),np.zeros_like(tfEE)
idx_range=np.arange(NPATCH)
for idxi,idxj in itertools.product(idx_range,idx_range):
    tfEE[idxi,idxj,:,:]=util.EI_matrix_transfer_fix(SEE[idxi,idxj],vedges_b,)
    tfEI[idxi,idxj,:,:]=util.EI_matrix_transfer_fix(-SEI[idxi,idxj],vedges_b,)
    tfIE[idxi,idxj,:,:]=util.EI_matrix_transfer_fix(SIE[idxi,idxj],vedges_b,)
    tfII[idxi,idxj,:,:]=util.EI_matrix_transfer_fix(-SII[idxi,idxj],vedges_b,)

tfEE_t=np.zeros((NPATCH,NPATCH,nbins,nbins))
tfEI_t,tfIE_t,tfII_t=np.zeros_like(tfEE_t),np.zeros_like(tfEE_t),np.zeros_like(tfEE_t)
for idxi,idxj in itertools.product(idx_range,idx_range):
    _,tfEE_t[idxi,idxj,:,:]=flux_matrix(vedges_b,-SEE[idxi,idxj],1)
    _,tfEI_t[idxi,idxj,:,:]=flux_matrix(vedges_b,SEI[idxi,idxj],1)
    _,tfIE_t[idxi,idxj,:,:]=flux_matrix(vedges_b,-SIE[idxi,idxj],1)
    _,tfII_t[idxi,idxj,:,:]=flux_matrix(vedges_b,SII[idxi,idxj],1)
    

#start to use some sort of "analytical approach, EI balance" to generate these sequences of EI spikes.
N_neuron_fire_first=np.array([2,0])
spksE,spksT,LE,LI=util.algrithm_3(NPATCH,NE,NI,v,vedges_b,vT,-1.0,rho_o,rho_ln_o, SEE,SIE,SEI,SII,tfEE,tfIE,tfEI,tfII,N_neuron_fire_first,)
#input parameters and variables (rho_o,rho_ln_o,SEE,SIE,SEI,SII,tf..,N_neuron_fire_first,Seqs_Add_buffer,Total_Add_scaler[index,type],    Seqs_subt_buffer[index,type]=[])

