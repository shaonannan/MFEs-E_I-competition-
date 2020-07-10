# This code is for finding out the transformations between multiple populations 
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

'''
The function calculates the population voltage distribution after inhibitory suppression 
'''
def suppression_matrix_transfer(weight,vedge,):
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
    
    retreat_flux = 1.0-overlapping.sum(axis=0)
    overlapping[-1,:]+=retreat_flux
    return overlapping

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
            print('144:',mapping[L][j])
    plt.imshow(mapping)
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
        for curr_zero_ind in zero_bin_ind_list:
            A[curr_zero_ind,:] += flux_to_zero_vector/len(zero_bin_ind_list)
    else:
        # Outflow:
        A = -np.eye(len(v)-1)*lam*p
        
        # Inflow:
        A += redistribute_probability_mass(v+w, v).T*lam*p
        
        
        missing_flux = -A.sum(axis=0)
        A[0,:] += missing_flux
        
        flux_to_zero_vector = np.zeros_like(A.sum(axis=0))

    return flux_to_zero_vector, A

def remap_EI_after_Exc(vedges,v_o,rho_o,rho_ln_o,S_EE,S_IE,S_EI,S_II,vT,vI,NE,NI,N_neuron_fire_first):
    ''' remap Excitatory populations'''
    # Find out which population fires
    NPATCH,vlen=np.shape(rho_o)
    idxf=np.where(N_neuron_fire_first>0)
    v=vedges[-1]-v_o[-1::-1]#v_o[-1]-v_o[-1::-1]
    h=v[1]-v[0]
    # transfer vedges
    vedges=vedges[-1]-vedges[-1::-1]
    rho,rho_ln=rho_o.copy(),rho_ln_o.copy()
    rho,rho_ln=rho[:,-1::-1],rho_ln[:,-1::-1]  
    cdf,cdf_ln=np.zeros_like(rho),np.zeros_like(rho_ln) 
    for ii in range(NPATCH):
        cdf[ii,:]=cal_CDF(v,rho[ii,:])
        cdf_ln[ii,:]=cal_CDF(v,rho_ln[ii,:])
    # S_EE[:,idxf]*N_neuron_fire_first
    tE,tI=S_EE[:,idxf]*N_neuron_fire_first[idxf],S_IE[:,idxf]*N_neuron_fire_first[idxf] 
    tE,tI=np.reshape(np.squeeze(tE),(2,1)),np.reshape(np.squeeze(tI),(2,1))
    print(S_EE[:,0],'fire:',N_neuron_fire_first[idxf],'60 tE:',tE)
    m_val=v[-1]
    ''' variables '''
    v_next,v_ln_next=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    v_inh,v_ln_inh=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    v_pre_push,v_ln_pre_push=np.zeros(NPATCH),np.zeros(NPATCH)

    f_pre,f_ln_pre=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    f_current,f_ln_current=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    f_next,f_ln_next=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    f_final,f_ln_final=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    f_pre_push,f_ln_pre_push=np.zeros(NPATCH),np.zeros(NPATCH)

    E_potentialSpks,E_Spks,I_Spks=np.zeros((1,NPATCH)),np.zeros((1,NPATCH)),np.zeros((1,NPATCH))

    # suppression matrix
    inh_matrixE,inh_matrixI = np.zeros((NPATCH,NPATCH,vlen,vlen)),np.zeros((NPATCH,NPATCH,vlen,vlen))
    for ii in range(NPATCH):
        for jj in range(NPATCH):
            inh_matrixE[ii,jj,:,:]=suppression_matrix_transfer(-S_EI[ii,jj],vedges)
            inh_matrixI[ii,jj,:,:]=suppression_matrix_transfer(-S_II[ii,jj],vedges)
            

    # remain_nn,remain_ln_nn=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    rho_new,rho_ln_new = np.zeros_like(rho_o),np.zeros_like(rho_ln_o)
    m_E,m_I=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    m_E_test={'p1':[],
            'p2':[]}
    m_I_test={'p1':[],
            'p2':[]}

    f_pre,f_ln_pre=np.zeros((1,NPATCH)),np.zeros((1,NPATCH))
    flag_stop=0
    for i in range(vlen):
        # calculate CDF
        vE_current,vI_current=tE[:,-1],tI[:,-1]
        # cdf for current E(fire)
        # print('current:',vE_current,'v:',v[:10])
        for ii in range(NPATCH):
            cdf[ii,:]=cal_CDF(v,rho[ii,:])
            cdf_ln[ii,:]=cal_CDF(v,rho_ln[ii,:])
        plt.figure(1)   
        plt.plot(v,cdf[0,:])
        print(cdf[0,-1])
        
        for ii in range(NPATCH):
            f_current[0,ii]=np.interp(vE_current[ii],v,cdf[ii,:],left=0.0,right=1.0)
            f_ln_current[0,ii]=np.interp(vI_current[ii],v,cdf_ln[ii,:],left=0.0,right=1.0)
            f_pre_push[ii]=np.interp(v_pre_push[ii],v,cdf[ii,:],left=0.0,right=1.0)
            f_ln_pre_push[ii]=np.interp(v_ln_pre_push[ii],v,cdf_ln[ii,:],left=0.0,right=1.0)
       # print('94: step',i,';info:',f_pre[0,0],f_current[0,0])
        # Excitatory effects, E-spikes can drive neurons all the way up to v_next to fire
        for ii in range(NPATCH):
            E_Spks[0,ii]=(f_current[0,ii]-f_pre[0,ii])*NE # determine v_next
            if E_Spks[0,ii] <0.0:
                E_Spks[0,ii]=0.0
            m_E[0,ii]+=E_Spks[0,ii]
        
        m_E_test['p1'].append(E_Spks[0,0])
        m_E_test['p2'].append(E_Spks[0,1])
        print('ESpks:',E_Spks)
        print('104E compare f_pre_unchanged and f_pre_push')
        print('f_pre_unchanged:',f_pre)
        print('f_pre_push:',f_pre_push)
        for ii in range(NPATCH):#receiver
            v_next[0,ii]=vE_current[ii]
            v_ln_next[0,ii]=vI_current[ii]
            for jj in range(NPATCH):#sender
                v_next[0,ii]+=S_EE[ii,jj]*E_Spks[0,jj]
                v_ln_next[0,ii]+=S_IE[ii,jj]*E_Spks[0,jj]

        # check whether there are sufficient neurons to be fired
        for ii in range(NPATCH):
            f_next[0,ii]=np.interp(v_next[0,ii],v,cdf[ii,:],left=0.0,right=1.0)
            f_ln_next[0,ii]=np.interp(v_ln_next[0,ii],v,cdf_ln[ii,:],left=0.0,right=1.0) # E trigger the next step
        # possibility
        for ii in range(NPATCH):
            E_potentialSpks[0,ii]=(f_next[0,ii]-f_current[0,ii])*NE
        if(np.max(E_potentialSpks)<1): # stop the cascaded chain
            flag_stop=1
            print('Breaking!!!')
            break
        tEnew,tInew=np.squeeze(v_next),np.squeeze(v_ln_next)
        tE=np.insert(tE,np.shape(tE)[1],values=tEnew,axis=1)
        tI=np.insert(tI,np.shape(tI)[1],values=tInew,axis=1)

        # suppose that Exc effects are faster than that of inhibitory, all these neurons can fire and the I-spikes can futher inhibit the remaining neurons(both E and I)
        # here, we consider the inhibitory effects.
        for ii in range(NPATCH):
            I_Spks[0,ii]=(f_ln_current[0,ii]-f_ln_pre[0,ii])*NI # have consequential effects
            m_I[0,ii]+=I_Spks[0,ii]

        m_I_test['p1'].append(I_Spks[0,0])
        m_I_test['p2'].append(I_Spks[0,1])
        print('ISpks:',I_Spks)
        print('137I compare f_pre_unchanged and f_pre_push')
        print('f_pre_unchanged:',f_ln_pre)
        print('f_pre_push:',f_ln_pre_push)
        # these I spikes further inhibit both E and I cells
        for ii in range(NPATCH):#receiver
            for jj in range(NPATCH):#sender
                v_inh[0,ii]+=S_EI[ii,jj]*I_Spks[0,jj]
                v_ln_inh[0,ii]+=S_II[ii,jj]*I_Spks[0,jj]
        '''
        # inhibition affects the remaining neurons        
        for ii in range(NPATCH):
            idx=np.where(v>vE_current[ii])
            print('117:',len(idx))
            if len(idx)<2: # to the end
                remain_nn[0,ii] = vlen
            elif idx[0][0]==0:
                remain_nn[0,ii] = 0
            else:
                remain_nn[0,ii] = idx[0][0]-1
            
            idx_ln=np.where(v>vI_current[ii])
            if len(idx_ln)<2: # to the end
                remain_ln_nn[0,ii] = vlen
            elif idx_ln[0][0]==0:
                remain_ln_nn[0,ii] = 0
            else:
                remain_ln_nn[0,ii] = idx_ln[0][0]-1
        '''
#        # inhibition delay
#        for ii in range(NPATCH):
#            vnew=v+v_inh[0,ii] # new x-axis
#            rho_new[ii,:]=np.interp(v,vnew,rho[ii,:],left=0,right=0)
#            total=np.sum(np.squeeze(rho_new[ii,:]))*h
#            rho_new[ii,-1]+=(1.0-total)/h
#            vnew_ln=v+v_ln_inh[0,ii]
#            rho_ln_new[ii,:]=np.interp(v,vnew_ln,rho_ln[ii,:],left=0,right=0)
#            # print('154:',v_ln_inh[0,ii],np.sum(rho_ln_new[ii,:])*h)
#            total=np.sum(np.squeeze(rho_ln_new[ii,:]))*h            
#            rho_ln_new[ii,-1]+=(1.0-total)/h
#            # print('156:',np.sum(rho_ln_new[ii,:])*h)
            
        # w should be negative
        # vedges have not been inversed.
        for ii in range(NPATCH):
            rho_new[ii,:]=rho[ii,:] 
            rho_ln_new[ii,:]=rho_ln[ii,:] 
            for jj in range(NPATCH):
                EI=np.squeeze(inh_matrixE[ii,jj,:,:])
                II=np.squeeze(inh_matrixI[ii,jj,:,:])
                ispk=int(I_Spks[0,jj])
                for ispk in range(ispk):
                    tmp=np.dot(EI,np.reshape(rho_new[ii,:],(vlen,1)))
                    rho_new[ii,:]=np.squeeze(tmp)
                    tmpln=np.dot(II,np.reshape(rho_ln_new[ii,:],(vlen,1)))
                    rho_ln_new[ii,:]=np.squeeze(tmpln)
            # print('156:',np.sum(rho_ln_new[ii,:])*h)
        rho,rho_ln=rho_new.copy(),rho_ln_new.copy()    
        f_pre,f_ln_pre=f_current.copy(),f_ln_current.copy()    
        # or use v_cur_push
        print('Unseeing!!!')
        v_pre_push,v_ln_pre_push = vE_current+v_inh[0,:], vI_current+v_ln_inh[0,:]      
                     
    cutoffv,cutoffv_ln=tE[:,-1],tI[:,-1]
    if flag_stop==1:
        for ii in range(NPATCH):
            I_Spks[0,ii]=(f_ln_current[0,ii]-f_ln_pre[0,ii])*NI # have consequential effects
            m_I[0,ii]+=I_Spks[0,ii]   
            
    print('cutoff value:',cutoffv)
#    for ii in range(NPATCH):
#        f_final[0,ii]=np.interp(cutoffv[ii],v,cdf[ii,:])
#        f_ln_final[0,ii]=np.interp(cutoffv_ln[ii],v,cdf_ln[ii,:])
#        m_E_test[0,ii]=f_final[0,ii]*NE
#        m_I_test[0,ii]=f_ln_final[0,ii]*NI
        
    return(tE,tI,m_E,m_I,m_E_test,m_I_test)

# parameters
nbins=2000
vedges=np.linspace(-1.0,1.0,nbins+1)
v_o=0.5*(vedges[0:-1] + vedges[1:]) 
vT = 1.0
# _,Adipde=flux_matrix(v_o, -0.003, 1.0, p=1)
# v=v_o[-1]-v_o[-1::-1]
# Asyx=suppression_matrix_transfer(-0.003,v)
NPATCH=2
rho_o,rho_ln_o=np.zeros((NPATCH,nbins)),np.zeros((NPATCH,nbins))
for ii in range(NPATCH):
#    rho_o[ii,:],_=generate_voltage_dist([0.98,0.01],v_o)
    rho_o[ii,:],_=generate_voltage_dist([0.93,0.02],v_o)
    rho_ln_o[ii,:],_=generate_voltage_dist([0.85,0.05],v_o)
NE,NI=128,128
DEE = np.zeros((NPATCH, NPATCH))
DEI = np.zeros((NPATCH, NPATCH))
DII = np.zeros((NPATCH, NPATCH))
DIE = np.zeros((NPATCH, NPATCH))
for ii in range(NPATCH):
    DEE[ii,ii]=0.012#0.008
    DIE[ii,ii]=0.010
    DEI[ii,ii]=0.010
    DII[ii,ii]=0.007
#DEE[0,1],DEE[1,0],DIE[0,1],DIE[1,0]=0.005,0.005,0.002,0.002
DEE[0,1],DEE[1,0],DIE[0,1],DIE[1,0]=0.002,0.002,0.002,0.002

v=vedges[-1]-v_o[-1::-1]
vedges_b=vedges[-1]-vedges[-1::-1]
h=v[1]-v[0]
testE=redistribute_probability_mass(vedges_b-DEE[1,1],vedges_b)
testI=redistribute_probability_mass(vedges_b+DEI[1,1],vedges_b)

#N_neuron_fire_first=np.array([2,0])
#
#tE,tI,mE,mI,mEt,mIt=remap_EI_after_Exc(vedges,v_o,rho_o,rho_ln_o,DEE,DIE,DEI,DII,vT,-1,NE,NI,N_neuron_fire_first)
#print('m:',mE,mI)