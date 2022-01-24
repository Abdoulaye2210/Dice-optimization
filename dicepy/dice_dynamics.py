import csv
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
seaborn.set(style='ticks')

###############################################################################
# All arrays have been shifted to have length numtimes + 1 and to start at
# 1 in line with matlab code
###############################################################################
# Due to my essential wish to use the same code for the optimiser and to get
# the state info I have created two functions, one wrapped inside the other.
# The optimiser call fOBJ which returns a float for the optimiser. However
# internally this calls the simulateDynamics function which can return either
# the single value (utility to be maximised) or the state information.
###############################################################################


@njit(cache=True, fastmath=True)
def objFn(x, *args):
    """ This is the pass-through function that returns a single float value of
    the objective function for the benefit of the optimisation algorithm. """

    out = simulateDynamics(x, *args)
    return out[0,0]

###############################################################################

@njit(cache=True, fastmath=True)
def simulateDynamics(x, sign, outputType, num_times, num_paths, 
                     tstep, al, ll, sigma, cumetree, forcoth,
                     cost1, etree,
                     scale1, scale2,
                     ml0, mu0, mat0, cca0,
                     a1, a2, a3,
                     c1, c3, c4,
                     b11, b12, b21, b22, b32, b23, b33,
                     fco22x, t2xco2, rr, gama,
                     tocean0, tatm0, elasmu, prstp, expcost2,
                     k0, dk, pbacktime, 
                     scc_period, e_bump, c_bump):
    """ This is the simulation of the DICE 2016 model dynamics. It is optimised
    for speed. For this reason I have avoided the use of classes. """
    
    ##########################################################################  

    #########################################################################
#    print(scc_period, e_bump, c_bump)
    
    LOG2 = np.log(2)
    L = ll  # NORDHAUS RENAMES IT TO UPPER CASE IN EQUATIONS
    MILLE = 1000.0

    # We take care to ensure that the indexing starts at 1 to allow comparison
    # with matlab
    MIUopt = np.zeros(num_times+1)
    Sopt = np.zeros(num_times+1)

    ###########################################################################
    # Set the optimisation variables
    ###########################################################################

    for i in range(1, num_times+1):
        MIUopt[i] = x[i-1]
        Sopt[i] = x[num_times + i-1]
        
    ###########################################################################
    # We redefine our parameters: 
    
    d_al = np.ones((len(al), num_paths))
    for i in range(1,num_times+1):
      d_al[i,] = al[i]*d_al[i,]
      
    d_L = np.ones((len(L), num_paths))
    for i in range(1,num_times+1):
      d_L[i,] = L[i]*d_L[i,]
        
    d_sigma = np.ones((len(sigma), num_paths))
    for i in range(1,num_times+1):
      d_sigma[i,] = sigma[i]*d_sigma[i,]    
      
    d_cumetree = np.ones((len(cumetree), num_paths))
    for i in range(1,num_times+1):
      d_cumetree[i,] = cumetree[i]*d_cumetree[i,]  
      
    d_forcoth = np.ones((len(forcoth), num_paths))
    for i in range(1,num_times+1):
      d_forcoth[i,] = forcoth[i]*d_forcoth[i,]    
      
    d_cost1 = np.ones((len(cost1), num_paths))
    for i in range(1,num_times+1):
      d_cost1[i,] = cost1[i]*d_cost1[i,]    
      
    d_etree = np.ones((len(etree), num_paths))
    for i in range(1,num_times+1):
      d_etree[i,] = etree[i]*d_etree[i,]    
      
    d_rr = np.ones((len(rr), num_paths))
    for i in range(1,num_times+1):
      d_rr[i,] = rr[i]*d_rr[i,]  
      
    d_sigma = np.ones((len(sigma), num_paths))
    for i in range(1,num_times+1):
      d_sigma[i,] = sigma[i]*d_sigma[i,]    
      
    d_pbacktime = np.ones((len(pbacktime), num_paths))
    for i in range(1,num_times+1):
      d_pbacktime[i,] = pbacktime[i]*d_pbacktime[i,]    
  
    d_MIUopt = np.ones((len(MIUopt), num_paths))
    for i in range(1,num_times+1):
      d_MIUopt[i,] = MIUopt[i]*d_MIUopt[i,] 
      
    d_Sopt = np.ones((len(Sopt), num_paths))
    for i in range(1,num_times+1):
      d_Sopt[i,] = Sopt[i]*d_Sopt[i,] 

###########################################################################

    FORC = np.zeros((num_times+1,num_paths))
    TATM = np.zeros((num_times+1,num_paths))
    TOCEAN = np.zeros((num_times+1,num_paths))
    MAT = np.zeros((num_times+1,num_paths))
    MU = np.zeros((num_times+1,num_paths))
    ML = np.zeros((num_times+1,num_paths))
    E = np.zeros((num_times+1,num_paths))
    EIND = np.zeros((num_times+1,num_paths))
    C = np.zeros((num_times+1,num_paths))
    K = np.zeros((num_times+1,num_paths))
    CPC = np.zeros((num_times+1,num_paths))
    II = np.zeros((num_times+1,num_paths))
    RI = np.zeros((num_times+1,num_paths))
    Y = np.zeros((num_times+1,num_paths))
    YGROSS = np.zeros((num_times+1,num_paths))
    YNET = np.zeros((num_times+1,num_paths))
    DAMAGES = np.zeros((num_times+1,num_paths))
    DAMFRAC = np.zeros((num_times+1,num_paths))
    ABATECOST = np.zeros((num_times+1,num_paths))
    MCABATE = np.zeros((num_times+1,num_paths))
    CCA = np.zeros((num_times+1,num_paths))
    CCATOT = np.zeros((num_times+1,num_paths))
    PERIODU = np.zeros((num_times+1,num_paths))
    CPRICE = np.zeros((num_times+1,num_paths))
    CEMUTOTPER = np.zeros((num_times+1,num_paths))
    
    # Fixed initial values
    CCA[1,] = cca0*np.ones(num_paths)
    K[1,] = k0*np.ones(num_paths)
    MAT[1,] = mat0*np.ones(num_paths)
    MU[1,] = mu0*np.ones(num_paths)
    ML[1,] = ml0*np.ones(num_paths)
    TATM[1,] = tatm0*np.ones(num_paths)
    TOCEAN[1,] = tocean0*np.ones(num_paths)

    #YGROSS[1,] = d_al[1,]*( np.power( d_L[1,]/MILLE, 1.0-gama)  ) * (np.power(K[1,], gama))


    ##################################################################### GBM
    S_0 =1 # 
    T_ = 1 # 
    N_ = 100 # 
    dt = T_ / N_  
    sigma_ = 1
    drift = np.zeros((num_times+1, num_paths))
    diffusion = np.zeros((num_times+1,num_paths))
    S_  = np.zeros((num_times+1,num_paths))
    t = np.linspace(0, 1, N_+1)
    W1 = np.random.standard_normal(size = round(num_paths/2))
    W = np.concatenate((W1,-W1)) 
    W = np.cumsum(W*np.sqrt(dt))
    drift[1,] = np.log( np.multiply(d_al[1,] , np.power((d_L[1,]/MILLE),(1.0-gama)) , np.power(K[1,] ,gama) )) - 0.5*sigma_**2*t[1]*np.ones(num_paths)      
    diffusion[1,] = sigma_*W
    YGROSS[1,] = S_0*np.exp(drift[1,] + diffusion[1,])

###############################################################################

    EIND[1,] = np.multiply( d_sigma[1,] , YGROSS[1,] ,  np.ones(num_paths)- d_MIUopt[1,] )
    
    E[1,] = EIND[1,] + d_etree[1,]

    if scc_period == 1:
    
       E[1,] = E[1,] + e_bump*np.ones(num_paths)

    CCATOT[1,] = CCA[1,] + d_cumetree[1,]
    
    FORC[1,] = fco22x * np.log(MAT[1,]/588.0)/LOG2 + d_forcoth[1,]
    
    DAMFRAC[1,] = a1*TATM[1,] + a2*TATM[1,]**a3
    
    DAMAGES[1,] = np.multiply( YGROSS[1,] , DAMFRAC[1,] )
    
    ABATECOST[1,] = np.multiply( YGROSS[1,] , d_cost1[1,] , np.power(d_MIUopt[1,] , expcost2))
    
    MCABATE[1,] = np.multiply( d_pbacktime[1,] , np.power(d_MIUopt[1,] , expcost2-1))
    
    CPRICE[1,] = np.multiply( d_pbacktime[1,] , np.power(d_MIUopt[1,] , expcost2-1) )

    YNET[1,] = np.multiply( YGROSS[1,] , (np.ones(num_paths) - DAMFRAC[1,]))
    
    Y[1,] = YNET[1,] - ABATECOST[1,]
    
    II[1,] = np.multiply ( d_Sopt[1,] , Y[1,] )
    
    C[1,] = Y[1,] - II[1,]

    if scc_period == 1:
        C[1,] = C[1,] + c_bump*np.ones(num_paths)

    CPC[1,] = MILLE * np.divide( C[1,] , d_L[1,])
       # RI[T,] is set at end

    PERIODU[1,] = (   np.power( np.divide(C[1,]*MILLE, d_L[1,]) , (1.0-elasmu)) - np.ones(num_paths)) / (1.0 - elasmu) - np.ones(num_paths)
    
    CEMUTOTPER[1,] = np.multiply (PERIODU[1,] , d_L[1,] , d_rr[1,])

    # Reference 
    # http://www.econ.yale.edu/~nordhaus/homepage/homepage/DICE2016R-091916ap.gms

    eta = (fco22x/t2xco2)

    for i in range(2, num_times+1):
            
        # Depend on t-1
        CCA[i,] = CCA[i-1,] + EIND[i-1,] * (5.0 / 3.666)
        

        MAT[i,] = np.maximum(10.0*np.ones(num_paths), MAT[i-1,] * b11 + MU[i-1,] * b21 + E[i-1,] * (5.0 / 3.666))
        
        
        MU[i,] = np.maximum(100.0*np.ones(num_paths), MAT[i-1,] * b12 + MU[i-1,] * b22 + ML[i-1,] * b32)
        
        ML[i,] = np.maximum(1000.0*np.ones(num_paths), ML[i-1,] * b33 + MU[i-1,] * b23)

        TOCEAN[i,] = np.maximum(-1.0*np.ones(num_paths), TOCEAN[i-1,] + c4 * (TATM[i-1,] - TOCEAN[i-1,]))
        
        TOCEAN[i,] = np.minimum(20.0*np.ones(num_paths), TOCEAN[i,])
        
        CCATOT[i,] = CCA[i,] + d_cumetree[i,]

        # Depend on t
        K[i,] = np.maximum(1.0*np.ones(num_paths), (1.0-dk)**tstep * K[i-1,] + tstep * II[i-1,])

        W1 = np.random.standard_normal(size = round(num_paths/2))
        W = np.concatenate((W1,-W1)) 
        W = np.cumsum(W*np.sqrt(dt))
        drift[i,] = np.log(  np.multiply(d_al[i,] , np.power((d_L[i,]/MILLE),(1.0-gama)) , np.power(K[i,] ,gama) )) - 0.5*sigma_**2*t[1]*np.ones(num_paths)      
        diffusion[i,] = sigma_ * W
        YGROSS[i,] = S_0*np.exp(drift[i,] + diffusion[i,])

        #YGROSS[i,] = al[i] * ((L[i]/MILLE)**(1.0 - gama)) * K[i,]**gama
        
        EIND[i,] = np.multiply( d_sigma[i,] , YGROSS[i,] , (1.0*np.ones(num_paths) - d_MIUopt[i,]) )

        E[i,] = EIND[i,] + d_etree[i,]

        if scc_period == i:
            E[i,] = E[i,] + e_bump*np.ones(num_paths)

        FORC[i,] = fco22x * np.log(MAT[i,]/588.000)/LOG2 + d_forcoth[i,]
        
        TATM[i,] = TATM[i-1,] + c1 * (FORC[i,] - eta * TATM[i-1,] - c3 * (TATM[i-1,] - TOCEAN[i-1,]))
        
        TATM[i,] = np.maximum(-1.0*np.ones(num_paths), TATM[i,])
        
        TATM[i,] = np.minimum(12.0*np.ones(num_paths), TATM[i,]) # Nordhaus has this twice at 12 and 20 so I use the 20

        DAMFRAC[i,] = a1 * TATM[i,] + a2*(  np.power(TATM[i,] ,a3))
        
        DAMAGES[i,] = np.multiply( YGROSS[i,] , DAMFRAC[i,] )

        ABATECOST[i,] = np.multiply( YGROSS[i,] , d_cost1[i,] ,  np.power(d_MIUopt[i,] , expcost2))
                
        MCABATE[i,] = np.multiply( d_pbacktime[i,] , np.power (d_MIUopt[i,] , expcost2-1))
        
        CPRICE[i,] = np.multiply ( d_pbacktime[i,] , np.power (d_MIUopt[i,] , expcost2-1))

        YNET[i,] = np.multiply ( YGROSS[i,] , 1.0*np.ones(num_paths) - DAMFRAC[i,]) 
        
        Y[i,] = YNET[i,] - ABATECOST[i,]

        II[i,] = np.multiply ( d_Sopt[i,] , Y[i,] )
        
        C[i,] = np.maximum(2.0*np.ones(num_paths), Y[i,] - II[i,])
 
        if scc_period == i:
           C[i,] = C[i,] + c_bump*np.ones(num_paths)

        CPC[i,] = np.maximum(0.01*np.ones(num_paths), MILLE * np.divide( C[i,] , d_L[i,] )) 
        

        #PERIODU[i,] = ((C[i]*MILLE/L[i])**(1.0-elasmu) - 1.0) / \ (1.0 - elasmu) - 1.0

        PERIODU[i,] = (np.power( np.divide(C[i,]*MILLE, d_L[i,]) , (1.0-elasmu)) - np.ones(num_paths)) / (1.0 - elasmu) - np.ones(num_paths)

        CEMUTOTPER[i,] = np.multiply ( PERIODU[i,] , d_L[i,] , d_rr[i,] )
    for i in range(1, num_times):

          RI[i,] = (1.0 + prstp) * np.power( np.divide(CPC[i+1,] , CPC[i,]) , (elasmu/tstep)) - 1.0*np.ones(num_paths)

    RI[-1,] = np.zeros(num_paths)

    #output = np.array((num_times, 50))

    #if outputType == 0:
     #   resUtility = np.zeros(num_paths)
        #for i in range(1,num_times+1):
      #  resUtility = tstep * scale1 * np.sum(CEMUTOTPER) + scale2*np.ones(num_paths)
       # resUtility *= sign
        #output = [0,0]

           #print(output[0,0])
        #return output
        #print(output)

    if outputType == 1:

        # EXTRA VALUES COMPUTED LATER
         CO2PPM = np.zeros((num_times+1, num_paths))
         for i in range(1, num_times+1):
            CO2PPM[i,] = MAT[i,] / 2.13

         SOCCC = np.zeros((num_times+1, num_paths))
         for i in range(1, num_times+1):
            SOCCC[i,] = -999.0*np.ones(num_paths)

         output0 = np.zeros((num_times+1, num_paths))
         output1 = np.zeros((num_times+1, num_paths))
         output2 = np.zeros((num_times+1, num_paths))
         output3 = np.zeros((num_times+1, num_paths))
         output4 = np.zeros((num_times+1, num_paths))
         output5 = np.zeros((num_times+1, num_paths))
         output6 = np.zeros((num_times+1, num_paths))
         output7 = np.zeros((num_times+1, num_paths))
         output8 = np.zeros((num_times+1, num_paths))
         output9 = np.zeros((num_times+1, num_paths))
         output10 = np.zeros((num_times+1, num_paths))
         output11= np.zeros((num_times+1, num_paths))
         output12= np.zeros((num_times+1, num_paths))
         output13 = np.zeros((num_times+1, num_paths))
         output14 = np.zeros((num_times+1, num_paths))
         output15 = np.zeros((num_times+1, num_paths))
         output16 = np.zeros((num_times+1, num_paths))
         output17= np.zeros((num_times+1, num_paths))
         output18 = np.zeros((num_times+1, num_paths))
         output19= np.zeros((num_times+1, num_paths))
         output20= np.zeros((num_times+1, num_paths))
         output21 = np.zeros((num_times+1, num_paths))
         output22 = np.zeros((num_times+1, num_paths))
         output23 = np.zeros((num_times+1, num_paths))
         output24 = np.zeros((num_times+1, num_paths))
         output25= np.zeros((num_times+1, num_paths))
         output26= np.zeros((num_times+1, num_paths))
         output27= np.zeros((num_times+1, num_paths))
         output28= np.zeros((num_times+1, num_paths))
         output29= np.zeros((num_times+1, num_paths))
         output30= np.zeros((num_times+1, num_paths))

         for iTime in range(1, num_times+1):
            col = 0
            jTime = iTime - 1
            output0[jTime, ] = EIND[iTime,]
            col += 1  # 0
            output1[jTime, ] = E[iTime,]
            col += 1  # 1
            output2[jTime, ] = CO2PPM[iTime]
            col += 1  # 2
            output3[jTime, ] = TATM[iTime,]
            col += 1  # 3
            output4[jTime, ] = Y[iTime,]
            col += 1  # 4
            output5[jTime, ] = DAMFRAC[iTime,]
            col += 1  # 5
            output6[jTime, ] = CPC[iTime,]
            col += 1  # 6
            output7[jTime, ] = CPRICE[iTime,]
            col += 1  # 7
            output8[jTime, ] = MIUopt[iTime]
            col += 1  # 8
            output9[jTime, ] = RI[iTime,]
            col += 1  # 9
            output10[jTime, ] = SOCCC[iTime]
            col += 1  # 10

            output11[jTime, ] = ll[iTime]
            col += 1  # 11
            output12[jTime, ] = al[iTime]
            col += 1  # 12
            output13[jTime, ] = YGROSS[iTime,]
            col += 1  # 13

            output14[jTime, ] = K[iTime,]
            col += 1  # 14
            output15[jTime, ] = Sopt[iTime]
            col += 1  # 15
            output16[jTime, ] = II[iTime,]
            col += 1  # 16
            output17[jTime, ] = YNET[iTime,]
            col += 1  # 17

            output18[jTime, ] = CCA[iTime,]
            col += 1  # 18
            output19[jTime, ] = CCATOT[iTime,]
            col += 1  # 19
            output20[jTime, ] = ML[iTime,]
            col += 1  # 20
            output21[jTime, ] = MU[iTime,]
            col += 1  # 21
            output22[jTime, ] = FORC[iTime,]
            col += 1  # 22
            output23[jTime, ] = TOCEAN[iTime,]
            col += 1  # 23
            output24[jTime, ] = DAMAGES[iTime,]
            col += 1  # 24
            output25[jTime, ] = ABATECOST[iTime,]

            output =  [output0, output1, output2, output3, output4, output5, output6, output7, output8, output9,  output10, output11, output12, output13, output14, output15, output16, output17, output18, output19, output20, output21, output22,  output23, output24, output25, np.zeros((num_times+1,num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1,num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths)), np.zeros((num_times+1, num_paths))]
         return output 
    else:
        raise Exception("Unknown output type.")

    return output

