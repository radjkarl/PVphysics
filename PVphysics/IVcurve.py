"""
TODO: MOD DESCRIPTION

@author: Chantal. 21/7/16
@author: Karl. 04/01/2016
"""
#currently nomenclature dictates a==n, ideality

import numpy as np
from scipy.optimize import newton
from scipy.optimize import curve_fit


k = 1.38E-23
q = 1.602E-19



def setparameters(G, T, Iph_stc, I0_stc, Rs, Rsh_stc, a_stc, b_stc=0, m_stc=0, Vbr_stc=None):
    """adjusts certain parameters from STC to given conditions"""
    #only relevant to making sim-data
    k=8.167E-5 #in eV
    Eg = 1.1*(1-0.0002677*(T-298))

    Iph = Iph_stc * G/1000 #adjusting Iph with irradiance
    I0 = I0_stc * (T/298)**3 * np.exp((1/k)*((1.1/298)-(Eg/T)))
    Rsh = Rsh_stc * G/1000 #adjust with G only, see notes
    a = a_stc * T/298 # adjust a with temp
    b=b_stc
    m=m_stc
    Vbr=Vbr_stc
    parameters ={'Iph': Iph, 'I0': I0, 'Rs': Rs, 'Rsh': Rsh, 
                 'a': a, 'b': b, 'm': m, 'Vbr': Vbr}     
    return parameters
    
    
def I_of_V(I,V,parameters):
    """function f(I,V)=0; the one-diode model"""
    I0, Iph, Rs, Rsh, a, b, m, Vbr = _unpackParam(parameters)
    func = Iph - I0*(np.exp((V+I*Rs)/a)-1) - ((V+I*Rs)/Rsh)*(
                        1+b*(1-((V+I*Rs)/Vbr))**(-m)) - I
    return func 


def _unpackParam(p):
    return p['I0'], p['Iph'], p['Rs'], p['Rsh'], p['a'], p['b'], p['m'], p['Vbr']
  
    
def I_of_VwrtV(V,I,parameters):
    """Same as above, only different independent variable, for newton"""
    return I_of_VwrtV2((V,I), *_unpackParam(parameters))

    
def I_of_VwrtV2(I_V_points,Iph,I0,Rs,Rsh,a,b,m,Vbr):
    """
    As above, only all parameters are individual scalars not a list, 
    for input to curve_fit
    """
    V,I = I_V_points
    funcv = Iph - I0*(np.exp((V+I*Rs)/a)-1) - ((
            V+I*Rs)/Rsh)*(1+b*(1-((V+I*Rs)/Vbr))**(-m)) - I
    return funcv 
  
    
def I_of_V_prime(I,V,parameters):
    """First derivative with respect to I"""
    I0, _, Rs, Rsh, a, b, m, Vbr = _unpackParam(parameters)
    funcp = -(I0*Rs/a)*np.exp((V+I*Rs)/a) - Rs/Rsh -1 - (
              b*Rs/Rsh)*((1-(V+I*Rs)/Vbr)**(-m))*(1+m/((Vbr/(V+I*Rs))-1))
    return funcp
    
    
def I_of_V_primewrtV(V,I,parameters):
    """First derivative wrt V"""
    I0 = parameters['I0']
    Rs = parameters['Rs']
    Rsh = parameters['Rsh']
    a = parameters['a']
    b = parameters['b']
    m = parameters['m']
    Vbr = parameters['Vbr']
    funcpv =  -(I0/a)*np.exp((V+I*Rs)/a) - 1/Rsh -(
                                b/Rsh)*((1-(V+I*Rs)/Vbr)**(-m))*(
                                1+m/((Vbr/(V+I*Rs))-1))
    return funcpv
  
    
def plotIV(I, V, Ifit=None):
    """
    Plots the curve with specific settings. Mostly used for sim-data
    """ 
    #import here, to save time when module loaded by other resources...
    import pylab as plt
    plt.plot(V,I, label='data')
    #plt.axis([-120,40,0,10])
    plt.ylim(0,I.max())
    plt.xlim(0,V[I>0].max())
    plt.xlabel('Voltage, (V)')
    
    plt.ylabel('Current, (A)')
    #plt.ylim(0,15)
    plt.title('I-V Fitting')
    if Ifit is not None:
        plt.plot(V,Ifit, 'g-',  label='fit')
    plt.legend()
    plt.show()
        
        
def solvecurve(parameters,Vspace):  
    """For given set of voltages Vspace, solve for a range of I"""
    Ispace = np.empty_like(Vspace)
    for n, V in enumerate(Vspace):
        Ispace[n] = newton(I_of_V, args = (V,parameters),x0 = 0, 
                           fprime=I_of_V_prime, maxiter = 1000)
    return Ispace   
 
    
def fitcurve(I_V_points, y_data,initial): 
    """curve fitting function, runs curve_fit with certain conditions"""
    #initial involves parameters Iph, I0, Rs, Rsh, a
    guess = curve_fit(I_of_VwrtV2, I_V_points, 
                      y_data, p0=initial, 
                      #I0->1, a->4
                    bounds=([0,0,0,0,0,0,0,-1000],
                            [10, 1E-2, 1E4, 1E4, 5.0, 1,100,0]), 
                      method='trf')
    return guess 
  
    
def chi_squared(Imeas, Icalc, sigma=1):
    """Definition for Ralph's form of chi-squared. Currently not used"""
    Imeas = np.array(Imeas)
    Icalc=np.array(Icalc)
    N = len(Imeas)
    if N != len(Icalc):
        raise Exception("The arrays are of different dimensions")
    X_squared = 1/N * np.sqrt(np.sum(((Imeas-Icalc)**2)/(sigma)))
    return X_squared    
   
    
def area_criterion(Vmeas, Imeas, Icalc):
    """
    Area Criterion error function
    """
    Vmeas=np.array(Vmeas)
    Imeas=np.array(Imeas)
    Icalc=np.array(Icalc)
    if len(Imeas) != len(Icalc) or len(Imeas) != len(Vmeas):
        raise Exception("The arrays are of different dimensions")    
    for i in range(0, (len(Imeas)-1)):
        del_I = Icalc[i]-Imeas[i+1]
        #print(del_I)
        if del_I ==0 or (del_I>0 and Icalc[i+1]-Imeas[i+1]<0) or (
                         del_I<0 and Icalc[i+1]-Imeas[i+1]>0):
            m=i
            break   
    al = np.abs((Vmeas[m+1]-Vmeas[m])*((Icalc[m]-Imeas[m+1])**2+(
                Icalc[m+1]-Imeas[m+2])**2)/(2*(np.abs(Icalc[m]-Imeas[m+1])+
                                               np.abs(Icalc[m+1]-Imeas[m+2]))))
    bet=0 
    for j in range(0, len(Imeas)-2):
        if j==m :
            continue
        else:
            tri = np.abs((((Icalc[j]-Imeas[j+1])+(
                            Icalc[j+1]-Imeas[j+2]))*(
                            Vmeas[j+1]-Vmeas[j]))/2)
            bet = bet + tri
    del_A = al+bet
    #print(del_A*100)
    E = del_A*100/np.trapz(Imeas,dx=0.1) #normalising function??
    return E
    
    
def fitting_algorithm(Ipoints, Vpoints):#, fixed_ideality=0):
    """
    Estimates parameter using one diode model
        see W. De Soto, S. a. Klein, and W. a. Beckman, 
            "Improvement and validation of a model for photovoltaic array performance" 
            Sol. Energy, vol. 80, no. 1, pp. 78-88, Jan. 2006.
    Returns:
        dict: 'Iph'   --> Photo current [A]
              'I0'    --> Saturation current [A]
              'Rs'    --> Series resistance [Ohm]
              'Rsh'   --> Shunt resistance [Ohm] 
              'a'     --> modified diode equality factor
              'b'     --> ???
              'm'     --> ???
              'Vbr'   --> Breakdown voltage [V]
    """
    #creates data for input to curve_fit
    y_data = np.zeros(len(Vpoints)) #array of zeros
    I_V_points = np.array([Vpoints, Ipoints]) #2-D array for input to curve_fit
    #get this to work w/actual data
        #Finds closest point to Isc to use as guess for Iph:
    index1=np.argmin(np.abs(Vpoints))#np.where(np.abs(Vpoints)==min(np.abs(Vpoints)))
    I_zero =Ipoints[index1] 
    I_zero=float(I_zero) #changes to float from array
    index2=np.argmin(np.abs(Ipoints))#min(np.abs(Ipoints))==np.abs(Ipoints))
    V_zero=float(Vpoints[index2])#both original Izero and Vzero have problems  
    AC=10 #intialise AC
    AC_list = [AC]
    #initialising ranges of values to be guessed:
    guess_values = [I_zero,1E-10,0.0001,5,0.05,0.1,3,-3*V_zero] 
    i = 0 #loop initialiser
    #May need to change this initial AC value - it's currently 
    #have-to-run-it-twice and take a good guess from the first run
    while (AC > 0.8):
        try:
            #new guess values from curve_fit:
            returns = fitcurve(I_V_points,y_data,guess_values)[0] 
            params = {'Iph': returns[0], 'I0': returns[1], 'Rs': returns[2], 
                      'Rsh': returns[3], 'a': returns[4], 'b': returns[5],
                      'm': returns[6],'Vbr': returns[7],} #formatting
#             if fixed_ideality:
#                 params['a'] = idealityFToA(fixed_ideality)
            I_guess = solvecurve(params,Vpoints)#solves curve for guessed parameters
            #X_2.append(chi_squared(Ipoints,I_guess))
            AC = area_criterion(Vpoints, Ipoints,I_guess)
            AC_list.append(AC)  
            #this lower bit's a bit fudgy too
            if i<2:
                guess_values = [params['Iph'], params['I0'], 
                                params['Rs'], params['Rsh'], 
                                params['a'], params['b'],
                                params['m'], params['Vbr']]   
            #not really used -to remove
            if AC_list[-1] >= AC_list[-2]: 
                #if (AC_list[-1]==AC_list[-2] and AC_list[-2]==AC_list[-3]):
                guess_values[3]=guess_values[3]/1.5    
            else:
                guess_values = [params['Iph'], params['I0'], 
                                params['Rs'], params['Rsh'], 
                                params['a'], params['b'],
                                params['m'], params['Vbr']]  
        except RuntimeError:
            #print('RT\n')
            guess_values[2]=guess_values[2]/1.2
            guess_values[3]=guess_values[3]*1.1
            #return 0
            #break
        except ValueError: #allows for a guess of Vbr being too low
            guess_values[7]*=1.5
            #print(guess_values[7])
        i+=1    
        if i >= 100: #if doesn't go below the while, break
            break
 
    five_params = {'Iph': guess_values[0], 'I0': guess_values[1], 
                   'Rs':guess_values[2], 'Rsh': guess_values[3], 
                   'a': guess_values[4], 'b': guess_values[5],
                   'm': guess_values[6],'Vbr': guess_values[7],}
    
    return five_params
    

def aToidealityF(a):
    '''calculates ideality factor from modified factor [a]'''
    return a*q/(k*298.0)


def idealityFToA(idF):
    '''calculates ideality factor from modified factor [a]'''
    return (idF*k*298.0)/q


def fitIV(I,V,nCells=1):
    '''
    Returns:
        diode parameters, performance p., fitted current, relative RMSE fit error
    '''
    Vcell = V/nCells
    diode_params = fitting_algorithm(I,Vcell)#, fixed_ideality)
    diode_params['ideality'] = aToidealityF(diode_params['a'])
    
    I_guess = solvecurve(diode_params,Vcell)
    
    error = 100 * ((I-I_guess)**2).mean()**0.5 / I.mean()
    
    perf = dict(zip(('Voc', 'Isc', 'FF', 'Vmax', 'Imax', 'Pmax'), 
                    evalPerformance(I_guess,V)))


    return diode_params, perf, I_guess, error


def trimIV(I,V):
    '''
    *reduce I, V to only positive values
    * sort values
    * 1st I value == Isc
    * last V value == Voc
    '''
    #remove negative
    ind = np.logical_and(I>0, V>0)
    V = V[ind]
    I = I[ind]
    #sort order:
    indV = np.argsort(V)
    I = I[indV]
    V = V[indV]
    #remove duplicates
    V, ind = np.unique(V, return_index=True)
    I = I[ind]
    
    #insert Isc
        #TODO: make better
    Isc = np.polyfit(V[:4], I[:4] ,0)[0]
    I = np.insert(I,0,Isc)
    V = np.insert(V,0,0)
    #insert Voc
        #TODO: make better
    Voc = V[-1]
    V = np.append(V,Voc)
    I = np.append(I,0)
    return I,V


def evalPerformance(I,V):
    I,V = trimIV(I,V)
    Voc = V[-1]
    Isc = I[0]

    P = I*V
    i = np.argmax(P)
    Pmax = P[i]
    Vmax = V[i]
    Imax = I[i]
    
    FF = 100*(Vmax*Imax) / (Voc*Isc)
    
    #TODO: calc P and eff
    return Voc, Isc, FF, Vmax, Imax, Pmax

    
if __name__ == "__main__":
    import pprint 
    
    #generate synthetic measurement data:
    params = {'Iph': 8, 'I0': 1E-10, 'Rs': 0.0001, 
                      'Rsh': 2, 'a': 0.05, 'b': 0.1,
                      'm': 3,'Vbr': -3*5,} #formatting
    V = np.linspace(0,5,500)
    I = solvecurve(params, V)
    ind = I>0
    I = I[ind]
    V = V[ind]
    I+=np.random.rand(len(I))*0.1
    V+=np.random.rand(len(V))*0.01
    #fit:
    diod, perf, Ifit, error= fitIV(I,V)
    #print:
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(diod)
    pp.pprint(perf)
    #plot:
    plotIV(I, V, Ifit)
