#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ahkab
from ahkab import circuit
import numpy as np



class PVmodule(object):
    def __init__(self, ncells=60, currents=9.,Voc=50, title='PVmodule',
                 bypassI=(0,20,40), 
                 diodeParams=dict(Rs=0.01,Rsh=200.,ideality=1.2,
                                  I0=1e-10)):#I0-->satCurrent
        '''
        current (float or 1darray): either one value, if same for all cells
                      or 1d array with one value per cell
                      if every cells has an individual current
        bypassI (tuple): index positions of bypass diodes   
                         for a 60 cell module these ones are typically 
                         at (0,20,40)        
        
        '''
  
        self.c = c = circuit.Circuit(title=title)
        self.res = None #simulation result
        gnd = c.get_ground_node()
        c.add_model("diode", "diode", {"name": "..diode..", 
                                                   'N':diodeParams['ideality'],
                                                   'IS':diodeParams['I0']})
        c.add_model("diode", "bypassdiode", 
                            {"name": "..bypassdiode.."}, 
                            #couldnt find useful information specified bypass diodes, so
                            #i d rather take the default settings...
        #                                            'N':idealityF,
        #                                            'IS':satCurr}
                            )
        self.Voc = Voc
 
        #voltage step function: linear increase till time=1, then hold:
        vs = (0, 0, 1, Voc,2,Voc)
        x, y = vs[::2], vs[1::2]
        self.voltage_sweep = ahkab.time_functions.pwl(x, y, repeat=0)
        c.add_vsource("V cell", n1=gnd, n2='nlast', 
                              dc_value=0,function=self.voltage_sweep
                            )
        #add cells:
        nout = gnd
        bypassnodes = []
        try:
            assert currents.shape==ncells
        except:
            currents = np.full(shape=ncells,fill_value=currents)
 
        Rsh = diodeParams['Rsh']
        try:
            assert Rsh.shape==ncells
        except:
            Rsh = np.full(shape=ncells,fill_value=Rsh)

        Rs = diodeParams['Rs']
        try:
            assert Rs.shape==ncells
        except:
            Rs = np.full(shape=ncells,fill_value=Rs)
 
        for i, cur, Rshi, Rsi in zip(range(ncells), currents, Rsh, Rs):
            if i == ncells -1:
                n2 = 'nlast'
            else:
                n2 = 'n%s' %i
            if i in bypassI:
                bypassnodes.append(nout)
            nout = self.addCell('cell%s' %i, nout, n2, cur, 
                                Rshi, 
                                Rsi*ncells)
        #add bypass diodes:   
        bypassnodes.append(nout)
        for n,n2 in zip(bypassnodes[:-1], bypassnodes[1:]):
            c.add_diode("diode", n1=n2, n2=n, 
                            model_label="bypassdiode")
         

    def addCell(self, name, n1in, n2in, current, Rsh, Rs):
        n1out = 'n1' +name
        self.c.add_isource("I cell", n1=n1out, n2=n1in, dc_value=current)
        self.c.add_diode("diode", n1=n1in, n2=n1out, model_label="diode")
        self.c.add_resistor("Rshunt", n1=n1out, n2=n1in, value=Rsh)
        self.c.add_resistor("Rseries", n1=n1out, n2=n2in, value=Rs)
        return n1out


    def IV_steadyState(self, nPoints=100):
        print('simulating ...')

        aa = ahkab.new_dc(0, self.Voc, nPoints, "V cell")

        self.res = ahkab.run(self.c, an_list=[aa])
        I = self.res['dc']['I(V CELL)']
        V = self.res['dc']["V cell"]
        print('done')
        return I,V


    def IV_transient(self, tstart=0, tstop=1, tstep=2e-2):
        print('simulating ...')
        tran_analysis = ahkab.new_tran(tstart=tstart, tstop=tstop, tstep=tstep, x0=None)
        self.res = ahkab.run(self.c, an_list=[#op_analysis, 
                                              tran_analysis])

        time = self.res['tran']['T']
        myg = np.frompyfunc(self.voltage_sweep, 1, 1)
        V = myg(time)
        I = self.res['tran']['I(V CELL)']
        return I,V



if __name__ == '__main__':
    import pylab as plt
    
    currents = np.full(shape=60,fill_value=9.)
    currents[18]=6
#     currents[19]=4
#      currents[52]=6

    import time
    mod = PVmodule(currents=currents)
    
    start = time.time()
#     I,V = mod.IV_transient()
    I,V = mod.IV_steadyState()
    end = time.time()
    
    print('time needed: %.3f s' %(end-start))

    plt.plot(V,I)
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A]')

    plt.show()
