from pdb import set_trace
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


mu0 = 4*pi*1e-7 # Vacuum permiability (H/m)
kb = 1.38e-23 # Boltzmann constant (J/K)
qe = 1.6e-19 # Charge of an electron (C or J/eV)

h = 10.0e-3
dt = 0.2e-9
burn = 'Ndddot'

# scan = np.load('preheat_nernstON_h7p5_Bz20_scan.npz')
scan = np.load('rhog0.7_h0.01_Bz20.0_nernst1_endloss2_laserabs0_Nls20.npz')
scan = scan['arr_0'].item()
n_elements = len(scan.keys())
Ezbl = np.zeros(n_elements)
TgkeV = np.zeros([n_elements,1206])
Phizg = np.zeros([n_elements,1206])
rg = np.zeros([n_elements,1206])
rl = np.zeros([n_elements,1206])
pBzg = np.zeros([n_elements,1206])
pTh = np.zeros([n_elements,1206])
n = np.zeros([n_elements,1206])
nd = np.zeros([n_elements,1206])
t = np.zeros([n_elements,1206])
Yddn = np.zeros([n_elements,1206])
scan_dict = dict()
i = 0
for key in scan:
    key_ = key.replace('run_','')
    Ezbl[i] = int(key_)
    TgkeV[i] = scan[key]['TgkeV']
    Phizg[i] = scan[key]['Phizg']
    rg[i] = scan[key]['rg']
    rl[i] = scan[key]['rl']
    pBzg[i] = ((Phizg[i]/(pi*rg[i]**2))**2)/(2.0*mu0)
    t[i] = scan[key]['t']
    n[i] = (1+scan[key]['Zgbar'])*scan[key]['Nd']/(pi*h*rg[i]**2)
    nd[i] = scan[key]['Nd']
    pTh[i] = n[i]*TgkeV[i]*qe*1e3
    CR = rg[i][0]/rg[i]
    Yddn[i] = scan[key]['Yddn']
    index_peak_burn = np.where(scan[key][burn] == scan[key][burn].max())[0].min()
    cr_pb = scan[key]['rg'][0]/scan[key]['rg'][index_peak_burn]
    index_cr = np.where(CR == CR.max())[0]
    Er = scan[key]['Erdot']*dt/scan[key]['Eg']
    E_ends = scan[key]['E_endsdot']*dt/scan[key]['Eg']
    Ethg = scan[key]['Ethg']
    Vg = (pi*h*rg[i]**2)
    scan_dict[key_] = {'TgkeV':TgkeV[i],'pBzg':pBzg[i],'t':t[i],'n':n[i],'pTh':pTh[i],'beta':pTh[i]/pBzg[i],'rg':rg[i],'CRmax':CR,'nd':nd[i],'Yddn':Yddn[i], 'CRpb':cr_pb, 'pbi':index_peak_burn,'Er':Er,'E_ends':E_ends, 'cri':index_cr, 'Ethg':Ethg, 'Vg':Vg}
    i += 1
h7 = scan_dict

h = h

# scan = np.load('preheat_nernstON_h10_Bz20_scan.npz')
# scan = np.load('rhog0.7_h0.01_Bz20.0_nernst1_endloss0_laserabs0_Nls20.npz')
scan = np.load('rhog0.7_h0.01_Bz20.0_nernst0_endloss2_laserabs0_Nls20.npz')
scan = scan['arr_0'].item()
n_elements = len(scan.keys())
Ezbl = np.zeros(n_elements)
TgkeV = np.zeros([n_elements,1206])
Phizg = np.zeros([n_elements,1206])
rg = np.zeros([n_elements,1206])
rl = np.zeros([n_elements,1206])
pBzg = np.zeros([n_elements,1206])
pTh = np.zeros([n_elements,1206])
n = np.zeros([n_elements,1206])
nd = np.zeros([n_elements,1206])
t = np.zeros([n_elements,1206])
Yddn = np.zeros([n_elements,1206])
scan_dict = dict()
i = 0
for key in scan:
    key_ = key.replace('run_','')
    Ezbl[i] = int(key_)
    TgkeV[i] = scan[key]['TgkeV']
    Phizg[i] = scan[key]['Phizg']
    rg[i] = scan[key]['rg']
    rl[i] = scan[key]['rl']
    pBzg[i] = ((Phizg[i]/(pi*rg[i]**2))**2)/(2.0*mu0)
    t[i] = scan[key]['t']
    n[i] = (1+scan[key]['Zgbar'])*scan[key]['Nd']/(pi*h*rg[i]**2)
    nd[i] = scan[key]['Nd']
    pTh[i] = n[i]*TgkeV[i]*qe*1e3
    CR = rg[i][0]/rg[i]
    Yddn[i] = scan[key]['Yddn']
    index_peak_burn = np.where(scan[key][burn] == scan[key][burn].max())[0].min()
    cr_pb = scan[key]['rg'][0]/scan[key]['rg'][index_peak_burn]
    index_cr = np.where(CR == CR.max())[0]
    Er = scan[key]['Erdot']*dt/scan[key]['Eg']
    E_ends = scan[key]['E_endsdot']*dt/scan[key]['Eg']
    Ethg = scan[key]['Ethg']
    Vg = (pi*h*rg[i]**2)
    scan_dict[key_] = {'TgkeV':TgkeV[i],'pBzg':pBzg[i],'t':t[i],'n':n[i],'pTh':pTh[i],'beta':pTh[i]/pBzg[i],'rg':rg[i],'CRmax':CR,'nd':nd[i],'Yddn':Yddn[i], 'CRpb':cr_pb, 'pbi':index_peak_burn,'Er':Er,'E_ends':E_ends, 'cri':index_cr, 'Ethg':Ethg, 'Vg':Vg}
    i += 1
h10 = scan_dict

t_pb = 0.0
t_crmax = 0.0
for key in h7:
    print(h7[key]['pbi'], h10[key]['pbi'])
    t_pb += 0.5*(h7[key]['t'][h7[key]['pbi']] + h7[key]['t'][h10[key]['pbi']])
    t_crmax += 0.5*(h7[key]['t'][np.where(h7[key]['CRmax'] == h7[key]['CRmax'].max())] + h10[key]['t'][np.where(h10[key]['CRmax'] == h10[key]['CRmax'].max())])
t_pb = t_pb.sum()/5.
t_crmax = t_crmax.sum()/5.


PHstr = ['500','1000','2000','3000','4000']
PHstr = ['200', '200', '300', '400', '750', '1000', '1500', '2000', '3000', '4000']
tscale = 1e9
pscale = 1e-9*1e-5

plt.figure(100)
plt.clf()
plt.yscale('log')
k = 0
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key]['beta'],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.plot(h10[key]['t']*tscale, h10[key]['beta'],'--',color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1
    
# plt.axvline(x=t_crmax*tscale,color='black')
plt.plot(h7['4000']['t']*1e9,np.ones(h7['4000']['t'].size),'k--')
plt.xlim([3040,h7['4000']['t'][-1]*tscale])
plt.legend()
plt.ylabel('$beta$')
plt.xlabel('t [ns]')
plt.show()


plt.figure(101)
plt.clf()
plt.subplot(3,1,1)
k = 0
param = 'pBzg'
param2 = 'pTh'
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param]+h7[key][param2],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.plot(h10[key]['t']*tscale, h10[key][param]+h10[key][param2],'--',color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1
    
# plt.axvline(x=t_crmax*tscale,color='black')
plt.xlim([3100,3120])
plt.legend()
plt.ylabel('p$_{tot}$')

plt.subplot(3,1,2)
k = 0
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.plot(h10[key]['t']*tscale, h10[key][param],'--',color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1
    
# plt.axvline(x=t_crmax*tscale,color='black')
plt.xlim([3100,3120])
plt.legend()
plt.ylabel('p$_{Bzg}$')

plt.subplot(3,1,3)
k = 0
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param2],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.plot(h10[key]['t']*tscale, h10[key][param2],'--',color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1
    
# plt.axvline(x=t_crmax*tscale,color='black')
plt.xlim([3100,3120])
plt.legend()
plt.ylabel('p$_{Th}$')

plt.xlabel('t [ns]')
plt.show()



plt.figure(102)
plt.clf()
plt.subplot(2,1,1)
k = 0
param = 'Er'
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.plot(h10[key]['t']*tscale, h10[key][param],'--',color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1

# plt.axvline(x=t_crmax*tscale,color='black')
plt.plot(h7['4000']['t']*1e9,np.ones(h7['4000']['t'].size),'k--')
plt.xlim([3040,3135])
plt.yscale('log')
plt.ylim([1e-4,1e2])
plt.legend()
plt.ylabel('$E_{rad,loss}/E_{g}$')


plt.subplot(2,1,2)
k = 0
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param]-h10[key][param],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1
    
# plt.axvline(x=t_crmax*tscale,color='black')
plt.plot(h7['4000']['t']*1e9,np.ones(h7['4000']['t'].size),'k--')
plt.xlim([3040,3135])
plt.yscale('log')
plt.ylim([1e-2,1e2])
# plt.legend()
plt.ylabel(r'$\frac{E_{rad,loss}}{E_{g}}$$|_{h=7.5mm}$ - $\frac{E_{rad,loss}}{E_{g}}$$|_{h=10mm}$')
plt.xlabel('t [ns]')
plt.show()


plt.figure(103)
plt.clf()
plt.subplot(2,1,1)
k = 0
param = 'E_ends'
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.plot(h10[key]['t']*tscale, h10[key][param],'--',color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1

# plt.axvline(x=t_crmax*tscale,color='black')
plt.plot(h7['4000']['t']*1e9,np.ones(h7['4000']['t'].size),'k--')
plt.xlim([3040,3135])
plt.yscale('log')
plt.ylim([1e-4,1e2])
plt.legend()
plt.ylabel('$E_{ends,loss}/E_{g}$')

plt.subplot(2,1,2)
k = 0
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param]-h10[key][param],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1
    
# plt.axvline(x=t_crmax*tscale,color='black')
plt.plot(h7['4000']['t']*1e9,np.ones(h7['4000']['t'].size),'k--')
plt.xlim([3040,3135])
plt.yscale('log')
plt.ylim([1e-2,1e2])
# plt.legend()
plt.ylabel(r'$\frac{E_{ends,loss}}{E_{g}}$$|_{h=7.5mm}$ - $\frac{E_{ends,loss}}{E_{g}}$$|_{h=10mm}$')
plt.xlabel('t [ns]')
plt.show()


plt.figure(104)
plt.clf()
k = 0
param = 'CRmax'
for key in PHstr:
    
    pc = plt.plot(h7[key]['t']*tscale, h7[key][param],'-',label='E$_{zbl}$='+PHstr[k]+' J')
    plt.plot(h10[key]['t']*tscale, h10[key][param],'--',color=pc[0].get_color(),label=None)
    plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    # plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    # plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1

# plt.axvline(x=t_crmax*tscale,color='black')
plt.plot(h7['4000']['t']*1e9,np.ones(h7['4000']['t'].size),'k--')
plt.xlim([3100,3120])
# plt.yscale('log')
# plt.ylim([1e-4,1e2])
plt.legend()
plt.ylabel('Convergence')
plt.xlabel('t [ns]')
plt.show()

plt.figure(108)
plt.clf()
k = 0
param = 'pTh'
Ezbl = np.array([200.0, 300.0, 400.0, 500.0, 750.0, 1.0e3, 1.5e3, 2.0e3, 3.0e3, 4.0e3])
for key in PHstr:
    
    pbi = h7[key]['pbi']
    pc = plt.plot(Ezbl[k], h7[key][param][pbi],'.',label='E$_{zbl}$='+PHstr[k]+' J')
    pbi = h10[key]['pbi']
    plt.plot(Ezbl[k], h10[key][param][pbi],'*',color=pc[0].get_color(),label=None)
    del pbi
    # plt.axvline(x=h7[key]['t'][h7[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    # # plt.axvline(x=h7[key]['t'][h7[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    # plt.axvline(x=h10[key]['t'][h10[key]['pbi']]*tscale,color=pc[0].get_color(),label=None)
    # # plt.axvline(x=h10[key]['t'][h10[key]['cri']]*tscale,color=pc[0].get_color(),label=None)
    k += 1

# plt.axvline(x=t_crmax*tscale,color='black')
# plt.plot(h7['4000']['t']*1e9,np.ones(h7['4000']['t'].size),'k--')
# plt.xlim([3100,3120])
plt.yscale('log')
# # plt.ylim([1e-4,1e2])
# plt.legend(['Nernst/Endloss','Nernst'])
plt.legend(['Nernst/Endloss','Endloss'])
# plt.ylabel('Convergence')
# plt.xlabel('t [ns]')
plt.show()
