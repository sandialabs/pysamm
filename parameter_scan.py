from pdb import set_trace
from pySAMM import *
import numpy as np

def save_data(dictionary, path, filename):
    np.savez(path+filename+'_quickouts', dictionary)
    return
        
def load_data(path, filename):
    data = np.load(path+filename+'.npz')
    keys = data.files[0]
    data = data[keys]
    return data.item()

def parameter_scan():
    """
    Make this more general with keyword arguments, dicts, etc
    """
    # res = 20
    # Ezbl_scan = np.linspace(1.5e3,2.5e3,3)
    # Ezbl_scan = np.array([2.0e3])
    rhog_scan = np.linspace(0.7,1.4,2)
    Bz_scan = np.linspace(15,25,3)
    rl_scan = np.linspace(0.00279,0.00339,7)
    Ezbl_= 2.0e3
    h_=10.0e-3
    n_endloss_ = 2
    laserabs_ = 1
    AR_ = 9
    # resi=len(Ezbl_scan);
    resi=len(rhog_scan);
    resj=len(Bz_scan);
    resk=len(rl_scan);
    ddn = np.zeros([resi,resj,resk])
    crmax = np.zeros([resi,resj,resk])
    cr_peak_burn = np.zeros([resi,resj,resk])
    path = '/Users/thomoor/data/pysamm/rl_scan/'
    status = open(path+"param_scans_completed.txt", "w")
    status.close()
    i_ = 0
    j_ = 0
    k_ = 0
    for i in range(i_,resi):
        for j in range(j_,resj):
            for k in range(k_,resk):
            # for j in range(j_,res):
                # filename = 'run_Bz'+str(int(Bz_scan[j]))+'_Ezbl'+str(int(Ezbl_scan[i]))+'_rl'+str(int(rl_scan[k]*1e4)*1.e-4)
                filename = 'run_Bz'+str(int(Bz_scan[j]))+'_rhog'+str(int(rhog_scan[i]))+'_rl'+str(int(rl_scan[k]*1e4)*1.e-4)
                print('Running '+filename),
                print('(i=%s, j=%s, k=%s)' % (i,j,k))
                try:
                    run_ = run_pySAMM(rl0=rl_scan[k], Bz0=Bz_scan[j], Ezbl=Ezbl_, rhog0=rhog_scan[i], AR=AR_, h=h_, n_endloss=n_endloss_, laserabs=laserabs_, VI_mode='V', param_file=path+filename+'.npz', save_inputs=1, save_outputs=1, verbose=0)
                    # run_ = run_pySAMM(rl0=rl_scan[k], Bz0=Bz_scan[j], Ezbl=Ezbl_scan[i], AR=AR_, h=h_, n_endloss=n_endloss_, laserabs=laserabs_, VI_mode='V', param_file=path+filename+'.npz', save_inputs=1, save_outputs=1, verbose=0)
                    outs = run_.outputs
                    ddn[i,j,k] = run_.outputs['Yddn'].max()
                    index_peak_burn = np.where(run_.outputs['Ndddot'] == run_.outputs['Ndddot'].max())[0].min()
                    cr_peak_burn[i,j,k] = run_.outputs['rg'][0]/run_.outputs['rg'][index_peak_burn]
                    crmax[i,j,k] = run_.outputs['rg'][0]/run_.outputs['rg'].min()
                except ValueError as err:
                    outs = {'ValueError':err[0]}
                    ddn[i,j,k] = -999
                    crmax[i,j,k] = -999
                except UnboundLocalError as err:
                    outs = {'UnboundLocalError':err[0]}
                    ddn[i,j,k] = -999
                    crmax[i,j,k] = -999
                save_data(outs, path, filename)
                results = {'Yddn':ddn,'CR_max':crmax, 'CR_pb':cr_peak_burn}
                np.savez(path+'inner_liner_radius_scan', results)
                status = open(path+"param_scans_completed.txt", "a")
                status.write('i=%s, j=%s, k=%s\n' % (i,j,k))
                status.close()
                print('Approximately %3.3s ' % ((i*1.0)/(resi)*100.0) +'% complete')
            
    return results

def load_parameter_scan(path):
    # res = 20
    # Ezbl_scan = np.linspace(1.5e3,2.5e3,3)
    # Bz_scan = np.linspace(10,25,4)
    # rl_scan = np.linspace(0.00219,0.00289,8)
    # Ezbl_scan = np.array([2.0e3])
    # Bz_scan = np.linspace(15,25,3)
    # rl_scan = np.linspace(0.00279,0.00379,10)
    rhog_scan = np.linspace(0.7,1.4,2)
    Bz_scan = np.linspace(15,25,3)
    rl_scan = np.linspace(0.00279,0.00339,7)
    # resi=len(Ezbl_scan);
    resi=len(rhog_scan);
    resj=len(Bz_scan);
    resk=len(rl_scan);
    ddn = np.zeros([resi,resj,resk])
    crmax = np.zeros([resi,resj,resk])
    cr_peak_burn = np.zeros([resi,resj,resk])
    # path = '/Users/thomoor/data/pysamm/rl_scan/'
    for i in range(0,resi):
        for j in range(0,resj):
            for k in range(0,resk):
            # for j in range(j_,res):
                # filename = 'run_Bz'+str(int(Bz_scan[j]))+'_Ezbl'+str(int(Ezbl_scan[i]))+'_rl'+str(int(rl_scan[k]*1e4)*1.e-4)+'_quickouts'
                filename = 'run_Bz'+str(int(Bz_scan[j]))+'_rhog'+str(int(rhog_scan[i]))+'_rl'+str(int(rl_scan[k]*1e4)*1.e-4)+'_quickouts'
                print('Running '+filename),
                print('(i=%s, j=%s)' % (i,j))
                data = load_data(path, filename)
                try:
                    data['Yddn']
                    ddn[i,j,k] = data['Yddn'].max()
                    # index_peak_burn = np.where(data['Ndddot'] == data['Ndddot'].max())
                    index_peak_burn = np.where(data['TgkeV'] == data['TgkeV'].max())[0][0]
                    # peak_burn = data['Ndddot'] + data['Ndd3Hedot']
                    # index_peak_burn = np.where(peak_burn == peak_burn.max())
                    cr_peak_burn[i,j,k] = data['rg'][0]/data['rg'][index_peak_burn]
                    crmax[i,j,k] = data['rg'][0]/data['rg'].min()
                except KeyError:
                    ddn[i,j,k] = -999
                    crmax[i,j,k] = -999
    # set_trace()
    # # plot_parameter_scan(rhog0_scan,Ezbl_scan,ddn,crmax)
    # plt.figure()
    # plt.subplot(2,1,1)
    # # plt.subplot2grid((5,1),(2,0),rowspan=2)
    # plt.plot(rl_scan,ddn[:,3,:],'-',color='magenta')
    # plt.plot(rl_scan,ddn[:,3],'-',color='orange')
    # plt.plot(rl_scan,ddn[:,2],'-',color='green')
    # plt.plot(rl_scan,ddn[:,1],'-',color='blue')
    # plt.plot(rl_scan,ddn[:,0],'-',color='grey')
    # plt.legend(['B$_{z0}$ = 30 T','B$_{z0}$ = 25 T','B$_{z0}$ = 20 T','B$_{z0}$ = 15 T','B$_{z0}$ = 10 T'])
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.xlim([0,40])
    # # plt.ylim([1e9,2.5e14])
    # plt.ylabel('DD Neutron Yield')
    # plt.gca().axes.get_xaxis().set_ticklabels([])
    # plt.subplot(2,1,2)
    # # plt.subplot2grid((5,1),(2,0),rowspan=2)
    # # plt.plot(Ezbl_scan,crmax[:,4],'-',color='magenta')
    # # plt.plot(Ezbl_scan,crmax[:,3],'-',color='orange')
    # # plt.plot(Ezbl_scan,crmax[:,2],'-',color='green')
    # # plt.plot(Ezbl_scan,crmax[:,1],'-',color='blue')
    # # plt.plot(Ezbl_scan,crmax[:,0],'-',color='grey')
    # plt.plot(Ezbl_scan,cr_peak_burn[:,4],'-',color='magenta')
    # plt.plot(Ezbl_scan,cr_peak_burn[:,3],'-',color='orange')
    # plt.plot(Ezbl_scan,cr_peak_burn[:,2],'-',color='green')
    # plt.plot(Ezbl_scan,cr_peak_burn[:,1],'-',color='blue')
    # plt.plot(Ezbl_scan,cr_peak_burn[:,0],'-',color='grey')
    # plt.legend(['B$_{z0}$ = 30 T','B$_{z0}$ = 10 T','B$_{z0}$ = 3 T','B$_{z0}$ = 1 T','B$_{z0}$ = 0 T'])
    # plt.xscale('log')
    # plt.xlim([30,1e4])
    # plt.ylim([0,200])
    # plt.ylabel('Convergence Ratio @ Peak Burn')
    # plt.tight_layout()
    # plt.show()
    
    # plt.close('all')
    # plt.figure(10), plt.contourf(Bz_scan,Ezbl_scan,ddn[:,:,1],cmap='jet')
    # plt.xlabel('$B_{z}$ $[T]$') #'$r_{L,inner}$ $[mm]$'
    # plt.ylabel('$E_{ZBL}$ $[J]$')
    # cbar = plt.colorbar()
    # cbar.set_label('$Yddn_{max}$')
    
    contour_levels = np.linspace(20,35,16)
    # for i in range((len(Ezbl_scan))):
    for i in range((len(rhog_scan))):
        plt.figure(i), plt.contourf(rl_scan,Bz_scan,ddn[i,:,:],cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label('$Yddn_{max}$')
        cs=plt.contour(rl_scan,Bz_scan,cr_peak_burn[i,:,:],levels=contour_levels,colors='k')
        plt.clabel(cs,levels=cs.levels,fmt='$CR_{%3.0f}$')
        plt.ylabel('$B_{z}$ $[T]$')
        plt.xlabel('$r_{L,inner}$ $[mm]$')
        # plt.title('$E_{ZBL}$ = ' + str(Ezbl_scan[i]) + ' J')
        plt.title('$E_{ZBL}$ = ' + str(rhog_scan[i]) + ' kg/cm^3')
    # return ddn, crmax, cr_peak_burn, Ezbl_scan, Bz_scan, rl_scan
    return ddn, crmax, cr_peak_burn, rhog_scan, Bz_scan, rl_scan