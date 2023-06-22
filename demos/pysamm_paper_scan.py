import pySAMM

def save_data(dictionary, path, filename):
    np.savez(path+filename, dictionary)
    return
        
def load_data(path, filename):
    data = np.load(path+filename+'.npz')
    keys = data.files[0]
    data = data[keys]
    return data.item()

def parameter_scan_from_paper():
    """
    Make this more general with keyword arguments, dicts, etc
    """
    # res = 20
    Ezbl_scan = np.array([20., 40., 80., 150., 300., 600., 1400., 2500., 5000., 10000.])
    Bz_scan = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    n_endloss_ = 2
    laserabs_ = 0
    resi=len(Ezbl_scan);
    resj=len(Bz_scan);
    ddn = np.zeros([resi,resj])
    crmax = np.zeros([resi,resj])
    cr_peak_burn = np.zeros([resi,resj])
    path = '/Users/thomoor/data/samm_paper/scan/'
    status = open(path+"param_scans_completed.txt", "w")
    status.close()
    i_ = 0
    j_ = 0
    for i in range(i_,resi):
        for j in range(j_,resj):
        # for j in range(j_,res):
            filename = 'run_Bz'+str(int(Bz_scan[j]))+'_Ezbl'+str(int(Ezbl_scan[i]))
            print('Running '+filename),
            print('(i=%s, j=%s)' % (i,j))
            try:
                run_ = run_pySAMM(Bz0=Bz_scan[j], Ezbl=Ezbl_scan[i], n_endloss=n_endloss_, laserabs=laserabs_, VI_mode='I')
                outs = run_.outputs
                ddn[i,j] = run_.outputs['Yddn'].max()
                index_peak_burn = np.where(run_.outputs['Ndddot'] == run_.outputs['Ndddot'].max())[0].min()
                cr_peak_burn[i,j] = run_.outputs['rg'][0]/run_.outputs['rg'][index_peak_burn]
                crmax[i,j] = run_.outputs['rg'][0]/run_.outputs['rg'].min()
            except ValueError as err:
                outs = {'ValueError':err[0]}
                ddn[i,j] = -999
                crmax[i,j] = -999
            except UnboundLocalError as err:
                outs = {'UnboundLocalError':err[0]}
                ddn[i,j] = -999
                crmax[i,j] = -999
            save_data(outs, path, filename)
            results = {'Yddn':ddn,'CR_max':crmax, 'CR_pb':cr_peak_burn}
            np.savez(path+'preheat_density_scan', results)
            status = open(path+"param_scans_completed.txt", "a")
            status.write('i=%s, j=%s\n' % (i,j))
            status.close()
            print('Approximately %3.3s ' % ((i*1.0)/(resi)*100.0) +'% complete')
            
    return results

def load_parameter_scan_from_paper():
    # res = 20
    Ezbl_scan = np.array([20., 40., 80., 150., 300., 600., 1400., 2500., 5000., 10000.])
    Bz_scan = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    resi=len(Ezbl_scan);
    resj=len(Bz_scan);
    ddn = np.zeros([resi,resj])
    crmax = np.zeros([resi,resj])
    cr_peak_burn = np.zeros([resi,resj])
    path = '/Users/thomoor/data/samm_paper/scan/'
    for i in range(0,resi):
        for j in range(0,resj):
            filename = 'run_Bz'+str(int(Bz_scan[j]))+'_Ezbl'+str(int(Ezbl_scan[i]))
            print('Running '+filename),
            print('(i=%s, j=%s)' % (i,j))
            data = load_data(path, filename)
            try:
                data['Yddn']
                ddn[i,j] = data['Yddn'].max()
                # index_peak_burn = np.where(data['Ndddot'] == data['Ndddot'].max())
                index_peak_burn = np.where(data['TgkeV'] == data['TgkeV'].max())[0][0]
                # peak_burn = data['Ndddot'] + data['Ndd3Hedot']
                # index_peak_burn = np.where(peak_burn == peak_burn.max())
                cr_peak_burn[i,j] = data['rg'][0]/data['rg'][index_peak_burn]
                crmax[i,j] = data['rg'][0]/data['rg'].min()
            except KeyError:
                ddn[i,j] = -999
                crmax[i,j] = -999
    # plot_parameter_scan(rhog0_scan,Ezbl_scan,ddn,crmax)
    plt.figure()
    plt.subplot(2,1,1)
    # plt.subplot2grid((5,1),(2,0),rowspan=2)
    plt.plot(Ezbl_scan,ddn[:,4],'-',color='magenta')
    plt.plot(Ezbl_scan,ddn[:,3],'-',color='orange')
    plt.plot(Ezbl_scan,ddn[:,2],'-',color='green')
    plt.plot(Ezbl_scan,ddn[:,1],'-',color='blue')
    plt.plot(Ezbl_scan,ddn[:,0],'-',color='grey')
    plt.legend(['B$_{z0}$ = 30 T','B$_{z0}$ = 10 T','B$_{z0}$ = 3 T','B$_{z0}$ = 1 T','B$_{z0}$ = 0 T'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([30,1e4])
    plt.ylim([1e9,2.5e14])
    plt.ylabel('DD Neutron Yield')
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.subplot(2,1,2)
    # plt.subplot2grid((5,1),(2,0),rowspan=2)
    # plt.plot(Ezbl_scan,crmax[:,4],'-',color='magenta')
    # plt.plot(Ezbl_scan,crmax[:,3],'-',color='orange')
    # plt.plot(Ezbl_scan,crmax[:,2],'-',color='green')
    # plt.plot(Ezbl_scan,crmax[:,1],'-',color='blue')
    # plt.plot(Ezbl_scan,crmax[:,0],'-',color='grey')
    plt.plot(Ezbl_scan,cr_peak_burn[:,4],'-',color='magenta')
    plt.plot(Ezbl_scan,cr_peak_burn[:,3],'-',color='orange')
    plt.plot(Ezbl_scan,cr_peak_burn[:,2],'-',color='green')
    plt.plot(Ezbl_scan,cr_peak_burn[:,1],'-',color='blue')
    plt.plot(Ezbl_scan,cr_peak_burn[:,0],'-',color='grey')
    plt.legend(['B$_{z0}$ = 30 T','B$_{z0}$ = 10 T','B$_{z0}$ = 3 T','B$_{z0}$ = 1 T','B$_{z0}$ = 0 T'])
    plt.xscale('log')
    plt.xlim([30,1e4])
    plt.ylim([0,200])
    plt.ylabel('Convergence Ratio @ Peak Burn')
    plt.tight_layout()
    plt.show()
    return ddn, crmax, cr_peak_burn, Ezbl_scan, Bz_scan