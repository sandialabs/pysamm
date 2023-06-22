from pdb import set_trace

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, cos, sin, log, pi

import pandas as pd

from traits.api import HasTraits, CFloat, Property, ReadOnly, Int, Instance, Array, Dict, Str, List, File
from traitsui.api import View, Group, Item, ArrayEditor, TableEditor
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from traitsui.menu import OKButton, CancelButton
from traitsui.table_column import ObjectColumn, ExpressionColumn

from scipy.integrate import odeint, ode
from scipy import integrate
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import signal

import os
import sys
import warnings
import time

from dimensionless_curve import dimensionless_curve

########################################################################################################################################################################################
####################### ENVIRONMENTAL VARIABLES ########################################################################################################################################
########################################################################################################################################################################################

DRIVER_PATH = os.getenv('DRIVER_PATH','/Users/thomoor/code/pysamm/pysamm/')
PARAM_PATH = os.getenv('PARAM_PATH','/Users/thomoor/data/run_parameters/pysamm/')

########################################################################################################################################################################################
####################### Initial Condidtions, Constants and MagLIF System Parameters ####################################################################################################
########################################################################################################################################################################################

class PhysicalConstants(HasTraits):# Physical constants
    """
    Physical constants - this class is not currently used
    """
    # pi = np.pi
    qe = 1.6e-19 # Charge of an electron (C or J/eV)
    # c = 2.99792458e8 # Speed of light in a vacuum (m/s)
    c = 3.0e8 # Speed of light in a vacuum (m/s)
    # hbar = 1.054571800e-34 # Reduced Planck constant (Js/rad)
    hbar = 1.05457e-34 # Reduced Planck constant (Js/rad)
    mu0 = 4*pi*1e-7 # Vacuum permiability (H/m)
    # eps0 = 1/(mu0*c**2)# Vacuum permittivity (F/m)
    eps0 = 8.854e-12
    # kb = 1.38064852e-23 # Boltzmann constant (J/K)
    kb = 1.38e-23 # Boltzmann constant (J/K)
    sigma = 5.67e-8 # Stefan-Boltzmann constant (W/(m^2K^4))
    md = 3.34e-27 # Dueteron mass (kg)
    mt = 5.01e-27 # Triton mass (kg)
    # me = 9.10938356e-31 # Electron mass (kg)
    me = 0.910938e-30 # Electron mass (kg)
    u = 1.66e-27 # Unified atomic mass unit (kg)
    # u = 1.660539040e-27 # Unified atomic mass unit (kg)
    zeta = 2.407
    
"""
Physical constants are currently global parameters.
It would probably be better to throw them in a module.
"""   
# pi = np.pi
qe = 1.6e-19 # Charge of an electron (C or J/eV)
# c = 2.99792458e8 # Speed of light in a vacuum (m/s)
c = 3.0e8 # Speed of light in a vacuum (m/s)
# hbar = 1.054571800e-34 # Reduced Planck constant (Js/rad)
hbar = 1.05457e-34 # Reduced Planck constant (Js/rad)
mu0 = 4*pi*1e-7 # Vacuum permiability (H/m)
# eps0 = 1/(mu0*c**2)# Vacuum permittivity (F/m)
eps0 = 8.854e-12
# kb = 1.38064852e-23 # Boltzmann constant (J/K)
kb = 1.38e-23 # Boltzmann constant (J/K)
sigma = 5.67e-8 # Stefan-Boltzmann constant (W/(m^2K^4))
md = 3.34e-27 # Dueteron mass (kg)
mt = 5.01e-27 # Triton mass (kg)
# me = 9.10938356e-31 # Electron mass (kg)
me = 0.910938e-30 # Electron mass (kg)
u = 1.66e-27 # Unified atomic mass unit (kg)
# u = 1.660539040e-27 # Unified atomic mass unit (kg)
zeta = 2.407

# Used for precision when calculating rh, TB, rhoc and Tc in Pr() in OdeSystem Class
# This is taken from McBride's code
def nonlinspace(x0,xn,n,relstep):
    n = int(n)
    x = np.ones(n)*x0
    for i in range(1,n):
        x[i] = x[i-1] + (xn - x[i-1])*relstep
    #endfor
    dx = np.diff(x)
    dx = dx*(xn-x0)/np.sum(dx)
    for i in range(1,n):
        x[i] = x[i-1] + dx[i-1]
    #endfor
    return x


########################################################################################################################################################################################
####################### MagLIF System Classes ##########################################################################################################################################
########################################################################################################################################################################################

class InputDeck(HasTraits):
    """
    Default input parameters are set to Z2591
        - To change defaults, use keyword assignments (**kwargs) for the input parameters
        
    Default liner material is Be
        - To change defaults, use keywords (*args) for the liner material (e.g. 'Li' or 'Al')
    """
    
    input_param_names = List(['VI_mode','tmin','tmax','deltat','liner_material','nernst','laserabs','n_endloss','AR','rl0','h','Nls', \
                                'Bz0','Troom','rLEC','rdump','fdops','Adops','Zdops','Nr','rhog0','ftrit','Zdt','deltazlec','tzbl', \
                                'tauzbl','Ezbl','lambda_b','xi_b','fsharp','bestfocus','dolew','Taur','Z0','L','L0','C','Rloss', \
                                'voltage_factor','current_factor'], \
                                label='input parameter names', help='input parameters to be saved in paramater file')
    
    # List of parameters that may be set by the user
    permitted_parameters = List(['rhog0','ftrit','fdops','Adops','Zdops','Nr', \
                                 'rl0','rhol0','AR','h','Nls', \
                                 'Bz0','Troom','tmin','tmax', \
                                 'Ezbl','rLEC','rdump','tzbl','tauzbl','lambda_b','xi_b','deltazlec','fsharp','bestfocus','dolew', \
                                 'Taur', \
                                 'voltage_factor','current_factor', \
                                 'Z0','L','L0','C','Rloss'])
    
    hot_spot_parameter = Dict({'rphf':[0,1]})
    
    switches = Dict({'nernst':[0,1],'laserabs':[0,1],'n_endloss':[0,1,2]})
    
    drive_mode = Dict({'VI_mode':['V','I']})
    
    verbose_mode = Dict({'verbose':[0,1]})
    
    # fname should be run with save_inputs = 0 and save_outputs = 1
    save_filename = Dict({'param_file':'string'}) # checks to make sure param_file is a string
    param_file = Str(default_value=PARAM_PATH+'default.npz', label='parameter file', help='file to store and read the parameters in/from') # should not be used if calling 'fname'! will overwrite
    save_mode = Dict({'save_inputs':[0,1],'save_outputs':[0,1]}) # save_inputs = 1 should really only be called in conjunction with 'param_file'
    
    VI_mode = Str(default_value='V')
    voltage_factor = CFloat(default_value=1.0, label='Voltage Factor', help='Multiplicative factor applied to voltage pulse')
    current_factor = CFloat(default_value=1.0, label='Current Factor', help='Multiplicative factor applied to current pulse')
    tmin = CFloat(default_value=2900.e-9)
    tmax = CFloat(default_value=3141.e-9)
    deltat = CFloat(default_value=0.2*1.e-9, help='time step (s)')
    
    def load(self):
        '''
        Loads the meta information for the object
        '''
        data = np.load(os.path.expanduser(self.param_file))
        for attribute, value in data.items():
            if value.dtype == 'S1' or value.dtype == 'S2':
                value = str(value)
                setattr(self, attribute, value)
            else:
                try: getattr(self, attribute)[:] = value
                except TypeError: setattr(self, attribute, np.array(value))
        return
        
    def save(self):
        '''
        Saves the meta information for the object
        '''
        save_list = {}
        for name in self.input_param_names:
            save_list[name] = getattr(self, name)
        np.savez(os.path.expanduser(self.param_file), **save_list)
        return
    
    def __init__(self, fname=None, *args, **kwargs):
        # set_trace()
        if not fname is None:
            if 'save_inputs' in kwargs:
                if kwargs['save_inputs'] == 1:
                    kwargs['save_inputs'] = 0
                    print("WARNING: save_input has been set to 0 to prevent '" + fname + "' from being overwritten!")
            self.param_file = fname
            self.load()
        if len(kwargs)==0:
            pass
        else:
            for key, value in kwargs.iteritems():
                if key in self.permitted_parameters: setattr(self,key,value)
                else:
                    if str(key) in self.switches:
                        if value in self.switches[str(key)]: setattr(self,key,value)
                        else: print('%s is out of range for %s' % (value,key))
                    else:
                        if str(key) in self.drive_mode:
                            if value in self.drive_mode[str(key)]: setattr(self,key,value)
                            else: print('%s not allowable for %s.' % (value,key))
                        else:
                            if str(key) in self.hot_spot_parameter:
                                if value >= self.hot_spot_parameter[str(key)][0] and value <= self.hot_spot_parameter[str(key)][1]: setattr(self,key,value)
                                else: print('%s is out of range for %s.' % (value,key))
                            else:
                                if str(key) in self.save_mode:
                                    if value in self.save_mode[str(key)]: setattr(self,key,value)
                                    else: print('%s is out of range for %s' % (value,key))
                                else:
                                    if str(key) in self.save_filename:
                                        if isinstance(value, basestring) == True: setattr(self,key,str(value))
                                        else: print('%s must be a string.' % (value))
                                    else:
                                        if str(key) in self.verbose_mode:
                                            if value in self.verbose_mode[str(key)]: setattr(self,key,value)
                                            else: print('%s is out of range for %s' % (value,key))
                                        else: print('%s not loaded' % (key))
                    
        if len(args) > 1:
            raise SystemExit('Length of args -- string representation of liner material -- cannot be greater than zero.')
        for string in args: liner_material = string
        if len(args)==0 or liner_material=='Be':
            # print('Be')
            ## Be Liner ##
            self.liner_material = 'Be'
            self.rhol0 = 1850.0 # Initial mass density of liner material (kg/m^3)
            self.rhol0_zero = 1845.0 # Initial mass density of liner material (kg/m^3) (For zero temperature)
            self.rhoe = 36.0*1.0e-9 # Initial electrical resistivity of liner (Ohms*m) -- (Be=36, Li=92.8, Al=28.2)
            self.Zlnuc = 4.0 # Nuclear charge of liner material
            self.Alu = 9.01218 # Atomic mass number of liner material
            self.A1 = 130.0e9 # Cold curve fitting parameter
            self.A2 = 3.9993 # Cold curve fitting parameter
            self.gamma1 = 1.85 # Cold curve fitting parameter
            self.gamma2 = 1.18 # Cold curve fitting parameter
        elif liner_material=='Li':
            print('Li')
            ## Li Liner ##
            self.liner_material = 'Li'
            self.rhol0 = 534.0 # Initial mass density of liner material (kg/m^3)
            self.rhol0_zero = 534.0 # Initial mass density of liner material (kg/m^3) (For zero temperature)
            self.rhoe = 92.8*1.0e-9 # Initial electrical resistivity of liner (Ohms*m)
            self.Zlnuc = 3.0 # Nuclear charge of liner material
            self.Alu = 6.94 # Atomic mass number of liner material
            self.A1 = 11.0e9 # Cold curve fitting parameter
            self.A2 = 3.999 # Cold curve fitting parameter
            self.gamma1 = 1.9 # Cold curve fitting parameter
            self.gamma2 = 1.18 # Cold curve fitting parameter
        elif liner_material=='Al':
            print('Al')
            ## Al Liner ##
            self.liner_material = 'Al'
            self.rhol0 = 2700.0 # Initial mass density of liner material (kg/m^3)
            self.rhol0_zero = 2700.0 # Initial mass density of liner material (kg/m^3) (For zero temperature)
            self.rhoe = 28.2*1.0e-9 # Initial electrical resistivity of liner (Ohms*m)
            self.Zlnuc = 13.0 # Nuclear charge of liner material
            self.Alu = 26.98 # Atomic mass number of liner material
            self.A1 = 76.0e9 # Cold curve fitting parameter
            self.A2 = 3.9 # Cold curve fitting parameter
            self.gamma1 = 7.0/3.0 # Cold curve fitting parameter
            self.gamma2 = 5.0/3.0 # Cold curve fitting parameter
            
        if self.VI_mode == 'V':
            drive = np.loadtxt(DRIVER_PATH + 'open_voltage.txt')
            drive_mult_factor=95./80.*(0.99*19.4/27.)
            drive[:,1] = drive[:,1]*drive_mult_factor
            drive[:,1] = drive[:,1]*drive_mult_factor*self.voltage_factor
        elif self.VI_mode == 'I':
            drive = np.loadtxt(DRIVER_PATH + 'current.txt')
            drive_mult_factor=1.23*18.15/16.44
            drive[:,1] = drive[:,1]*drive_mult_factor*self.current_factor
        number_points = round((self.tmax - self.tmin)/self.deltat + 1.)
        self.t = np.linspace(self.tmin, self.tmax, int(number_points))
        self.drive_amp = interp1d(drive[:,0], drive[:,1], kind='linear', bounds_error=False, fill_value=0)(self.t)
        self.drivedot = []
        if self.VI_mode == 'I':
            amp_diff = np.diff(self.drive_amp)
            amp_diff = np.append(amp_diff,amp_diff[-1])
            time_diff = np.diff(self.t)
            time_diff = np.append(time_diff,time_diff[-1])
            self.drivedot = amp_diff/time_diff
            
    # Switches
    nernst = CFloat(default_value=1, help='1 toggles Nernst on, 0 toggles Nernst off')
    laserabs = CFloat(default_value=1, help='1 toggles laser absorption on, 0 toggles laser absorption off')
    n_endloss = CFloat(default_value=2, help='2 - top and bottom end losses on, 1 - top end losses only, 0 - no endlosses')
    save_inputs = Int(default_value=0)
    save_outputs = Int(default_value=0)
    verbose = Int(default_value=0)
    
    # Liner inputs
    AR = CFloat(default_value=6.0, label='AR', help='Aspect ratio AR = rl0/(rl0-rg0)')
    rl0 = CFloat(default_value=0.00279, help='Inintial outer liner radius)')
    rg0 = Property(depends_on=['rl0','AR'], label='rg0', help='Initial fuel-liner interface radius')
    def _get_rg0(self):
        return self.rl0*(self.AR - 1.0)/self.AR
    h = CFloat(default_value=7.5e-3, label='h', help='Height of liner')
    Nls = Int(default_value=20, label='Nls', help='Number of concentric shells (Nls>=20)')
    Bz0 = CFloat(default_value=10.0, label='Bz0', help='Initial axial magnetic field (T) Default')
    Troom = CFloat(default_value=300.0, label='Troom', help='Room temperature of Z facility (K)')
    rLEC = CFloat(default_value=0.0015, label='rLEC', help='Radius of laser entrance channel -- affects top end losses') 
    rdump = CFloat(default_value=1.0000e-03, label='rdump', help='Radius of beam dump -- affects bottom end losses')
    
    Tl0 = Property(depends_on=['Troom'], label='Tl0', help='Initial temperature of liner (K)')
    def _get_Tl0(self):
        return self.Troom
    TlkeV0 = Property(depends_on=['Tl0'])
    def _get_TlkeV0(self):
        return kb*self.Tl0/qe*1e-3
    Zlbar0 = Property(depends_on=['TlkeV0','Zlnuc'])
    def _get_Zlbar0(self):
        return min(20.*sqrt(self.TlkeV0),self.Zlnuc)
    
    fdops = CFloat(default_value=np.array([0.0]), label='fdops', help='Fraction of dopants in fuel')
    Adops = CFloat(default_value=np.array([0.0]), label='Adops', help='Atomic mass number of dopants in fuel')
    Zdops = CFloat(default_value=np.array([0.0]), label='Zdops', help='Atomic number of dopants in fuel')
    mdops = Property(depends_on=['Adops'], label='mdops', help='Unit mass of dopants in fuel')
    def _get_mdops(self):
        return self.Adops*u
    
    # Fuel inputs
    Nr = CFloat(default_value=100, label='Nr', help='# of radial points in fuel (usually 100)')
    # rg0 = rg0
    Tg0 = Property(depends_on=['Troom'], label='Tl0', help='Initial temperature of fuel (K)')
    def _get_Tg0(self):
        return self.Troom
    TgkeV0 = Property(depends_on=['Tg0'])
    def _get_TgkeV0(self):
        return kb*self.Tg0/qe*1e-3
    rhog0 = CFloat(default_value=0.7, label='rhog0', help='Fuel mass density (kg/m^3)')
    mg0 = Property(depends_on=['rhog0','rg0','h'])
    def _get_mg0(self):
        return self.rhog0*pi*self.rg0**2*self.h
    ftrit = 0.0
    fdeut = Property(depends_on=['ftrit'])
    def _get_fdeut(self):
        return 1.0 - self.ftrit
    ffuel = Property(depends_on=['fdops'])
    def _get_ffuel(self):
        return 1.0 - self.fdops.sum()
    Ng0 = Property(depends_on=['fdops','mdops','fdeut','ftrit','ffuel'])
    def _get_Ng0(self):
        return self.mg0/((md*self.fdeut+mt*self.ftrit)*self.ffuel + (self.mdops*self.fdops).sum())
    Nd0 = Property(depends_on=['fdeut','ffuel','Ng0'])
    def _get_Nd0(self):
        return self.ffuel*self.fdeut*self.Ng0
    Nt0 = Property(depends_on=['ftrit','ffuel','Ng0'])
    def _get_Nt0(self):
        return self.ffuel*self.ftrit*self.Ng0
    Ndops0 = Property(depends_on=['fdops','Ng0'])
    def _get_Ndops0(self):
        return self.fdops*self.Ng0
    Zdt = CFloat(np.array([1.0]), label='Zdt', help='Atomic number of H')
    Znucs0 = Property(depends_on=['Zdt','Zdops'])
    def _get_Znucs0(self):
        return np.append(self.Zdt, self.Zdops)
    Zgbars0 = Property(depends_on=['TgkeV0','Znucs0'])
    def _get_Zgbars0(self):
        return np.array([20.0*sqrt(self.TgkeV0)*np.ones(self.Znucs0.size), self.Znucs0]).min(0)
    Zgbar0 = Property(depends_on=['Zgbars0','Nd0','Nt0','Ndops0','Ng0'])
    def _get_Zgbar0(self):
        return np.dot(self.Zgbars0,np.append(np.array([self.Nd0+self.Nt0]), self.Ndops0)/self.Ng0)
    loir = Property(depends_on=['h'], label='loir', help='Length of imploding region')
    def _get_loir(self):
        return self.h
    
    species = Property(sepends_on=[''])
    def _get_species(self):
        species = np.zeros([self.fdops.size,3])
        for i in range(0,self.fdops.size):
            species[i] = np.array([[self.Zdops[i], self.Adops[i], self.Ndops0[i]]])
        return species
    
    # Zbeamlet/PreHeat
    deltazlec = CFloat(default_value=(1.5+0.6)*1.e-3, label='deltazlec', help='Laser entrance window channel length (m)')
    tzbl = CFloat(default_value=3.04350e-6, label='tzbl', help='Time of Z Beamlet Laser onset -- preheat start time (s)')
    tauzbl = CFloat(default_value=2.0*1.e-9, label='tauzbl', help='ZBeamlet pulse time -- preheat pulse width (s)')
    Ezbl =  CFloat(default_value=0.235*1.e3, label='Ezbl', help='Preheat energy from ZBeamlet Laser (J)')
    Pin = Property(depends_on=['Ezbl','tauzbl'], label='Pin', help='Total preheat power delivered Pin=Ezbl/tauzbl')
    def _get_Pin(self):
        return self.Ezbl/self.tauzbl # Laser power in post LEW, ignoring laser absorption
        
    # Z Beamlet Laser
    lambda_b = CFloat(default_value=527.0*1e-9, label='lambda_b', help='Laser wavelength (m)')
    xi_b = CFloat(default_value=1.75, label='xi_b')
    fsharp = CFloat(default_value=10., label='fsharp', help='Beam f-number (f/#)')
    bestfocus = CFloat(default_value=250.0*1.e-6, label='bestfocus', help='Diameter of the beam in the beam plane of best focus (m)')
    dolew = CFloat(default_value=3.5*1.e-3, label='dolew', help='Beam focus height above LEW (m)')
    
    # Z Accelerator
    Taur = CFloat(default_value=130.0*1.0e-9, label='Taur', help='Rise time of driving B_theta pulse (s)')
        
    # Reactivity Parameters
    data = [['C0', 6.6610, 6.2696, 6.2696],
            ['C1', 643.41*0.98*1.0e-22, 3.5741*1.0e-22, 3.7212*1.0e-22],
            ['C2', 15.136*1.0e-3, 5.8577*1.0e-3, 3.4127*1.0e-3],
            ['C3', 75.189*1.0e-3, 7.6822*1.0e-3, 1.9917*1.0e-3],
            ['C4', 4.6064*1.0e-3, 0.0*1.0e-3, 0.0*1.0e-3],
            ['C5', 13.500*1.0e-3, -0.002964*1.0e-3, 0.010506*1.0e-3],
            ['C6', -0.10675*1.0e-3, 0.0*1.0e-3, 0.0*1.0e-3],
            ['C7', 0.01366*1.0e-3, 0.0*1.0e-3, 0.0*1.0e-3]]
    df = pd.DataFrame(data,columns=['coeff', 'DT', 'DD,3He','DD,T'])
    df = df.set_index('coeff')
    
    # Energy Yields
    data_energy_yields = [['Qdt',(3.5 + 14.1)*1e6*qe],
            ['Qdd,3He', (0.82 + 2.45)*1e6*qe],
            ['Qdd,t', (1.01 + 3.02)*1e6*qe]]
    df_energy_yields = pd.DataFrame(data_energy_yields)
    df_energy_yields = df_energy_yields.set_index(0)
    
    # Circuit model constants
    Z0 = CFloat(default_value=0.18, label='Z0', help='(Ohms)')
    L = CFloat(default_value=9.58*1.0e-9, label='L', help='(H)')
    L0 = CFloat(default_value=5.46*1.0e-9, label='L0', help='(H)') # Crude method to adjust peak current: L0 = 10.00*1.0e-9 (H) ~ 20 MA, L0 = 18.00*1.0e-9 (H) ~ 16 MA
    C = CFloat(default_value=8.41*1.0e-9, label='C', help='(H)')
    Rloss = CFloat(default_value=100.0, label='Rloss', help='(Ohms)')
    rc = Property(depends_on=['rl0'], label='rc', help='Return can radius (m)')
    def _get_rc(self):
        # return self.rl0 + 4.0e-3
        return 12.0e-3

class Liner(HasTraits):
    
    Vl0 = Property(depends_on=['rl0','rg0','h'], label='Vl0', help='Initial liner volume')
    def _get_Vl0(self):
        return pi*(self.rl0**2-self.rg0**2)*self.h
    Phizl0 = Property(depends_on=['rg0','rl0','Bz0'], label='Phizl0', help='Initial axial magnetic flux in the liner')
    def _get_Phizl0(self):
        return pi*(self.rl0**2 - self.rg0**2)*self.Bz0
    dskin = Property(depends_on=['rhoe','Taur'], label='dskin', help='Skin depth')
    def _get_dskin(self):
        return sqrt(4.0*self.rhoe*self.Taur/(pi*mu0))
    beta = Property(depends_on=['rl0','rg0','dskin'],label='beta', help='Constant power found by forcing the amplitude of B_thetal(r) to drop by 1 e-folding time within one skin depth')
    def _get_beta(self):
        return max([1., np.abs(log(1.0/np.exp(1.0))/log(((self.rl0-self.rg0)-self.dskin)/(self.rl0-self.rg0)))])
    # Nls = Int(Nls, label='Nls', help='Number of concentric shells (Nls>=20)')
    Nli = Property(depends_on='Nls', label='Nli', help='Number of liner interfaces')
    def _get_Nli(self):
        return int(self.Nls + 1)
    ml = Property(depends_on=['rhol0','Vl0'], label='ml', help='Total liner mass')
    def _get_ml(self):
        return self.rhol0*self.Vl0
    Nl = Property(depends_on=['ml'], label='Nl', help='Number of liner particles')
    def _get_Nl(self):
        return self.ml/(self.Alu*u)
    mls = Property(depends_on=['ml','Nls'])
    def _get_mls(self):
        return self.ml/self.Nls
    rli0 = Property(dependps_on=['rg0','rhol0','mls','h','Nli'])
    def _get_rli0(self):
        rli0_ = np.zeros(int(self.Nli-2+2)) # Initialialize array
        rli0_[1] = self.rg0
        for i in range(1,len(rli0_)-1):
            rli0_[i+1] = sqrt(rli0_[i]**2 + self.mls/(pi*self.h*self.rhol0))
        #endfor
        rli0_ = np.append(rli0_,self.rl0) # Add rli0[i=Nli]=rl0
        return rli0_            
                        
class Fuel(HasTraits):
    
    def __init__(self):
        self.species = np.zeros([self.fdops.size,3])
        for i in range(0,self.fdops.size):
            self.species[i] = np.array([[self.Zdops[i], self.Adops[i], self.Ndops0[i]]])
            
    r0 = Property(depends_on=['self.rg0','Nr'], label='r', help='radial position of fuel region')
    def _get_r0(self):
        return np.linspace(0,self.rg0,Nr)
    Vg0 = Property(depends_on=['self.rg0','self.h'])
    def _get_Vg0(self):
        return pi*(self.rg0**2)*self.h
    Phizg0 = Property(depends_on=['self.rg0','self.Bz0'])
    def _get_Phizg0(self):
        return (pi*self.rg0**2)*self.Bz0
        
class ZBeamlet(HasTraits):
    
    wb = Property(depends_on=['lambda_b'], help='Laser frequency')
    def _get_wb(self):
        return 2.*pi*c/self.lambda_b
        
    cb = Property(depends_on=['dolew','cb','fsharp'])
    def _get_cb(self):
        return (self.bestfocus/(450.0e-6 - self.dolew/self.fsharp) - 1)/self.dolew
    
    deltazb = Property(depends_on=['dolew','loir','deltazlec'])
    def _get_deltazb(self):
        return (self.dolew + self.deltazlec + self.loir/2.)
    
    rb = Property(depends_on=['deltazb','fsharp','deltazlec','loir','bestfocus','cb'], help='Beam radius')
    def _get_rb(self):
        # return (1./2)*(self.deltazb/self.fsharp + self.bestfocus/(1. + self.cb*self.deltazb)) # from McBride's 2015
        return (self.bestfocus/2.)*(1 + (self.deltazb/(self.bestfocus*self.fsharp))**self.xi_b)**(1./self.xi_b) # updated correction from McBride's 2016 paper
        
    rph = Property(depends_on=['rphf','rb','rg0'], help='Beam radius')
    def _get_rph(self):
        if hasattr(self, 'rphf') == False: rph_ = self.rb
        else: rph_ = self.rphf*self.rg0
        return rph_
        
    laser_abs_trigger = CFloat(0, label='laser_abs_trigger', help='switch for calculating laser pulse profile')
    Pzbl = Property(depends_on=['Pin','tzbl','tauzbl'], help='Laser power square pulse')
    def _get_Pzbl(self):
        Pzbl_ = np.ones(self.t.size)*self.Pin
        Pzbl_[self.t<self.tzbl] = 0
        Pzbl_[self.t>self.tzbl+self.tauzbl] = 0
        return Pzbl_
    
class ReactivityParameters(HasTraits):
    coeff = Property(depends_on=['df'], label='coeff', help='Reactivity coefficients')
    def _get_coeff(self):
        return self.df
    
class EnergyYields(HasTraits):
    Q = Property(depends_on=['df_energy_yields'], label='Q', help='Energy yield parameters')
    def _get_Q(self):
        return self.df_energy_yields

class CircuitModel(HasTraits):
    pass

########################################################################################################################################################################################
####################### MagLIF ODEs ####################################################################################################################################################
########################################################################################################################################################################################

class OdeSystem(PhysicalConstants, InputDeck, Liner, Fuel, ZBeamlet, ReactivityParameters, CircuitModel, EnergyYields):
    
    # Various coefficients and fitting parameters
    aq = CFloat(default_value=3.0, label='aq', help='Arbitrary coefficient that specifies the length scale of the artificial viscocity (1-3)') # used in qls()
    xi = Property(depends_on=['nernst'], label='xi', help='Power law parameter that specifies the curvature of the dimensionless profile (with Nernst xi = 2.0, without Nernst xi = 3.5)')
    def _get_xi(self):
        if self.nernst == 1: xi_ = 2.0
        elif self.nernst == 0: xi_ = 3.5
        return xi_
    Abr = CFloat(default_value=1.57e-40, label='Abr', help='(m^3*K^(-1/2)*J/s)') # used in Pr()
    alphas = CFloat(default_value=0.9, label='alphas', help='keep in range [0.5,0.95]')
    
    # For alpha particle heating rate
    Qalpha = Property(depends_on=['qe'], label='Qalpha', help='Energy carried by each alpha particle')
    
    def _get_Qalpha(self):
        return 3.5e6*qe
        
    malpha = CFloat(6.64e-27, label='malpha', help='Mass of an alpha particle')
    Zalpha = CFloat(2.0, label='Zalpha', help='Atomic number of an alpha particle')
    valpha0 = Property(depends_on=['Qalpha','malpha'], label='valpha0', help='Birth velocity of an alpha particle')
    def _get_valpha0(self):
        return sqrt(2.0*self.Qalpha/self.malpha)
    
    gammag = CFloat(default_value=5.0/3.0, label='gammag', help='Ratio of specific heats for an ideal gas')
    
    def __init__(self,*args, **kwargs):
        super(OdeSystem, self).__init__(*args, **kwargs)
        self.preheat_ = np.array([0])
        try:
            self.species
            sum_ = 0
            for i in range(0,self.species.shape[0]):
                sum_ = sum_ + self.species[i][2]
        except AttributeError: 
            self.Ns0 = self.Ndops0
            
        self.Fxgvp, self.xgvp = dimensionless_curve(self.nernst)
        
    Phizv = Property(depends_on=['rc','rl0','Bz0'], label='Phizv', help='Axial magnetic flux in vacuum')
    def _get_Phizv(self):
        f = pi*(self.rc**2 - self.rl0**2)*self.Bz0
        return f
        
    Ethg0 = Property(depends_on=['Ng0','Zgbar0','Tg0'], label='Ethg0', help='Total initial thermal fuel energy')
    def _get_Ethg0(self):
        return (3./2)*self.Ng0*(self.Zgbar0+1.)*kb*self.Tg0
    
    Eiong0 = Property(depends_on=[], label='Eiong0', help='Total initial fuel ionization energy')
    def _get_Eiong0(self):
        try:
            self.species
            sum_ = 0
            for i in range(0,self.species.shape[0]):
                sum_ = sum_ + (13.6*qe*(self.species[i][0]**zeta)/(self.species[i][0] - min(20*sqrt((1.0e-3)*kb*self.Tg0/qe),self.species[i][0]) + 1.0)**(1.0/zeta))*self.Ns0[i]
            #endfor
            f = sum_ + (13.6*qe*(1.0**zeta)/(1.0 - min(20*sqrt((1.0e-3)*kb*self.Tg0/qe),self.species[i][0]) + 1.0)**(1.0/zeta))*self.Nd0 + (13.6*qe*(1.0**zeta)/(1.0 - min(20*sqrt((1.0e-3)*kb*self.Tg0/qe),self.species[i][0]) + 1.0)**(1.0/zeta))*self.Nt0
        except AttributeError:
            f = (13.6*qe*(1.0**zeta)/(1.0 - min(20*sqrt((1.0e-3)*kb*self.Tg0/qe),1.0) + 1.0)**(1.0/zeta))*self.Nd0 + (13.6*qe*(1.0**zeta)/(1.0 - min(20*sqrt((1.0e-3)*kb*self.Tg0/qe),1.0) + 1.0)**(1.0/zeta))*self.Nt0
        return min(f, self.Ethg0)

        
    Eg0 = Property(depends_on=['Ethg0','Eiong0'], label='Eg0', help='Total initial fuel energy')
    def _get_Eg0(self):
        return self.Ethg0 + min(self.Ethg0, self.Eiong0) # WHY DO WE TAKE THE MIN OF ETHG0 & EIONG0
    
    El0 = Property(depends_on=[], label='El0', help='Total initial liner energy')
    def _get_El0(self):
        Nl = self.ml/(self.Alu*u)
        Zl0bar = min(20.0*sqrt(self.Tl0*kb/qe*1.e-3),self.Zlnuc)
        f = (3.0/2.0)*Nl*(Zl0bar + 1)*kb*self.Tl0
        return f
    
    Els0 = Property(depends_on=['self.rli0','self.Vls','self.El0'],label='Els0',help='Initial liner energy')
    def _get_Els0(self):
        Vls0 = pi*(np.diff(self.rli0**2))*self.h
        f = np.zeros_like(Vls0)
        for s in range(1,self.Nls+1):
            f[s] = (Vls0[s]/self.Vl0)*self.El0
        #endfor
        return f
    
    Zlbars0 = Property(depends_on=['self.Els0','self.Zlbar0'],label='Zgbars0',help='Initial liner shell ionization state')
    def _get_Zlbars0(self):
        Zls = np.ones(self.Els0.size)*self.Zlbar0
        Zls[0] = 0
        return Zls
    
    Tlbars0 = Property(depends_on=['self.Els0','self.Tlbar0'],label='Tl0',help='Initial liner shell temperatures')
    def _get_Tlbars0(self):
        Tls = np.ones(self.Els0.size)*self.Tl0
        Tls[0] = 0
        return Tls
    
    def Lv(self,rl):
        """
        Standard coaxial vacuum inductance
        """
        # f = (mu0*self.h/(2.0*pi))*log(self.rc/rl) # How its defined in 2015 McBride paper
        f = (mu0*self.h/(2.0*pi))*log(self.rl0/rl)
        return f

    def Lvdot(self,rldot,rl):
        f = -(mu0*self.h/(2.0*pi))*(rldot/rl)
        return f
    
    def rls(self,rli_1,rli_2):
        f = sqrt((rli_2**2 + rli_1**2)/2.0)
        return f
    
    def Llc(self,rg,rl):
        f = ((mu0*self.h)/(2*pi*(self.beta+1)))*(1 - rg/rl)
        return f
        
    def Llcdot(self,rgdot,rldot,rg,rl):
        f = ((mu0*self.h)/(2*pi*(self.beta+1)))*((rg*rldot)/(rl**2) - rgdot/rl)
        return f
        
    def Ll(self,rg,rl):
        f = mu0*self.h*(2*self.beta*rl + rl + rg)*(rl - rg)/(4.*pi*(rl**2)*(self.beta + 1)*(2*self.beta + 1))
        return f
        
    def Lldot(self,rgdot,rldot,rg,rl):
        """
        """
        f = -mu0*self.h*(rg + self.beta*rl)*(rgdot*rl - rg*rldot)/(2.*pi*(rl**3)*(self.beta + 1)*(2*self.beta + 1))
        return f
    
    def pBthetali(self,Il_,rl,rg,rli_):
        """
        Azimuthal magnetic pressure at an interface
        """
        # f = (mu0*Il_/(2.0*pi*rl))*((rli_ - rg)/(rl - rg))**self.beta
        Bthetals = (mu0*Il_/(2.0*pi*rl))*((rli_ - rg)/(rl - rg))**self.beta
        f = (Bthetals**2)/(2.0*mu0)
        return f
    
    def Phizl(self,Phizg):
        """
        Axial magnetic flux in the liner
        """
        f = self.Phizl0 + (self.Phizg0 - Phizg) # in paper
        # f = self.Phizl0 - (self.Phizg0 - Phizg) # in McBride's code
        return f
        
    def pBthetalv(self,Il_,rl):
        """
        Azimuthal magnetic pressure in the vacuum
        """
        f = (mu0*Il_**2)/(8.0*(pi**2)*rl**2)
        return f
    
    def pBzvbar(self,rl):
        """
        Average magnetic pressure in the vacuum
        """
        f = ((self.Phizv/(pi*(self.rc**2 - rl**2)))**2)/(2.0*mu0)
        return f
        
    def pBbarzg(self,Phizg,rg):
        """
        Average axial magnetic pressure in the fuel
        """
        f = (Phizg/(pi*rg**2))**2/(2*mu0)
        return f
    
    def Tlbar(self,Els_sum):
        """
        Average liner temperature
        """
        # Zlbar = min(20.0*sqrt(self.Tlbar_old*kb/qe*1.0e-3),self.Zlnuc)
        Zlbar = self.Zlbar_old
        return (2.0/3.0)*(u*self.Alu*Els_sum/(self.ml*(1+Zlbar)*kb))
    
    def PBB(self,Els_sum,rl):
        """
        Blackbody power loss
        """
        f = sigma*(self.Tlbar(Els_sum)**4)*2.0*pi*rl*self.h
        return f
    
    def Eions(self,Znucs):
        """
        Ionization energy for species 's'
        """
        f = 13.6*qe*(Znucs**zeta)/(Znucs - self.Zsbar(Znucs) + 1.0)**(1.0/zeta)
        return f
        
    def Eiong(self,Nd,Nt,Ns,Eg):
        """
        Ionization energy of the fuel
        """
        try:
            self.species
            sum_ = 0
            for i in range(0,self.species.shape[0]):
                sum_ = sum_ + self.Eions(self.species[i][0])*Ns[i]
            #endfor
            f = sum_ + self.Eions(1.0)*Nd + self.Eions(1.0)*Nt
        except AttributeError:
            f = self.Eions(1.0)*Nd + self.Eions(1.0)*Nt
        if f >= Eg:
            f = 0.5*Eg;
        return f
        
    def Ethg(self,Eg,Nd,Nt,Ns):
        """
        Thermal energy in the fuel
        """
        return Eg - min(Eg*0.5,self.Eiong_old)
    
    def Zsbar(self,Znucs):
        """
        Average ionization state for species 's'
        """
        f = min(20*sqrt((1.0e-3)*kb*self.Tgbar_old/qe),Znucs)
        return f
    
    def Zgbar(self,Nd,Nt,Ns):
        """
        Average ionization state of the fuel
        """
        # try:
        #     self.species
        #     sum_ = 0
        #     for i in range(0,self.species.shape[0]):
        #         sum_ = sum_ + (1/self.Ni(Nd,Nt,Ns))*self.Zsbar(self.species[i][0])*Ns[i]
        #     #endfor
        #     f = sum_ + (1/self.Ni(Nd,Nt,Ns))*self.Zsbar(1.0)*Nd + (1/self.Ni(Nd,Nt,Ns))*self.Zsbar(1.0)*Nt
        # except AttributeError:
        #     f = (1/self.Ni(Nd,Nt,Ns))*self.Zsbar(1.0)*Nd + (1/self.Ni(Nd,Nt,Ns))*self.Zsbar(1.0)*Nt
        # return f
        return self.Zgbar_old
                        
    def Ne(self,Nd,Nt,Ns):
        """
        Number of electrons
        """
        return self.Ni(Nd,Nt,Ns)*self.Zgbar(Nd,Nt,Ns)
    
    def Ni(self,Nd,Nt,Ns):
        """
        Number of ions
        """
        try:
            self.species
            sum_ = 0
            for i in range(0,self.species.shape[0]):
                sum_ = sum_ + Ns[i]
            #endfor
            f = sum_ + Nd + Nt
        except AttributeError:
            f = Nd + Nt
        return f
                
    def Tgbar(self,Nd,Nt,Ns,Eg):
        """
        Average  fuel temperature
        """
        f = (2./3.)*(self.Ethg_new/((self.Ni(Nd,Nt,Ns) + self.Ne(Nd,Nt,Ns))*kb)) # in paper
        return f
        
    def Vg(self,rg):
        """
        Fuel volume
        """
        f = pi*(rg**2)*self.h
        return f
        
    def rhogbar(self,rg):
        """
        Average fuel density
        """
        f = self.mg/self.Vg(rg)
        return f
    
    def Pr(self,Nd,Nt,Ns,Eg,fmh,rg,t_):
        """
        Radiativ power loss
            Tc - central peak temperature
            TB - brightness temperature
        """
        if Eg < self.Eion:
            self.TB = self.Tgbar_new
            self.Tc = self.Tgbar_new
            self.Tg = self.Tgbar_new
            self.Th = self.Tgbar_new
            self.rh = rg
            self.rhoc = self.rhogbar_new
            self.rhog = self.rhogbar_new
            self.ni = self.Ni(Nd,Nt,Ns)*self.rhog/self.mg
            self.ne = self.ni*self.Zgbar(Nd,Nt,Ns)
            # Clear previous
            try:
                self.r
                del self.r
                del self.dTgdr
            except AttributeError: pass
            Prv = 0
        else:
            if fmh < 1: # Calculate hot spot radial profile
                rh = rg/2.;
                rhi = rg;
                rlo = 0.;
                eps = 1.0
                imax = 200
                iteration = 0
                while eps > 0.01 and iteration <= imax:
                    # print(iteration)
                    # if iteration > 10: set_trace()
                    iteration += 1
                    TB = (4./9)*((2.*pi*self.h*self.rhogbar_new*self.Tgbar_new)/(rh**(1./4.)))*(rg**(9./4.) - rh**(9./4.))/((1 - fmh)*self.mg)
                    rhoB = self.rhogbar_new*self.Tgbar_new/TB
                    rhohbar = fmh*self.mg/(pi*(rh**2)*self.h)
                    Th = self.Tgbar_new*self.rhogbar_new/rhohbar
                    # rhoB = rhohbar*Th/TB
                    if TB >= Th:
                        rlo = rh;
                        rh = (rlo+rhi)/2.;
                    else:
                        xh = interp1d(self.Fxgvp,self.xgvp)(rhohbar/rhoB)
                        rhoc = rhoB*(1.0 - xh**(self.xi))
                        Tc = rhoB*TB/rhoc
                        # Tc = Th*rhohbar/rhoc
                        Nrs=max(3.0,round(self.Nr*(rg-rh)/rg))
                        Nrh = self.Nr - Nrs
                        r = nonlinspace(0,rh,Nrh,0.05)
                        rhog = rhoc*(1.0 - ((r/rh)**self.xi)*(1 - (TB/Tc)))**(-1.) # for 0 <= r <= rh <= rg
                        ni = self.Ni(Nd,Nt,Ns)*rhog/self.mg
                        ne = ni*self.Zgbar(Nd,Nt,Ns)
                        Tg = Tc*(1-((r/rh)**self.xi)*(1 - (TB/Tc)))
                        Prv = self.Abr*2*pi*self.h*(self.Zgbar(Nd,Nt,Ns))**2*integrate.trapz(ni*ne*sqrt(Tg)*(1 - (TB/Tg)**4)*r,r)
                        Prs = (1-self.alphas)*sigma*(TB**4.)*2.*pi*rh*self.h
                        eps = abs((Prs - Prv)/max([Prs,Prv]))
                        if eps < 0.01: # this may be redundant because while loop should break the loop when eps < 0.01
                            break
                        elif Prs > Prv:
                            rlo = rh
                            rh = (rlo+rhi)/2
                        elif Prs < Prv:
                            rhi = rh
                            rh = (rlo+rhi)/2
                if rh == rg: # No shelf region, calculate raidal profile in fuel region
                    fmh = 1 # Forces fmh == 1 condition on next while loop iteration --> replaces Pr calculation assuming no shelf region
                    TB = self.Tgbar_new/2.0
                    TBhi = self.Tgbar_new
                    TBlo = 0
                    rh = rg
                    eps = 1.0
                    imax = 200
                    iteration = 0
                    while eps > 0.01 and iteration <= imax:
                        iteration += 1
                        rhoB = self.rhogbar_new*self.Tgbar_new/TB
                        rhohbar = fmh*self.mg/(pi*(rh**2)*self.h)
                        Th = self.Tgbar_new*self.rhogbar_new/rhohbar
                        # rhoB = rhohbar*Th/TB
                        xh = interp1d(self.Fxgvp,self.xgvp)(rhohbar/rhoB)
                        rhoc = rhoB*(1.0 - xh**(self.xi))
                        Tc = rhoB*TB/rhoc
                        # Tc = Th*rhohbar/rhoc
                        Nrs=max(3.0,round(self.Nr*(rg-rh)/rg))
                        Nrh = self.Nr - Nrs
                        r = nonlinspace(0,rh,Nrh,0.05)
                        rhog = rhoc*(1.0 - ((r/rh)**self.xi)*(1 - (TB/Tc)))**(-1.) # for 0 <= r <= rh <= rg
                        ni = self.Ni(Nd,Nt,Ns)*rhog/self.mg
                        ne = ni*self.Zgbar(Nd,Nt,Ns) # define Zgbar from EQUATION # 69
                        Tg = Tc*(1-((r/rh)**self.xi)*(1 - (TB/Tc)))
                        dTgdr = -(-(self.xi/r[1:])*(Tc - TB)*(r[1:]/rh)**self.xi) # the negative sign reflects McBride's code
                        dTgdr = np.append([np.nan],dTgdr)
                        Prv = self.Abr*2*pi*self.h*(self.Zgbar(Nd,Nt,Ns))**2*integrate.trapz(ni*ne*sqrt(Tg)*(1 - (TB/Tg)**4)*r,r)
                        Prs = (1-self.alphas)*sigma*(TB**4.)*2.*pi*rh*self.h
                        eps = abs((Prs - Prv)/max([Prs,Prv]))
                        if eps < 0.01: # this may be redundant because while loop should break the loop when eps < 0.01
                            break
                        elif Prs < Prv:
                            TBlo = TB
                            TB = (TBlo+TBhi)/2
                        elif Prs > Prv:
                            TBhi = TB
                            TB = (TBlo+TBhi)/2
                else: # Compute radial profile for the shelf region and append to hot spot profile
                    try: Tg
                    except NameError: set_trace()
                    Tg_hot = Tg
                    rhog_hot = rhog
                    ni_hot = ni
                    ne_hot = ne
                    r_hot = r
                    dTgdr_hot = -(-(self.xi/r_hot[1:])*(Tc - TB)*(r_hot[1:]/rh)**self.xi) # the negative sign reflects McBride's code
                    dTgdr_hot = np.append([np.nan],dTgdr_hot)
                    r_shelf = np.linspace(rh,rg,Nrs+1)
                    Tg_shelf = TB*(rh/r_shelf)**(1.0/4)
                    dTgdr_shelf = -(-Tg_shelf/(4.0*r_shelf)) # the negative sign reflects McBride's code
                    rhog_shelf = self.rhogbar_new*self.Tgbar_new/Tg_shelf
                    ni_shelf = self.Ni(Nd,Nt,Ns)*rhog_shelf/self.mg
                    ne_shelf = ni_shelf*self.Zgbar(Nd,Nt,Ns)
                    r = np.append(r_hot, r_shelf[1:])
                    Tg = np.append(Tg_hot, Tg_shelf[1:])
                    rhog = np.append(rhog_hot, rhog_shelf[1:])
                    dTgdr = np.append(dTgdr_hot, dTgdr_shelf[1:])
                    ni = np.append(ni_hot, ni_shelf[1:])
                    ne = np.append(ne_hot, ne_shelf[1:])
                    self.stop = 1
            elif fmh == 1: # No shelf region, calculate radial profile in fuel
                TB = self.Tgbar_new/2.0
                TBhi = self.Tgbar_new
                TBlo = 0
                rh = rg
                eps = 1.0
                imax = 200
                iteration = 0
                while eps > 0.01 and iteration <= imax:
                    iteration += 1
                    rhoB = self.rhogbar_new*self.Tgbar_new/TB
                    rhohbar = fmh*self.mg/(pi*(rh**2)*self.h)
                    Th = self.Tgbar_new*self.rhogbar_new/rhohbar
                    # rhoB = rhohbar*Th/TB
                    xh = interp1d(self.Fxgvp,self.xgvp)(rhohbar/rhoB)
                    rhoc = rhoB*(1.0 - xh**(self.xi))
                    Tc = rhoB*TB/rhoc
                    # Tc = Th*rhohbar/rhoc
                    Nrs=max(3.0,round(self.Nr*(rg-rh)/rg))
                    Nrh = self.Nr - Nrs
                    r = nonlinspace(0,rh,Nrh,0.05)
                    rhog = rhoc*(1.0 - ((r/rh)**self.xi)*(1 - (TB/Tc)))**(-1.) # for 0 <= r <= rh <= rg
                    ni = self.Ni(Nd,Nt,Ns)*rhog/self.mg
                    ne = ni*self.Zgbar(Nd,Nt,Ns) # define Zgbar from EQUATION # 69
                    Tg = Tc*(1-((r/rh)**self.xi)*(1 - (TB/Tc)))
                    dTgdr = -(-(self.xi/r[1:])*(Tc - TB)*(r[1:]/rh)**self.xi) # the negative sign reflects McBride's code
                    dTgdr = np.append([np.nan],dTgdr)
                    Prv = self.Abr*2*pi*self.h*(self.Zgbar(Nd,Nt,Ns))**2*integrate.trapz(ni*ne*sqrt(Tg)*(1 - (TB/Tg)**4)*r,r)
                    Prs = (1-self.alphas)*sigma*(TB**4.)*2.*pi*rh*self.h
                    eps = abs((Prs - Prv)/max([Prs,Prv]))
                    if eps < 0.01: # this may be redundant because while loop should break the loop when eps < 0.01
                        break
                    elif Prs < Prv:
                        TBlo = TB
                        TB = (TBlo+TBhi)/2
                    elif Prs > Prv:
                        TBhi = TB
                        TB = (TBlo+TBhi)/2
            self.rh = rh
            self.Th = Th
            self.TB = TB
            self.Tc = Tc
            self.rhoc = rhoc
            self.rhog = rhog
            self.ni = ni
            self.ne = ne
            self.r = r
            self.Tg = Tg
            self.dTgdr = dTgdr

        return Prv

    def Bzgbar(self,Phizg,rg):
        """
        Average axial magnetic field in fuel region
        """
        f = (Phizg/(pi*rg**2))
        return f
        
    def Bzg(self,Phizg,rg):
        """
        Axial magnetic field in fuel region
        """
        f = self.Bzgbar(Phizg,rg)*(self.rhog/self.rhogbar_new) # This self.rhog is for 0 <= r <= rh <= rg
        return f
            
    def nuei(self,Ns,Nd,Nt):
        """
        Electron-ion collision frequency
        """
        try:
            self.species
            sum_ = 0
            for i in range(0,self.species.shape[0]):
                # set_trace()
                sum_ = sum_ + (Ns[i]*self.rhog/self.mg)*self.Zsbar(self.species[i][0])**2
            #endfor
            # set_trace()
            sum_ = sum_ + (Nd*self.rhog/self.mg)*self.Zsbar(1.)**2 + (Nt*self.rhog/self.mg)*self.Zsbar(1.)**2
            lambdaD = sqrt(eps0*kb*self.Tg/((qe**2)*(self.ne + sum_)))
        except AttributeError:
            sum_ = (Nd*self.rhog/self.mg)*self.Zsbar(1.)**2 + (Nt*self.rhog/self.mg)*self.Zsbar(1.)**2
            lambdaD = sqrt(eps0*kb*self.Tg/((qe**2)*(self.ne + sum_)))
        # set_trace()
        bmincl = self.Zgbar(Nd,Nt,Ns)*(qe**2)/(4.*pi*eps0*3*kb*self.Tg)
        vTe = sqrt(2.*kb*self.Tg/me)
        bminqm = hbar/(2*me*vTe)
        bmin = np.array([bmincl,bminqm]).max(0)
        logLambda = np.array([np.ones(bmin.size),log(lambdaD/bmin)]).max(0) # Equation A1 here
        f = (4.*sqrt(2.*pi)*sum_*(qe**4)*logLambda)/(((4*pi*eps0)**2)*3.*sqrt(me)*(kb*self.Tg)**(3./2.))
        return f
        
    def nuii(self,Ns,Nd,Nt):
        """
        Ion-ion collision frequency
        """
        Zhbar = self.Zsbar(1.) # Zhbar = Zdbar = Ztbar
        nh = (Nd*self.rhog/self.mg) + (Nt*self.rhog/self.mg)
        mhbar = (Nd*md+Nt*mt)/(Nd+Nt)
        try:
            self.species
            sum_ = 0
            for i in range(0,self.species.shape[0]):
                sum_ = sum_ + (Ns[i]*self.rhog/self.mg)*self.Zsbar(self.species[i][0])**2
            #endfor
            sum2_ = 0 + sum_
            sum_ = sum_ + (Nd*self.rhog/self.mg)*self.Zsbar(1.)**2 + (Nt*self.rhog/self.mg)*self.Zsbar(1.)**2
            lambdaD = sqrt(eps0*kb*self.Tg/((qe**2)*(self.ne + sum_)))
        except AttributeError:
            sum2_ = 0
            sum_ = (Nd*self.rhog/self.mg)*self.Zsbar(1.)**2 + (Nt*self.rhog/self.mg)*self.Zsbar(1.)**2
            lambdaD = sqrt(eps0*kb*self.Tg/((qe**2)*(self.ne + sum_)))
        bmincl = self.Zgbar(Nd,Nt,Ns)*(qe**2)/(4.*pi*eps0*3*kb*self.Tg)
        vTe = sqrt(2.*kb*self.Tg/me)
        bminqm = hbar/(2.0*me*vTe)
        bmin = np.array([bmincl,bminqm]).max(0)
        logLambda = np.array([np.ones(bmin.size),log(lambdaD/bmin)]).max(0) # Equation A1 here
        f = ((4.*sqrt(pi)*nh*(Zhbar**4)*(qe**4)*logLambda)/(((4.*pi*eps0)**2)*3.*sqrt(mhbar)*(kb*self.Tg)**(3./2.)))*(1. + (sqrt(2.)/(nh*(Zhbar**2)))*sum2_)
        return f
        
    def kappai(self,Phizg,rg,Ns,Nd,Nt):
        """
        Coefficient for ion thermal conduction
        """
        mibar = self.mg/self.Ni(Nd,Nt,Ns)
        x = (self.Zgbar(Nd,Nt,Ns)*qe*self.Bzg(Phizg,rg)/mibar)*(1./self.nuii(Ns,Nd,Nt))
        f = ((self.ni*kb*self.Tg*(1./self.nuii(Ns,Nd,Nt)))/mibar)*((2.645+2*x**2)/(0.677+2.70*x**2+x**4))
        return f
        
    def kappae(self,Phizg,rg,Ns,Nd,Nt):
        """
        Coefficient for electron thermal conduction
        """
        x = (qe*self.Bzg(Phizg,rg)/me)*(1./self.nuei(Ns,Nd,Nt))
        f = (self.ne*kb*self.Tg*(1./self.nuei(Ns,Nd,Nt))/me)*((6.18+4.66*x)/(1.93+2.31*x+5.35*x**2+x**3))
        return f
        
    def Pcitilda(self,Phizg,rg,Ns,Nd,Nt):
        """
        Thermal conductive power loss from ions
        """
        f = 2.*pi*self.r*self.h*self.kappai(Phizg,rg,Ns,Nd,Nt)*kb*self.dTgdr
        return f[1:]
        
    def Pcetilda(self,Phizg,rg,Ns,Nd,Nt):
        """
        Thermal conductive power loss from electrons
        """
        f = 2.*pi*self.r*self.h*self.kappae(Phizg,rg,Ns,Nd,Nt)*kb*self.dTgdr
        return f[1:]
    
    def Pc(self,Phizg,rg,Ns,Nd,Nt,Eg):
        """
        Thermal conductive power loss
        """
        if Eg <= self.Eion and self.Th <= self.TB:
            Pc = 0
            Pch = 0
            Pcs = 0
        else:
            Pcitilda = self.Pcitilda(Phizg,rg,Ns,Nd,Nt)
            Pcetilda = self.Pcetilda(Phizg,rg,Ns,Nd,Nt)
            Pctilda = Pcetilda + Pcitilda
            Pctilda_max = np.where(Pctilda == max(Pctilda[self.r[1:] <= self.rh]))
            Pchi = Pcitilda[Pctilda_max]
            Pche = Pcetilda[Pctilda_max]
            Pch = Pchi + Pche
            if self.rh < rg:
                Pcsi = Pcitilda[-1]
                Pcse = Pcetilda[-1]
                Pc = Pcsi + Pcse
            elif self.rh == rg:
                Pcsi = Pchi
                Pcse = Pche
                Pc = Pcsi + Pcse
            Pcs = Pcsi + Pcse
        return Pc, Pch, Pcs

    
    def falpha(self,rg,Eg,Nd,Nt,Ns,Phizg):
        """
        Fuel heating (rate) via magnetized alpha particle energy deposition
        """
        try:
            self.species
            sum_ = 0
            for i in range(0,self.species.shape[0]):
                sum_ = sum_ + (Ns[i]/self.Vg_new)*self.Zsbar(self.species[i][0])**2
            #endfor
            sum_ = sum_ + (Nd/self.Vg_new)*self.Zsbar(1.)**2 + (Nt/self.Vg_new)*self.Zsbar(1.)**2
            lambdaD = sqrt(eps0*kb*self.Tgbar_new/((qe**2)*(self.Ne(Nd,Nt,Ns)/self.Vg_new + sum_)))
        except AttributeError:
            sum_ = (Nd/self.Vg_new)*self.Zsbar(1.)**2 + (Nt/self.Vg_new)*self.Zsbar(1.)**2
            lambdaD = sqrt(eps0*kb*self.Tgbar_new/((qe**2)*(self.Ne(Nd,Nt,Ns)/self.Vg_new + sum_)))
        bmincl = self.Zgbar(Nd,Nt,Ns)*(qe**2)/(4.*pi*eps0*3*kb*self.Tgbar_new)
        vTe = sqrt(2.*kb*self.Tgbar_new/me)
        bminqm = hbar/(2*me*vTe)
        bmin = np.array([bmincl,bminqm]).max(0)
        logLambda = np.array([np.ones(bmin.size),log(lambdaD/bmin)]).max(0) # Equation A1 here
        ralphaL = (self.malpha*self.valpha0)/(self.Zalpha*qe*self.Bzgbar(Phizg,rg))
        b = rg/ralphaL
        lalpha = ((4.0*pi*eps0)**2)*(3.0/(4.0*sqrt(2.0*pi)))*(self.malpha*self.valpha0*((kb*self.Tgbar_new)**(3.0/2))/((self.Ne(Nd,Nt,Ns)/self.Vg_new)*(self.Zalpha**2)*(qe**4)*sqrt(me)*logLambda))
        xalpha = (8.0/3)*((rg/lalpha) + (b**2)/sqrt(9.0*(b**2) + 1000.0))
        falpha = (xalpha + xalpha**2)/(1 + 13.0*xalpha/9 + xalpha**2)
        return falpha
        
    def PpdV(self,rg,rgdot,Eg,Nd,Nt,Ns):
        """
        Adiabatic heating rate
        """
        Vgdot=2.*pi*rg*rgdot*self.h;
        func=-(2./3*self.Ethg_new/self.Vg_new)*Vgdot
        return func
    
    def Etopdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from top of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rLEC < rg:
                r_ = self.r[self.r < self.rLEC]
                r_ = np.append(r_, self.rLEC)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/rhog_)
            func = ((3.0/4)**4)*(Eg/self.Vg_new)*integrate.trapz(cg*2.0*pi*r_,r_)
        else:
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/self.rhogbar_new)
            func = pi*(Eg/self.Vg_new)*cg*min([self.rLEC,rg])**2
        return func
        
    def Ebotdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from bottom of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rdump < rg:
                r_ = self.r[self.r < self.rdump]
                r_ = np.append(r_, self.rdump)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/rhog_)
            # cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            func = ((3.0/4)**4)*(Eg/self.Vg_new)*integrate.trapz(cg*2.0*pi*r_,r_)
        else:
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/self.rhogbar_new)
            # cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhog))
            func = pi*(Eg/self.Vg_new)*cg*min([self.rdump,rg])**2
        return func
        
    def Eendsdot(self,rg,Eg,Nd,Nt,Ns,t_):
        func = self.Etopdot(rg,Eg,Nd,Nt,Ns) + self.Ebotdot(rg,Eg,Nd,Nt,Ns)
        return func
        
    def TopEndLosses(self,rg,Eg,Nd,Nt,Ns):
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rLEC < rg:
                r_ = self.r[self.r < self.rLEC]
                r_ = np.append(r_, self.rLEC)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/rhog_)
            # cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            nd = Nd*rhog_/self.mg
            nt = Nt*rhog_/self.mg
            ns = Ns.sum()*rhog_/self.mg
            Eloss = ((3.0/4)**4)*(Eg/self.Vg_new)*integrate.trapz(cg*2.0*pi*r_,r_)
            Ndloss = ((3.0/4)**4)*integrate.trapz(nd*cg*2.0*pi*r_, r_)
            Ntloss = ((3.0/4)**4)*integrate.trapz(nt*cg*2.0*pi*r_, r_)
            Nsloss = ((3.0/4)**4)*integrate.trapz(ns*cg*2.0*pi*r_, r_)
        else:
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/self.rhogbar_new)
            # cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            nd = Nd*self.rhogbar_new/self.mg
            nt = Nt*self.rhogbar_new/self.mg
            ns = Ns.sum()*self.rhogbar_new/self.mg
            Eloss = pi*(Eg/self.Vg_new)*cg*min([self.rLEC,rg])**2
            Ndloss = pi*nd*cg*min([self.rLEC,rg])**2
            Ntloss = pi*nt*cg*min([self.rLEC,rg])**2
            Nsloss = pi*ns*cg*min([self.rLEC,rg])**2
        return Eloss, Ndloss, Ntloss, Nsloss
        
    def BotEndLosses(self,rg,Eg,Nd,Nt,Ns):
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rdump < rg:
                r_ = self.r[self.r < self.rdump]
                r_ = np.append(r_, self.rdump)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/rhog_)
            # cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            nd = Nd*rhog_/self.mg
            nt = Nt*rhog_/self.mg
            ns = Ns.sum()*rhog_/self.mg
            Eloss = ((3.0/4)**4)*(Eg/self.Vg_new)*integrate.trapz(cg*2.0*pi*r_,r_)
            Ndloss = ((3.0/4)**4)*integrate.trapz(nd*cg*2.0*pi*r_, r_)
            Ntloss = ((3.0/4)**4)*integrate.trapz(nt*cg*2.0*pi*r_, r_)
            Nsloss = ((3.0/4)**4)*integrate.trapz(ns*cg*2.0*pi*r_, r_)
        else:
            pg = (2./3.)*(self.Ethg_new/self.Vg_new)
            cg = sqrt(self.gammag*pg/self.rhogbar_new)
            # cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            nd = Nd*self.rhogbar_new/self.mg
            nt = Nt*self.rhogbar_new/self.mg
            ns = Ns.sum()*self.rhogbar_new/self.mg
            Eloss = pi*(Eg/self.Vg_new)*cg*min([self.rdump,rg])**2
            Ndloss = pi*nd*cg*min([self.rdump,rg])**2
            Ntloss = pi*nt*cg*min([self.rdump,rg])**2
            Nsloss = pi*ns*cg*min([self.rdump,rg])**2
        return Eloss, Ndloss, Ntloss, Nsloss
        
    def Ndtopdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from top of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rLEC < rg:
                r_ = self.r[self.r < self.rLEC]
                r_ = np.append(r_, self.rLEC)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            nd = Nd*rhog_/self.mg
            func = ((3.0/4)**4)*integrate.trapz(nd*cg*2.0*pi*r_, r_)
        else:
            cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            nd = Nd*self.rhogbar_new/self.mg
            func = pi*nd*cg*min([self.rLEC,rg])**2
        return func
        
    def Ndbotdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from top of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rdump < rg:
                r_ = self.r[self.r < self.rdump]
                r_ = np.append(r_, self.rdump)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            nd = Nd*rhog_/self.mg
            func = ((3.0/4)**4)*integrate.trapz(nd*cg*2.0*pi*r_, r_)
        else:
            cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            nd = Nd*self.rhogbar_new/self.mg
            func = pi*nd*cg*min([self.rdump,rg])**2
        return func
        
    def Ndendsdot(self,rg,Eg,Nd,Nt,Ns,t_):
        func = self.Ndtopdot(rg,Eg,Nd,Nt,Ns) + self.Ndbotdot(rg,Eg,Nd,Nt,Ns)
        return func
        
    def Nttopdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from top of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rLEC < rg:
                r_ = self.r[self.r < self.rLEC]
                r_ = np.append(r_, self.rLEC)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            nt = Nt*rhog_/self.mg
            func = ((3.0/4)**4)*integrate.trapz(nt*cg*2.0*pi*r_, r_)
        else:
            cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            nt = Nt*self.rhogbar_new/self.mg
            func = pi*nt*cg*min([self.rLEC,rg])**2
        return func
        
    def Ntbotdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from top of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rdump < rg:
                r_ = self.r[self.r < self.rdump]
                r_ = np.append(r_, self.rdump)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            nt = Nt*rhog_/self.mg
            func = ((3.0/4)**4)*integrate.trapz(nt*cg*2.0*pi*r_, r_)
        else:
            cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            nt = Nt*self.rhogbar_new/self.mg
            func = pi*nt*cg*min([self.rdump,rg])**2
        return func
        
    def Ntendsdot(self,rg,Eg,Nd,Nt,Ns,t_):
        if (t_ < self.tzbl):
            func = 0
        else:
            func = self.Nttopdot(rg,Eg,Nd,Nt,Ns) + self.Ntbotdot(rg,Eg,Nd,Nt,Ns)
        return func
        
    def Nstopdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from top of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rLEC < rg:
                r_ = self.r[self.r < self.rLEC]
                r_ = np.append(r_, self.rLEC)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            ns = Ns*rhog_/self.mg
            func = ((3.0/4)**4)*integrate.trapz(ns*cg*2.0*pi*r_, r_)
        else:
            cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            ns = Ns*self.rhogbar_new/self.mg
            func = pi*ns*cg*min([self.rLEC,rg])**2
        return func
        
    def Nsbotdot(self,rg,Eg,Nd,Nt,Ns):
        """
        Energy loss rate from top of liner
        """
        if Eg > self.Eion and self.Tgbar_new > self.Tlcrit:
            if self.rdump < rg:
                r_ = self.r[self.r < self.rdump]
                r_ = np.append(r_, self.rdump)
            else:
                r_ = self.r
            rhog_ = interp1d(self.r, self.rhog)(r_)
            cg =  sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/rhog_))
            ns = Ns*rhog_/self.mg
            func = ((3.0/4)**4)*integrate.trapz(ns*cg*2.0*pi*r_, r_)
        else:
            cg = sqrt(self.gammag*(((2./3.)*(self.Ethg_new/self.Vg_new))/self.rhogbar_new))
            ns = Ns*self.rhogbar_new/self.mg
            func = pi*ns*cg*min([self.rdump,rg])**2
        return func
        
    def Nsendsdot(self,rg,Eg,Nd,Nt,Ns,t_):
        if (t_ < self.tzbl):
            func = 0
        else:
            func = self.Nstopdot(rg,Eg,Nd,Nt,Ns) + self.Nsbotdot(rg,Eg,Nd,Nt,Ns)
        return func
        
    def Fxe(self,Phizg,rg,Ns,Nd,Nt):
        xe = (qe*self.Bzg(Phizg,rg)/me)*(1./self.nuei(Ns,Nd,Nt))
        try:
            self.r
            xe = interp1d(self.r, xe)(self.r.max())
        except AttributeError:
            pass
        return (1.5*xe**3 + 3.053*xe)/(xe**4 + 14.79*xe**2 + 3.7703)
        
    def dTgdr_rg(self,rg):
        try:
            self.r
            func = self.dTgdr[-1]
        except AttributeError:
            func = -(-(self.xi/rg)*(self.Tc - self.TB)*(rg/self.rh)**self.xi) # reflects the "-1" being dropped in McBride's code
        return func
        
    def Thbar(self,fmh,rg,Nd,Nt,Ns,Eg):
        rhohbar = fmh*self.mg/(pi*(self.rh**2)*self.h)
        return self.rhogbar_new*self.Tgbar_new/(rhohbar)
        
    def Tsbar(self,fmh,rg,Nd,Nt,Ns,Eg):
        rhosbar = (1-fmh)*self.mg/(pi*(rg**2-self.rh**2)*self.h)
        return self.rhogbar_new*self.Tgbar_new/(rhosbar)
        
    def mshdot(self,Eg,Phizg,rg,Nd,Nt,Ns, fmh):
        if Eg <= self.Eion:
            # Pcs = 0
            # Pch = 0
            func = 0
        else:
            if fmh == 1:
                func = 0
            else:
                if self.Th > self.TB:
                    # Pcitilda = self.Pcitilda(Phizg,rg,Ns,Nd,Nt)
                    # Pcetilda = self.Pcetilda(Phizg,rg,Ns,Nd,Nt)
                    # Pctilda = Pcetilda + Pcitilda
                    # Pctidla_max = np.where(Pctilda == max(Pctilda[self.r[1:] <= self.rh]))
                    # Pchi = Pcitilda[Pctidla_max]
                    # Pche = Pcetilda[Pctidla_max]
                    # if self.rh < rg:
                    #     Pcsi = Pcitilda[-1]
                    #     Pcse = Pcetilda[-1]
                    # elif self.rh == rg:
                    #     Pcsi = Pchi
                    #     Pcse = Pche
                    # Pcs = Pcsi + Pcse
                    # Pch = Pchi + Pche
                    func = (2.0/3)*(self.mg/(self.Ni(Nd,Nt,Ns)+self.Ne(Nd,Nt,Ns)))*(self.Pch-self.Pcs)/(kb*(self.Th-self.TB)) # in McBride's code
                else:
                    func = 0
        return func
        
    def Ndtdot(self,rg,Eg,Nd,Nt,Ns):
        if Eg > self.Eion and self.Th > self.Tlcrit:
            try: self.r
            except AttributeError: set_trace()
            r_ = self.r[self.r <= rg]
            rhog_ = interp1d(self.r, self.rhog)(r_)
            nt = Nt*rhog_/self.mg
            nd = Nd*rhog_/self.mg
            TgkeV = (kb*self.Tg[self.r <= rg]/qe)*1.0e-3
            zeta_reactivity_param = 1 - (self.coeff.loc['C2','DT']*(TgkeV)+self.coeff.loc['C4','DT']*(TgkeV**2)+self.coeff.loc['C6','DT']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DT']*(TgkeV)+self.coeff.loc['C5','DT']*(TgkeV**2)+self.coeff.loc['C7','DT']*(TgkeV**3))
            eta_reactivity_param = self.coeff.loc['C0','DT']/(TgkeV**(1.0/3))
            reactivity_dt = self.coeff.loc['C1','DT']*(zeta_reactivity_param**(-5.0/6))*(eta_reactivity_param**2)*np.exp(-3.0*(zeta_reactivity_param**(1.0/3))*eta_reactivity_param)
            func = self.h/2.*integrate.trapz(nd*nt*reactivity_dt*2.0*pi*r_, r_)
        else:
            nt = Nt*self.rhogbar_new/self.mg
            nd = Nd*self.rhogbar_new/self.mg
            TgkeV = (kb*self.Tgbar_new/qe)*1.0e-3
            zeta_reactivity_param = 1 - (self.coeff.loc['C2','DT']*(TgkeV)+self.coeff.loc['C4','DT']*(TgkeV**2)+self.coeff.loc['C6','DT']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DT']*(TgkeV)+self.coeff.loc['C5','DT']*(TgkeV**2)+self.coeff.loc['C7','DT']*(TgkeV**3))
            eta_reactivity_param = self.coeff.loc['C0','DT']/(TgkeV**(1.0/3))
            reactivity_dt = self.coeff.loc['C1','DT']*(zeta_reactivity_param**(-5.0/6))*(eta_reactivity_param**2)*np.exp(-3.0*(zeta_reactivity_param**(1.0/3))*eta_reactivity_param)
            # func = self.h/2.*2.0*pi*nd*nt*reactivity_dt*(rg**2)
            func = nd*nt*reactivity_dt*1/(self.rhogbar_new/self.mg)
        return func
        
    def Ndd3Hedot(self,rg,Eg,Nd,Nt,Ns):
        if Eg > self.Eion and self.Th > self.Tlcrit:
            r_ = self.r[self.r <= rg]
            rhog_ = interp1d(self.r, self.rhog)(r_)
            # nt = Nt*rhog_/self.mg
            nd = Nd*rhog_/self.mg
            TgkeV = (kb*self.Tg[self.r <= rg]/qe)*1.0e-3
            zeta_reactivity_param = 1 - (self.coeff.loc['C2','DD,3He']*(TgkeV)+self.coeff.loc['C4','DD,3He']*(TgkeV**2)+self.coeff.loc['C6','DD,3He']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DD,3He']*(TgkeV)+self.coeff.loc['C5','DD,3He']*(TgkeV**2)+self.coeff.loc['C7','DD,3He']*(TgkeV**3))
            eta_reactivity_param = self.coeff.loc['C0','DD,3He']/(TgkeV**(1.0/3))
            reactivity_dd3He = self.coeff.loc['C1','DD,3He']*(zeta_reactivity_param**(-5.0/6))*(eta_reactivity_param**2)*np.exp(-3.0*(zeta_reactivity_param**(1.0/3))*eta_reactivity_param)
            func = self.h/2.0*integrate.trapz(nd*nd*reactivity_dd3He*2.0*pi*r_, r_)
        else:
            nd = Nd*self.rhogbar_new/self.mg
            TgkeV = (kb*self.Tgbar_new/qe)*1.0e-3
            zeta_reactivity_param = 1 - (self.coeff.loc['C2','DD,3He']*(TgkeV)+self.coeff.loc['C4','DD,3He']*(TgkeV**2)+self.coeff.loc['C6','DD,3He']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DD,3He']*(TgkeV)+self.coeff.loc['C5','DD,3He']*(TgkeV**2)+self.coeff.loc['C7','DD,3He']*(TgkeV**3))
            eta_reactivity_param = self.coeff.loc['C0','DD,3He']/(TgkeV**(1.0/3))
            reactivity_dd3He = self.coeff.loc['C1','DD,3He']*(zeta_reactivity_param**(-5.0/6))*(eta_reactivity_param**2)*np.exp(-3.0*(zeta_reactivity_param**(1.0/3))*eta_reactivity_param)
            func = (1./2.)*nd*nd*reactivity_dd3He*1/(self.rhogbar_new/self.mg)
        return func
            
    def Ndddot(self,rg,Eg,Nd,Nt,Ns):
        if Eg > self.Eion and self.Th > self.Tlcrit:
            r_ = self.r[self.r <= rg]
            rhog_ = interp1d(self.r, self.rhog)(r_)
            nd = Nd*rhog_/self.mg
            TgkeV = (kb*self.Tg[self.r <= rg]/qe)*1.0e-3
            zeta_reactivity_param = 1 - (self.coeff.loc['C2','DD,T']*(TgkeV) + self.coeff.loc['C4','DD,T']*(TgkeV**2) + self.coeff.loc['C6','DD,T']*(TgkeV**3))/ \
                                        (1.0 + self.coeff.loc['C3','DD,T']*(TgkeV) + self.coeff.loc['C5','DD,T']*(TgkeV**2) + self.coeff.loc['C7','DD,T']*(TgkeV**3))
            eta_reactivity_param = self.coeff.loc['C0','DD,T']/(TgkeV**(1.0/3))
            reactivity_ddt = self.coeff.loc['C1','DD,T']*(zeta_reactivity_param**(-5.0/6))*(eta_reactivity_param**2)*np.exp(-3.0*(zeta_reactivity_param**(1.0/3))*eta_reactivity_param)
            func = self.h/2.0*integrate.trapz(nd*nd*reactivity_ddt*2.0*pi*r_, r_)
        else:
            nd = Nd*self.rhogbar_new/self.mg
            TgkeV = (kb*self.Tgbar_new/qe)*1.0e-3
            zeta_reactivity_param = 1 - (self.coeff.loc['C2','DD,T']*(TgkeV)+self.coeff.loc['C4','DD,T']*(TgkeV**2)+self.coeff.loc['C6','DD,T']*(TgkeV**3))/ \
                                        (1.0+self.coeff.loc['C3','DD,T']*(TgkeV)+self.coeff.loc['C5','DD,T']*(TgkeV**2)+self.coeff.loc['C7','DD,T']*(TgkeV**3))
            eta_reactivity_param = self.coeff.loc['C0','DD,T']/(TgkeV**(1.0/3))
            reactivity_dd = self.coeff.loc['C1','DD,T']*(zeta_reactivity_param**(-5.0/6))*(eta_reactivity_param**2)*np.exp(-3.0*(zeta_reactivity_param**(1.0/3))*eta_reactivity_param)
            func = (1./2.)*nd*nd*reactivity_dd*1/(self.rhogbar_new/self.mg)
        return func
        
    def Reactionsdot(self,rg,Eg,Nd,Nt,Ns):
        if Eg > self.Eion and self.Th > self.Tlcrit:
            try: self.r
            except AttributeError: set_trace()
            r_ = self.r[self.r <= rg]
            rhog_ = interp1d(self.r, self.rhog)(r_)
            nt = Nt*rhog_/self.mg
            nd = Nd*rhog_/self.mg
            TgkeV = (kb*self.Tg[self.r <= rg]/qe)*1.0e-3
            zeta_reactivity_param = 1 - (self.coeff.loc['C2','DT']*(TgkeV)+self.coeff.loc['C4','DT']*(TgkeV**2)+self.coeff.loc['C6','DT']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DT']*(TgkeV)+self.coeff.loc['C5','DT']*(TgkeV**2)+self.coeff.loc['C7','DT']*(TgkeV**3))
            eta_reactivity_param = self.coeff.loc['C0','DT']/(TgkeV**(1.0/3))
            reactivity_dt = self.coeff.loc['C1','DT']*(zeta_reactivity_param**(-5.0/6))*(eta_reactivity_param**2)*np.exp(-3.0*(zeta_reactivity_param**(1.0/3))*eta_reactivity_param)
            Ndtreact = self.h/2.*integrate.trapz(nd*nt*reactivity_dt*2.0*pi*r_, r_)
            zeta_reactivity_param_Ndd3He = 1 - (self.coeff.loc['C2','DD,3He']*(TgkeV)+self.coeff.loc['C4','DD,3He']*(TgkeV**2)+self.coeff.loc['C6','DD,3He']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DD,3He']*(TgkeV)+self.coeff.loc['C5','DD,3He']*(TgkeV**2)+self.coeff.loc['C7','DD,3He']*(TgkeV**3))
            eta_reactivity_param_Ndd3He = self.coeff.loc['C0','DD,3He']/(TgkeV**(1.0/3))
            reactivity_dd3He = self.coeff.loc['C1','DD,3He']*(zeta_reactivity_param_Ndd3He**(-5.0/6))*(eta_reactivity_param_Ndd3He**2)*np.exp(-3.0*(zeta_reactivity_param_Ndd3He**(1.0/3))*eta_reactivity_param_Ndd3He)
            Ndd3Hereact = self.h/2.0*integrate.trapz(nd*nd*reactivity_dd3He*2.0*pi*r_, r_)
            zeta_reactivity_param_Ndd = 1 - (self.coeff.loc['C2','DD,T']*(TgkeV) + self.coeff.loc['C4','DD,T']*(TgkeV**2) + self.coeff.loc['C6','DD,T']*(TgkeV**3))/ \
                                        (1.0 + self.coeff.loc['C3','DD,T']*(TgkeV) + self.coeff.loc['C5','DD,T']*(TgkeV**2) + self.coeff.loc['C7','DD,T']*(TgkeV**3))
            eta_reactivity_param_Ndd = self.coeff.loc['C0','DD,T']/(TgkeV**(1.0/3))
            reactivity_dd = self.coeff.loc['C1','DD,T']*(zeta_reactivity_param_Ndd**(-5.0/6))*(eta_reactivity_param_Ndd**2)*np.exp(-3.0*(zeta_reactivity_param_Ndd**(1.0/3))*eta_reactivity_param_Ndd)
            Nddreact = self.h/2.0*integrate.trapz(nd*nd*reactivity_dd*2.0*pi*r_, r_)
        else:
            nt = Nt*self.rhogbar_new/self.mg
            nd = Nd*self.rhogbar_new/self.mg
            TgkeV = (kb*self.Tgbar_new/qe)*1.0e-3
            zeta_reactivity_param_dt = 1 - (self.coeff.loc['C2','DT']*(TgkeV)+self.coeff.loc['C4','DT']*(TgkeV**2)+self.coeff.loc['C6','DT']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DT']*(TgkeV)+self.coeff.loc['C5','DT']*(TgkeV**2)+self.coeff.loc['C7','DT']*(TgkeV**3))
            eta_reactivity_param_dt = self.coeff.loc['C0','DT']/(TgkeV**(1.0/3))
            reactivity_dt = self.coeff.loc['C1','DT']*(zeta_reactivity_param_dt**(-5.0/6))*(eta_reactivity_param_dt**2)*np.exp(-3.0*(zeta_reactivity_param_dt**(1.0/3))*eta_reactivity_param_dt)
            Ndtreact = nd*nt*reactivity_dt*1/(self.rhogbar_new/self.mg)
            zeta_reactivity_param_Ndd3He = 1 - (self.coeff.loc['C2','DD,3He']*(TgkeV)+self.coeff.loc['C4','DD,3He']*(TgkeV**2)+self.coeff.loc['C6','DD,3He']*(TgkeV**3))/ \
                                    (1.0+self.coeff.loc['C3','DD,3He']*(TgkeV)+self.coeff.loc['C5','DD,3He']*(TgkeV**2)+self.coeff.loc['C7','DD,3He']*(TgkeV**3))
            eta_reactivity_param_Ndd3He = self.coeff.loc['C0','DD,3He']/(TgkeV**(1.0/3))
            reactivity_dd3He = self.coeff.loc['C1','DD,3He']*(zeta_reactivity_param_Ndd3He**(-5.0/6))*(eta_reactivity_param_Ndd3He**2)*np.exp(-3.0*(zeta_reactivity_param_Ndd3He**(1.0/3))*eta_reactivity_param_Ndd3He)
            Ndd3Hereact = (1./2.)*nd*nd*reactivity_dd3He*1/(self.rhogbar_new/self.mg)
            zeta_reactivity_param_Ndd = 1 - (self.coeff.loc['C2','DD,T']*(TgkeV)+self.coeff.loc['C4','DD,T']*(TgkeV**2)+self.coeff.loc['C6','DD,T']*(TgkeV**3))/ \
                                        (1.0+self.coeff.loc['C3','DD,T']*(TgkeV)+self.coeff.loc['C5','DD,T']*(TgkeV**2)+self.coeff.loc['C7','DD,T']*(TgkeV**3))
            eta_reactivity_param_Ndd = self.coeff.loc['C0','DD,T']/(TgkeV**(1.0/3))
            reactivity_ddt = self.coeff.loc['C1','DD,T']*(zeta_reactivity_param_Ndd**(-5.0/6))*(eta_reactivity_param_Ndd**2)*np.exp(-3.0*(zeta_reactivity_param_Ndd**(1.0/3))*eta_reactivity_param_Ndd)
            Nddreact = (1./2.)*nd*nd*reactivity_ddt*1/(self.rhogbar_new/self.mg)
        return Ndtreact, Ndd3Hereact, Nddreact
        
    def Preheat(self, Pzbl, rg, Nd, Nt, Ns, t_):
          
        Ib0 = 0
        Ibh = 0
        Pdump = 0
        PLEC = 0
        eb = ((5./2)*self.kappa0*self.IbLEW0*(t_-self.tzbl)+0j)**(2./5)
        # zf = (self.deltazlec + self.h) - (5./3)*(self.IbLEW0/eb)*(t_-self.tzbl)
        # zbar = (self.deltazlec + self.h) - zf
        zbar = (5./3)*(self.IbLEW0/eb)*(t_-self.tzbl)
        # if np.isnan(zbar) == True:
        #     set_trace()
        if np.real(zbar) <= self.deltazlec:
            PLEC = Pzbl
        elif np.real(zbar) > self.deltazlec:
            Ibh = self.IbLEW0*(1.0 - self.deltazlec/zbar)**(2./3)
            # PLEC = Pzbl - (pi*self.rb**2)*Ibh
            PLEC = Pzbl - (pi*(self.rph)**2)*Ibh
            if np.real(zbar) > (self.deltazlec + self.h):
                Ib0 = self.IbLEW0*(1.0 - (self.deltazlec + self.h)/zbar)**(2./3)
                # Pdump = (pi*self.rb**2)*Ib0
                Pdump = (pi*(self.rph)**2)*Ib0
        Pph = Pzbl - PLEC - Pdump 
        self.preheat_ = Pph
        self.eb = eb
        
        return Pph
    
    def derivs(self,t_,y_,Eiong_old,Tlbar_old,Tgbar_old,mg,Eion,Zgbar_old,Zlbar_old,Zlbars_old,iterate):
        Is = y_[0]
        phic = y_[1]
        if self.VI_mode == 'V':
            Il = y_[2]
        if self.VI_mode == 'I':
            Il = interp1d(self.t, self.drive_amp, kind='linear', bounds_error=False, fill_value=0)(t_)
            # Il = UnivariateSpline(self.t, self.drive_amp, k=1)(t_)
            # Il = y_[2]
        rli = y_[3:self.Nli+4] # NOTE: rg = rli[1], rl = rli[Nli]
        rlidot = y_[self.Nli+4:2*self.Nli+5]
        Els = y_[2*self.Nli+5:3*self.Nli+5]
        Phizg = y_[3*self.Nli+5]
        # Phizg = max(samm.Bz0*pi*rli[1]**2, y_[3*self.Nli+5])
        Nd = y_[3*self.Nli+6]
        Nt = y_[3*self.Nli+7]
        Ns = y_[3*self.Nli+8:3*self.Nli+8+self.Ndops0.size]
        Eg = y_[3*self.Nli+8+self.Ndops0.size]
        fmh = y_[3*self.Nli+8+self.Ndops0.size+1]
        # Ns = y_[3*self.Nli+8]
        # Eg = y_[3*self.Nli+9]
        # fmh = y_[3*self.Nli+10]
        # Ndt = y_[3*self.Nli+11]
        # Ndd3He = y_[3*self.Nli+12]
        # Nddt = y_[3*self.Nli+13]
        for i in range(0,self.Ndops0.size):
            self.species[i][2] = Ns[i]
        
        if fmh > 1: fmh = 1
        
        # This bit was used to force a smaller time step in ode.integrate when using RK4, because of numerical issues with fmhdot
        # if fmh < 0: fmh = 1
        # if abs(fmh) > 1: fmh = 1
            
        self.Tlbar_old = Tlbar_old
        self.Tgbar_old = Tgbar_old
        self.mg = mg
        self.Eion = 13.6*(Nd + Nt + Ns.sum())*qe
        if Eiong_old >= Eg:
            self.Eiong_old = Eg/2.
        else:
            self.Eiong_old = Eiong_old
        self.Zgbar_old = Zgbar_old
        self.Zlbars_old = Zlbars_old
        self.Zlbar_old = Zlbar_old
        
        self.Tlcrit = (2./3.)*Els[1:]/(self.Nl/self.Nls*(1+Zlbars_old[1:]))/kb
        self.Tlcrit = self.Tlcrit[1]
        
        self.Ethg_new = self.Ethg(Eg,Nd,Nt,Ns)
        self.Tgbar_new = self.Tgbar(Nd,Nt,Ns,Eg)
        self.rhogbar_new = self.rhogbar(rli[1])
        self.Vg_new = self.Vg(rli[1])
        self.pg = (2./3)*self.Ethg_new/self.Vg_new
        
        self.Vls = pi*(np.diff(rli**2))*self.h
        self.rhols = self.mls/self.Vls
        
        self.pEOS = (3.0*self.A1/2.0)*((self.rhols/self.rhol0_zero)**self.gamma1 - (self.rhols/self.rhol0_zero)**self.gamma2)*(1 + (3.0/4.0)*(self.A2 - 4.0)*((self.rhols/self.rhol0_zero)**(2.0/3.0) - 1))
        
        Vl = pi*(rli[self.Nli]**2 - rli[1]**2)*self.h
        rhol = self.ml/Vl
        Bzls = self.Phizl(Phizg)/(pi*(rli[self.Nli]**2 - rli[1]**2))
        self.pBzls = ((Bzls/rhol*self.rhols)**2)/(2*mu0)
        
        self.qls = np.zeros_like(self.Vls)
        self.Vlsdot = np.zeros_like(self.Vls)
        for i in range(1,self.Nls+1):
            if rlidot[i+1] - rlidot[i] < 0:
                self.qls[i] = (self.aq**2)*(self.rhols[i])*(rlidot[i+1] - rlidot[i])**2
            else:
                self.qls[i] = 0
            self.Vlsdot[i] = 2.0*pi*self.h*(rli[i+1]*rlidot[i+1] - rli[i]*rlidot[i])
            #endif
        #endfor
        
        self.Vls[0] = 0
        self.rhols[0] = 0
        self.Vlsdot[0] = 0
        self.pEOS[0] = 0
        self.pBzls[0] = 0
        self.qls[0] = 0
        
        dydt_1 = np.zeros_like(self.rli0) # rlidot
        dydt_2 = np.zeros_like(self.rli0) # rliddot
        dydt_3 = np.zeros_like(self.Els0) # Elsdot
        
        dydt_1[1] = rlidot[1] # rgdot
        dydt_2[1] = (self.pg + \
                    self.pBbarzg(Phizg,rli[1]) - \
                    (self.pEOS[1] + (2.0/3.0)*Els[1]/(self.Vls[1]) + \
                    self.pBthetali(Il,rli[self.Nli],rli[1],rli[1]) + \
                    self.pBzls[1] + \
                    self.qls[1]))*(2*pi*rli[1]*self.h)/(self.mls/2.0 + self.mg)
                    
        for i in range(2,self.Nls+1):
            dydt_1[i] = rlidot[i]
            dydt_2[i] = ((self.pEOS[i-1] + (2.0/3.0)*Els[i-1]/(self.Vls[i-1]) - (self.pEOS[i] + (2.0/3.0)*Els[i]/(self.Vls[i]))) + \
                        (self.pBthetali(Il,rli[self.Nli],rli[1],rli[i-1]) - self.pBthetali(Il,rli[self.Nli],rli[1],rli[i])) + \
                        (self.pBzls[i-1] - self.pBzls[i]) + \
                        (self.qls[i-1] - self.qls[i]))*2.0*pi*rli[i]*self.h/self.mls
                        
        dydt_1[self.Nli] = rlidot[self.Nli]
        dydt_2[self.Nli] = (self.pEOS[self.Nls] + (2.0/3.0)*Els[self.Nls]/self.Vls[self.Nls] + \
                            (self.pBthetali(Il,rli[self.Nli],rli[1],rli[self.Nls]) - self.pBthetali(Il,rli[self.Nli],rli[1],rli[self.Nli])) + \
                            self.pBzls[self.Nls] + \
                            self.qls[self.Nls] - \
                            # self.pBthetalv(Il,rli[self.Nli]) - \ # redundant because of pBthetali
                            self.pBzvbar(rli[self.Nli]))*(2.0*pi*rli[self.Nli]*self.h)/(self.mls/2.0)
                        
        for s in range(1,self.Nls+1):
            dydt_3[s] = -((2./3.)*Els[s]/self.Vls[s] + self.qls[s])*self.Vlsdot[s]
        El_ = dydt_3[1:self.Nls+1].sum()
        
        if self.VI_mode == 'V':
            Ildot = (phic - Il*(self.Lvdot(rlidot[self.Nli],rli[self.Nli]) + self.Llcdot(rlidot[1],rlidot[self.Nli],rli[1],rli[self.Nli])))/(self.L0 + self.Lv(rli[self.Nli]) + self.Llc(rli[1],rli[self.Nli]))
        if self.VI_mode == 'I':
            Ildot = interp1d(self.t, self.drivedot, kind='linear', bounds_error=False, fill_value=0)(t_)
            # Ildot = UnivariateSpline(self.t, self.drivedot, k=1)(t_)
            # Ildot = 0
        Phildotint = self.Llcdot(rlidot[1],rlidot[self.Nli],rli[1],rli[self.Nli])*Il + self.Llc(rli[1],rli[self.Nli])*Ildot
        Phildot = self.Lvdot(rlidot[self.Nli],rli[self.Nli])*Il + self.Lv(rli[self.Nli])*Ildot + Phildotint
        PEM = Phildot*Il
        PB = (self.Lv(rli[self.Nli]) + self.Ll(rli[1],rli[self.Nli]))*Ildot*Il + (0.5)*(self.Lvdot(rlidot[self.Nli],rli[self.Nli]) + self.Lldot(rlidot[1],rlidot[self.Nli],rli[1],rli[self.Nli]))*Il**2
        # Pkin = (0.5)*(self.Lvdot(rlidot[self.Nli],rli[self.Nli]))*Il**2 # less precise, but good enough
        Pkin = sum(-2.0*pi*self.h*rli[2:]*np.diff(self.pBthetali(Il,rli[self.Nli],rli[1],rli[1:]))*rlidot[2:]) # more precise
        Pohmic = np.abs(PEM - PB - Pkin)
        
        self.Pr_new = self.Pr(Nd,Nt,Ns,Eg,fmh,rli[1],t_)
        PBB = self.PBB(Els.sum(),rli[self.Nli])
        
        dydt_3[1:self.Nls+1] = dydt_3[1:self.Nls+1] + (self.Pr_new + \
                        Pohmic - \
                        PBB)/self.Nls # EQUATION 53 HERE
                        
        if fmh > 1:
            fmh = 1
            
        self.Tau_ei = 0  
        Pph = 0
        if self.laser_abs_trigger == 1:
            Pzbl_ = interp1d(self.t,self.Pzbl, bounds_error=False, fill_value=0)(t_)
            if Pzbl_ > 0:
                if self.laserabs == 1:
                    Pph = self.Preheat(Pzbl_, rli[1], Nd, Nt, Ns, t_)
                    # self.Tau_ei = np.real(self.Tau_bar*self.eb**(3./2.))
                    self.Tau_ei = 1./self.nuei(Ns,Nd,Nt)
                elif self.laserabs == 0: Pph = Pzbl_
        
        self.Pc_new, self.Pch, self.Pcs = self.Pc(Phizg,rli[1],Ns,Nd,Nt,Eg)        
        if fmh == 1:
            # set_trace()
            dydt_3[1:self.Nls+1] = dydt_3[1:self.Nls+1] + self.Pc_new/self.Nls
            Egdot = self.PpdV(rli[1],rlidot[1],Eg,Nd,Nt,Ns) + Pph - self.Pr_new - self.Pc_new
        else:
            dydt_3[1:self.Nls+1] = dydt_3[1:self.Nls+1] + self.Pc_new/self.Nls
            Egdot = self.PpdV(rli[1],rlidot[1],Eg,Nd,Nt,Ns) + Pph - self.Pr_new - self.Pc_new

        if t_ >= self.tzbl: # save on runtime, but could be an incorrect assumption if preheat is off
            Ndtdot, Ndd3Hedot, Ndddot = self.Reactionsdot(rli[1],Eg,Nd,Nt,Ns)
            Nddot = -Ndtdot - 2.0*Ndd3Hedot - 2.0*Ndddot
            Ntdot = -Ndtdot
        else:
            Ndtdot = 0
            Ndd3Hedot = 0
            Ndddot = 0
            Nddot = 0
            Ntdot = 0
        self.Ndtdot = Ndtdot
        self.Ndd3Hedot = Ndd3Hedot
        self.Ndddot = Ndddot
        
        if Eg >= self.Eion and self.Th > self.TB:
            Phizgdot = self.nernst*(-2.0*pi*rli[1]*np.append(0,self.Fxe(Phizg,rli[1],Ns,Nd,Nt)*(kb/qe))[-1]*np.append(0,self.dTgdr_rg(rli[1]))[-1])
            Egdot = Egdot + Ndtdot*self.Qalpha*self.falpha(rli[1],Eg,Nd,Nt,Ns,Phizg)
            if fmh == 1:
                fmhdot = 0
            else:
                fmhdot = self.mshdot(Eg,Phizg,rli[1],Nd,Nt,Ns, fmh)/self.mg
                if fmhdot.size == 0:
                    fmhdot = 0
                    self.stop == True
        else:
            Phizgdot = 0
            fmhdot = 0
        
        Nsdot = np.zeros(self.Ndops0.size)
        self.E_endloss = 0.0
        self.N_endloss = 0.0  
        if t_ > self.tzbl:
            if self.n_endloss == 2:
                Etoploss, Ndtoploss, Nttoploss, Nstoploss = self.TopEndLosses(rli[1],Eg,Nd,Nt,Ns)
                Ebotloss, Ndbotloss, Ntbotloss, Nsbotloss = self.BotEndLosses(rli[1],Eg,Nd,Nt,Ns)
            elif self.n_endloss == 1:
                Etoploss, Ndtoploss, Nttoploss, Nstoploss = self.TopEndLosses(rli[1],Eg,Nd,Nt,Ns)
                Ebotloss, Ndbotloss, Ntbotloss, Nsbotloss = 0.0, 0.0, 0.0, 0.0
            elif self.n_endloss == 0:
                Etoploss, Ndtoploss, Nttoploss, Nstoploss = 0.0, 0.0, 0.0, 0.0
                Ebotloss, Ndbotloss, Ntbotloss, Nsbotloss = 0.0, 0.0, 0.0, 0.0
            Egdot = Egdot - Etoploss - Ebotloss
            Nddot = Nddot - Ndtoploss - Ndbotloss
            Ntdot = Ntdot - Nttoploss - Ntbotloss
            self.E_endloss = Etoploss + Ebotloss
            self.N_endloss = Ndtoploss + Ndbotloss + Nttoploss + Ntbotloss
            # if self.laser_abs_trigger == 1: set_trace()
            if self.Ndops0.size == 1:
                Nsdot = Nsdot - Nstoploss - Nsbotloss
            else:
                Nsdot = (Nsdot - Nstoploss - Nsbotloss)*self.Ndops0/self.Ndops0.sum()
                # Nsdot = (Nsdot - Nstoploss - Nsbotloss)*self.Ndops0/1.
        
        if self.VI_mode == 'V':       
            Voc = interp1d(self.t, self.drive_amp, kind='linear', bounds_error=False, fill_value=0 )(t_)
            Isdot = (Voc - self.Z0*Is - phic)/self.L
            Phicdot = (Is - Il - phic/self.Rloss)/self.C
                                                                   
            dydt = [Isdot, #Isdot
                    Phicdot, #phicdot
                    Ildot, #Ildot
                    dydt_1, #rlidot
                    dydt_2, #rliddot
                    dydt_3, #Elsdot
                    Phizgdot, #Phizgdot
                    Nddot, #Nddot
                    Ntdot, #Ntdot
                    Nsdot, #Nsdot
                    Egdot, #Egdot
                    fmhdot, # fmhdot
                    Ndtdot,
                    Ndd3Hedot,
                    Ndddot,
                    self.Pr_new,
                    self.E_endloss]
                    # self.Pr_new,
                    # self.E_endloss,
                    # self.N_endloss]
                    
        elif self.VI_mode == 'I':
            dydt = [0,
                    0,
                    Ildot, #Ildot
                    dydt_1, #rlidot
                    dydt_2, #rliddot
                    dydt_3, #Elsdot
                    Phizgdot, #Phizgdot
                    Nddot, #Nddot
                    Ntdot, #Ntdot
                    Nsdot, #Nsdot
                    Egdot, #Egdot
                    fmhdot, # fmhdot
                    Ndtdot,
                    Ndd3Hedot,
                    Ndddot,
                    self.Pr_new,
                    self.E_endloss]
                    # self.Pr_new,
                    # self.E_endloss,
                    # self.N_endloss]
        
        # print(t_, Egdot + (- Pph - self.PpdV(rli[1],rlidot[1],Eg,Nd,Nt,Ns) + self.E_endloss + self.Pr_new + self.Pc_new))
        # print(t_, dydt_3[1:self.Nls+1].sum() + (-El_ - Pohmic - self.Pc_new - self.Pr_new + PBB))
        
        dydt_ = []
        
        for i in range(0,len(dydt)):
            dydt_ = np.append(dydt_,dydt[i])
                    
        return dydt_

########################################################################################################################################################################################
####################### Run Class ######################################################################################################################################################
########################################################################################################################################################################################        
                        
class ModelOutputs(HasTraits):
    
    outputs = Dict(Str, Array(shape=(None)))
        
    def build_model(self, *args, **kwargs):
        samm_ = OdeSystem(*args, **kwargs)
        if samm_.save_inputs == 1:
            samm_.save()
        n = len(samm_.t)
        self.outputs['t'] = samm_.t
        self.outputs['Is'] = np.zeros(n)
        self.outputs['Vc'] = np.zeros(n)
        self.outputs['Il'] = np.zeros(n)
        if samm_.VI_mode == 'I':
            self.outputs['Il'][0] = interp1d(samm_.t, samm_.drive_amp, kind='linear', bounds_error=False, fill_value=0 )(samm_.t[0])
        self.outputs['rg'] = np.zeros(n)
        self.outputs['rg'][0] = samm_.rg0
        self.outputs['rl'] = np.zeros(n)
        self.outputs['rl'][0] = samm_.rl0
        self.outputs['rli'] = np.zeros([n,len(samm_.rli0)])
        self.outputs['rli'][0] = samm_.rli0
        self.outputs['rlidot'] = np.zeros([n,len(samm_.rli0)])
        self.outputs['Eg'] = np.zeros(n)
        self.outputs['Eg'][0] = samm_.Eg0
        self.outputs['Ethg'] = np.zeros(n)
        self.outputs['Ethg'][0] = samm_.Ethg0
        self.outputs['Eiong'] = np.zeros(n)
        self.outputs['Eiong'][0] = samm_.Eiong0
        self.outputs['mg'] = np.zeros(n)
        self.outputs['mg'][0] = samm_.mg0
        self.outputs['Els'] = np.zeros([n,len(samm_.Els0)])
        self.outputs['Els'][0] = samm_.Els0
        self.outputs['Zlbar'] = np.zeros(n)
        self.outputs['Zlbar'][0] = samm_.Zlbar0
        self.outputs['Zlbars'] = np.zeros([n,len(samm_.Zlbars0)])
        self.outputs['Zlbars'][0] = samm_.Zlbars0
        self.outputs['TlkeV'] = np.zeros(n)
        self.outputs['TlkeV'][0] = samm_.Tl0*kb/qe*1.0e-3
        self.outputs['Tl'] = np.zeros(n)
        self.outputs['Tl'][0] = samm_.Tl0
        self.outputs['Tls'] = np.zeros([n,len(samm_.Els0)])
        self.outputs['Tls'][0,1:] = np.ones(samm_.Els0[1:].size)*samm_.Tl0
        self.outputs['fmh'] = np.zeros(n)
        # self.outputs['fmh'][0] = (samm_.rb/samm_.rg0)**2
        self.outputs['fmh'][0] = ((samm_.rph)/samm_.rg0)**2
        self.outputs['Phizg'] = np.zeros(n)
        self.outputs['Phizg'][0] = samm_.Phizg0
        self.outputs['Nd'] = np.zeros(n)
        self.outputs['Nd'][0] = samm_.Nd0
        self.outputs['Nt'] = np.zeros(n)
        self.outputs['Nt'][0] = samm_.Nt0
        self.outputs['Ns'] = np.zeros([n,len(samm_.Ndops0)])
        self.outputs['Ns'][0] = samm_.Ndops0
        self.outputs['TgkeV'] = np.zeros(n)
        self.outputs['TgkeV'][0] = samm_.Tg0*kb/qe*1.0e-3
        self.outputs['Tg'] = np.zeros(n)
        self.outputs['Tg'][0] = samm_.Tg0
        self.outputs['Zgbar'] = np.zeros(n)
        self.outputs['Zgbar'][0] = samm_.Zgbar0
        self.outputs['Zgbars'] = np.zeros([n,len(samm_.Zgbars0)])
        self.outputs['Zgbars'][0] = samm_.Zgbars0
        self.outputs['Ndt'] = np.zeros(n)
        self.outputs['Ndd3He'] = np.zeros(n)
        self.outputs['Nddt'] = np.zeros(n)
        self.outputs['Yddn'] = np.zeros(n)
        self.outputs['Ydtn'] = np.zeros(n)
        self.outputs['Yn'] = np.zeros(n)
        self.outputs['Y'] = np.zeros(n)
        self.outputs['Ndtdot'] = np.zeros(n)
        self.outputs['Ndd3Hedot'] = np.zeros(n)
        self.outputs['Ndddot'] = np.zeros(n)
        self.outputs['Er'] = np.zeros(n)
        self.outputs['E_ends'] = np.zeros(n)
        self.outputs['N_ends'] = np.zeros(n)
        self.outputs['Erdot'] = np.zeros(n)
        self.outputs['Ecdot'] = np.zeros(n)
        self.outputs['E_endsdot'] = np.zeros(n)
        self.outputs['Omega'] = np.array([samm_.wb])
        self.outputs['Tau_ei'] = np.zeros(n)
        return samm_
    
    def update_ics(self, iter_):
        yinit = self.outputs['Is'][iter_]
        yinit = np.append(yinit,self.outputs['Vc'][iter_])
        yinit = np.append(yinit,self.outputs['Il'][iter_])
        yinit = np.append(yinit,self.outputs['rli'][iter_,:])
        yinit = np.append(yinit,self.outputs['rlidot'][iter_,:])
        yinit = np.append(yinit,self.outputs['Els'][iter_,:])
        yinit = np.append(yinit,self.outputs['Phizg'][iter_])
        yinit = np.append(yinit,self.outputs['Nd'][iter_])
        yinit = np.append(yinit,self.outputs['Nt'][iter_])
        yinit = np.append(yinit,self.outputs['Ns'][iter_])
        yinit = np.append(yinit,self.outputs['Eg'][iter_])
        yinit = np.append(yinit,self.outputs['fmh'][iter_])
        yinit = np.append(yinit,self.outputs['Ndt'][iter_])
        yinit = np.append(yinit,self.outputs['Ndd3He'][iter_])
        yinit = np.append(yinit,self.outputs['Nddt'][iter_])
        yinit = np.append(yinit,self.outputs['Er'][iter_])
        yinit = np.append(yinit,self.outputs['E_ends'][iter_])
        return yinit
        
    def update_dict(self,dict_key,iter_,val):
        self.outputs[dict_key][iter_] = val
        return

########################################################################################################################################################################################
####################### Run Function ###################################################################################################################################################
########################################################################################################################################################################################                                 
                                                                                                                  
def run_pySAMM(*args, **kwargs):
    """
    A function to run pySAMM
    - *args and **kwargs defined in InputDeck Class
    """
    start_time = time.time()
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    output = ModelOutputs()
    samm = output.build_model(*args, **kwargs)
    samm.stop = False
    yinits = output.update_ics(0)
    Eion_ = 13.6*(output.outputs['Nd'][0] + output.outputs['Nt'][0] + output.outputs['Ns'][0].sum())*qe
    tcsl = np.ones(samm.Nls+1)*samm.deltat*5.0
    tcsg = samm.deltat
    
    # for iterate in range(1, 2):
    for iterate in range(1, len(samm.t)):
        
        if samm.laser_abs_trigger == 0 and samm.Pzbl[iterate] > 0:
            samm.laser_abs_trigger = 1
            sum2_ = 0
            sum3_ = 0
            sum4_ = 0
            sum5_ = 0
            try:
                samm.species
                for i in range(0,samm.species.shape[0]):
                    sum2_ = sum2_ + samm.species[i][0]*output.outputs['Ns'][iterate-1,i]
                    sum3_ = sum3_ + samm.species[i][0]**zeta
                    sum4_ = sum4_ + (samm.species[i][0]**2)*output.outputs['Ns'][iterate-1,i]/samm.Vg(output.outputs['rg'][iterate-1])
                    sum5_ = sum5_ + (samm.species[i][0]**zeta)*output.outputs['Ns'][iterate-1,i]
                #endfor
                ne_bar = (sum2_ + 1.0*output.outputs['Nd'][iterate-1] + 1.0*output.outputs['Nt'][iterate-1])/samm.Vg(output.outputs['rg'][iterate-1]) # Average electron density in fuel
                ni_bar = samm.Ni(output.outputs['Nd'][iterate-1],output.outputs['Nt'][iterate-1],output.outputs['Ns'][iterate-1])/samm.Vg(output.outputs['rg'][iterate-1]) # Average ion density in fuel
                kT = (2.0/(3.0*samm.Ni(output.outputs['Nd'][iterate-1],output.outputs['Nt'][iterate-1],output.outputs['Ns'][iterate-1])))*13.6*qe*(sum5_ + (1.**zeta)*output.outputs['Nd'][iterate-1] + (1.**zeta)*output.outputs['Nt'][iterate-1])
                lambdaD = sqrt(eps0*kT/((qe**2)*(ne_bar + sum4_ + 1.*output.outputs['Nd'][iterate-1]/samm.Vg(output.outputs['rli'][iterate-1,1]) + 1.*output.outputs['Nt'][iterate-1]/samm.Vg(output.outputs['rg'][iterate-1]))))
                bmin_cl = (1.0/samm.Ni(output.outputs['Nd'][iterate-1],output.outputs['Nt'][iterate-1],output.outputs['Ns'][iterate-1]))*(sum2_ + 1.0*output.outputs['Nd'][iterate-1] + 1.0*output.outputs['Nt'][iterate-1])*(qe**2)/(4.*pi*eps0*3.*kT)
                bmin_qm = hbar/(2.0*me*sqrt(2.*kT/me))
                bmin = max([bmin_cl, bmin_qm])
                Lambda = max([1.0, log(lambdaD/bmin)]) # Coulomb Logarithm
                kTb = (2./3)*1./(ni_bar + ne_bar)
                nu_ei = 4.*sqrt(2.*pi)*(sum4_ + (1.**2)*output.outputs['Nd'][iterate-1]/samm.Vg(output.outputs['rli'][iterate-1,1]) + (1.**2)*output.outputs['Nt'][iterate-1]/samm.Vg(output.outputs['rg'][iterate-1]))*(qe**4)*Lambda/(((4.*pi*eps0)**2)*3.*sqrt(me)*(kTb)**(3./2))
            except AttributeError:
                ne_bar = (sum2_ + 1.0*output.outputs['Nd'][iterate-1] + 1.0*output.outputs['Nt'][iterate-1])/samm.Vg(output.outputs['rli'][iterate-1,1]) # Average electron density in fuel
                ni_bar = samm.Ni(output.outputs['Nd'][iterate-1],output.outputs['Nt'][iterate-1],0)/samm.Vg(output.outputs['rli'][iterate-1,1]) # Average ion density in fuel
                kT = (2.0/(3.0*samm.Ni(output.outputs['Nd'][iterate-1],output.outputs['Nt'][iterate-1],0)))*13.6*qe*(sum5_ + (1.**zeta)*output.outputs['Nd'][iterate-1] + (1.**zeta)*output.outputs['Nt'][iterate-1])
                lambdaD = sqrt(eps0*kT/((qe**2)*(ne_bar + sum4_ + 1.*output.outputs['Nd'][iterate-1]/samm.Vg(output.outputs['rli'][iterate-1,1]) + 1.*output.outputs['Nt'][iterate-1]/samm.Vg(output.outputs['rg'][iterate-1]))))
                bmin_cl = (1.0/samm.Ni(output.outputs['Nd'][iterate-1],output.outputs['Nt'][iterate-1],0))*(sum2_ + 1.0*output.outputs['Nd'][iterate-1] + 1.0*output.outputs['Nt'][iterate-1])*(qe**2)/(4.*pi*eps0*3.*kT)
                bmin_qm = hbar/(2.0*me*sqrt(2.*kT/me))
                bmin = max([bmin_cl, bmin_qm])
                Lambda = max([1.0, log(lambdaD/bmin)]) # Coulomb Logarithm
                kTb = (2./3)*1./(ni_bar + ne_bar)
                nu_ei = 4.*sqrt(2.*pi)*(sum4_ + (1.**2)*output.outputs['Ns'][iterate-1]/samm.Vg(output.outputs['rli'][iterate-1,1]) + (1.**2)*output.outputs['Nt'][iterate-1]/samm.Vg(output.outputs['rg'][iterate-1]))*(qe**4)*Lambda/(((4.*pi*eps0)**2)*3.*sqrt(me)*(kTb)**(3./2))
            #endtry
            wp = sqrt(ne_bar*qe**2/(me*eps0)) # Plasma Frequency
            samm.kappa0 = (nu_ei/c)*(wp**2/samm.wb**2)*1.0/sqrt(1.0-(wp**2/samm.wb**2))
            # samm.IbLEW0 = samm.Pin/(pi*samm.rb**2)
            samm.IbLEW0 = samm.Pin/(pi*(samm.rph)**2)
            output.outputs['wp'] = np.array([wp])
            samm.Tau_bar = np.array([1.0/nu_ei])
    
        samm.iterate_ = iterate
            
        if samm.verbose == 1:
            if iterate%100 == 0:
                # print('Current sim time = %i ns' % (samm.t[iterate]*1.0e9))
                sys.stdout.write('\r' + 'Current sim time >>> ' + str(int(samm.t[iterate]*1.0e9)) + ' ns')
    
        stepsize = np.array([tcsl[np.isnan(tcsl)==0].min()/5.,tcsg/5., samm.deltat]).min()
        if samm.t[iterate] <= samm.tzbl:
            solver = ode(samm.derivs).set_integrator('dopri5',rtol=1.0e-4, atol=1.0e-6, first_step=stepsize, max_step=stepsize) # RK4 is computationally faster before preheat, but causes numerical issues in fmhdot
        else:
            solver = ode(samm.derivs).set_integrator('vode',rtol=1.0e-4, atol=1.0e-6, first_step=stepsize, max_step=stepsize)
        solver.set_initial_value(yinits, samm.t[iterate-1]).set_f_params(output.outputs['Eiong'][iterate-1], output.outputs['Tl'][iterate-1], output.outputs['Tg'][iterate-1], output.outputs['mg'][iterate-1], Eion_, output.outputs['Zgbar'][iterate-1], output.outputs['Zlbar'][iterate-1], output.outputs['Zlbars'][iterate-1,:], iterate)
        y = solver.integrate(samm.t[iterate])
        
        # if samm.stop == True:
        #     plt.figure()
        #     plt.clf()
        #     set_trace()
        #     plt.plot(samm.t*1.e9,output.outputs['rli'][:,-1]*1e3,'k')
        #     plt.plot(samm.t*1.e9,output.outputs['rli'][:,1]*1e3,'r')
        #     plt.plot(samm.t*1.e9,0.1*output.outputs['Il']*1e-6,'y')
        #     plt.plot(samm.t*1e9,output.outputs['TgkeV'],'magenta')
        #     plt.xlim((2900, 3140))
        #     plt.ylim((0, 3))
        #     plt.legend(['Liner Outer Radius [mm]', 'Liner Inner/ Fuel Radius [mm]', 'Liner Current [1/10 MA]', 'Fuel Temp [keV]'])
        #     plt.show()
        #     break
        
        if y[0] < 0:
            y[0] = 0
        if y[2] < 0:
            y[2] = 0
        
        output.update_dict('Is',iterate,y[0])
        output.update_dict('Vc',iterate,y[1])
        output.update_dict('Il',iterate,y[2])
        rl_i = y[3:samm.Nli+4]
        rl_idot = y[samm.Nli+4:2*samm.Nli+5]
        output.update_dict('rli', iterate, rl_i)
        output.update_dict('rlidot', iterate, rl_idot)
        # output.update_dict('rli',iterate,y[3:samm.Nli+4])
        # output.update_dict('rlidot',iterate,y[samm.Nli+4:2*samm.Nli+5])
        output.update_dict('rg',iterate,output.outputs['rli'][iterate,1])
        output.update_dict('rl',iterate,output.outputs['rli'][iterate,-1])
        output.update_dict('Els',iterate,y[2*samm.Nli+5:3*samm.Nli+5])
        output.update_dict('Phizg',iterate,max((samm.Bz0*pi*output.outputs['rg'][iterate]**2),y[3*samm.Nli+5]))
        output.update_dict('Nd',iterate,y[3*samm.Nli+6])
        output.update_dict('Nt',iterate,y[3*samm.Nli+7])
        output.update_dict('Ns',iterate,y[3*samm.Nli+8:3*samm.Nli+8+samm.Ndops0.size])
        output.update_dict('Eg',iterate,y[3*samm.Nli+8+samm.Ndops0.size])
        output.update_dict('fmh',iterate,y[3*samm.Nli+8+samm.Ndops0.size+1])
        output.update_dict('Ndt',iterate,y[3*samm.Nli+8+samm.Ndops0.size+2])
        output.update_dict('Ndd3He',iterate,y[3*samm.Nli+8+samm.Ndops0.size+3])
        output.update_dict('Nddt',iterate,y[3*samm.Nli+8+samm.Ndops0.size+4])
        output.update_dict('Er',iterate,y[3*samm.Nli+8+samm.Ndops0.size+5])
        output.update_dict('E_ends',iterate,y[3*samm.Nli+8+samm.Ndops0.size+6])
        # output.update_dict('N_ends',iterate,y[3*samm.Nli+8+samm.Ndops0.size+7])
        
        yinits = output.update_ics(iterate)
        if y[3*samm.Nli+8+samm.Ndops0.size+1] < 0: set_trace()
            
        try:
            samm.species
            sum_ = 0
            for i in range(0,samm.species.shape[0]):
                sum_ = sum_ + samm.species[i][1]*u*output.outputs['Ns'][iterate,i]
            #endfor
            mg= sum_ + md*output.outputs['Nd'][iterate] + mt*output.outputs['Nt'][iterate]
        except AttributeError:
            mg = md*output.outputs['Nd'][iterate] + mt*output.outputs['Nt'][iterate]
        output.update_dict('mg', iterate, mg)
            
        Eion_ = 13.6*(output.outputs['Nd'][iterate] + output.outputs['Nt'][iterate] + output.outputs['Ns'][iterate].sum())*qe
        E_thg = output.outputs['Eg'][iterate] - output.outputs['Eiong'][iterate-1]
        output.update_dict('Ethg', iterate, E_thg)
        N_g = output.outputs['Nd'][iterate] + output.outputs['Nt'][iterate] + output.outputs['Ns'][iterate].sum()
        T_Ethg = 2./3*output.outputs['Ethg'][iterate]/(N_g*(1+output.outputs['Zgbar'][iterate-1]))/kb
        output.update_dict('Tg', iterate, T_Ethg)
        output.update_dict('TgkeV', iterate, T_Ethg*kb/qe*1e-3)
        samm.Tgbar_old = output.outputs['Tg'][iterate]
        
        T_Ethl = 2./3*output.outputs['Els'][iterate,1:].sum()/(samm.Nl*(1+output.outputs['Zlbar'][iterate-1]))/kb
        output.update_dict('Tl', iterate, T_Ethl)
        output.update_dict('TlkeV', iterate, T_Ethl*kb/qe*1e-3)
        T_Ethls = 2./3*output.outputs['Els'][iterate,1:]/(samm.Nl/samm.Nls*(1+output.outputs['Zlbars'][iterate-1,1:]))/kb
        output.update_dict('Tls', iterate, np.append(0.0, T_Ethls))
        
        Ng_ = output.outputs['mg'][iterate]/((md*samm.fdeut+mt*samm.ftrit)*samm.ffuel + (samm.mdops*samm.fdops).sum())
        Znucs_ = np.append(samm.Zdt, samm.Zdops)
        
        Z_gbars = np.array([20.0*sqrt(output.outputs['Tg'][iterate]*kb/qe*1e-3)*np.ones(Znucs_.size), Znucs_]).min(0)
        output.update_dict('Zgbars', iterate, Z_gbars)
        Z_gbar = np.dot(output.outputs['Zgbars'][iterate],np.append(np.array([output.outputs['Nd'][iterate]+output.outputs['Nt'][iterate]]), output.outputs['Ns'][iterate])/Ng_)
        output.update_dict('Zgbar', iterate, Z_gbar)
        Z_lbars = np.array([20.0*sqrt(output.outputs['Tls'][iterate,1:]*kb/qe*1e-3), samm.Zlnuc*np.ones(output.outputs['Tls'][iterate,1:].size)]).min(0)
        output.update_dict('Zlbars', iterate, np.append(0.0, Z_lbars))
        Z_lbar = min(20*sqrt(output.outputs['Tl'][iterate]*kb/qe*1e-3),samm.Zlnuc)
        output.update_dict('Zlbar', iterate, Z_lbar)
        
        Eiongs_ = 13.6*qe*samm.Znucs0**zeta/(samm.Znucs0-output.outputs['Zgbars'][iterate,:]+1.)**(1./zeta)*np.append(np.array([output.outputs['Nd'][iterate]+output.outputs['Nt'][iterate]]), output.outputs['Ns'][iterate])
        E_iong = min(output.outputs['Eg'][iterate]*0.5, Eiongs_.sum())
        output.update_dict('Eiong', iterate, E_iong)
        
        output.update_dict('Ydtn', iterate, output.outputs['Ndt'][iterate])
        output.update_dict('Yddn', iterate, output.outputs['Ndd3He'][iterate])
        output.update_dict('Yn', iterate, output.outputs['Ydtn'][iterate] + output.outputs['Yddn'][iterate])
        
        output.update_dict('Y', iterate, output.outputs['Ndt'][iterate]*samm.Q.loc['Qdt',1] + output.outputs['Ndd3He'][iterate]*samm.Q.loc['Qdd,3He',1] + output.outputs['Nddt'][iterate]*samm.Q.loc['Qdd,t',1])
        
        output.update_dict('Ndtdot', iterate, samm.Ndtdot)
        output.update_dict('Ndd3Hedot', iterate, samm.Ndd3Hedot)
        output.update_dict('Ndddot', iterate, samm.Ndddot)
        
        output.update_dict('Erdot', iterate, samm.Pr_new)
        output.update_dict('Ecdot', iterate, samm.Pc_new)
        output.update_dict('E_endsdot', iterate, samm.E_endloss)
        # output.update_dict('N_ends', iterate, samm.N_ends)
        
        output.update_dict('Tau_ei', iterate, np.mean(samm.Tau_ei))
        
        csEOS_sq = (3.0*samm.A1/(2.0*samm.rhol0_zero))*(((samm.gamma1*samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma1-1.0) - (samm.gamma2*samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma2-1.0))*(1 + (3.0/4.0)*(samm.A2 - 4.0)*((samm.rhols[1:]/samm.rhol0_zero)**(2.0/3.0) - 1)) + \
                ((samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma1) - (samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma2))*(1.0/2)*(samm.A2 - 4)*(samm.rhols[1:]/samm.rhol0_zero)**(-1.0/3))
        csEOS = sqrt(abs(csEOS_sq))
        csl = sqrt(csEOS**2 + (5.0/3.0)*(2./3.)*(output.outputs['Els'][iterate,1:]/samm.mls))
        tcsl = np.diff(output.outputs['rli'][iterate,1:])/csl
        
        csg = sqrt((5.0/3.0)*((2./3.)*output.outputs['Ethg'][iterate]/samm.Vg(output.outputs['rg'][iterate]))/(output.outputs['mg'][iterate]/samm.Vg(output.outputs['rg'][iterate])))
        tcsg = output.outputs['rg'][iterate]/csg
        if np.isnan(tcsg) == 1:
            tcsg = 0
        
        
    total_runtime = time.time() - start_time
    print("")
    print("Total runtime: {:6.2f} seconds.".format(total_runtime))  
        
    if samm.verbose == 1:    
        plt.figure()
        plt.clf()
        plt.plot(samm.t[0:iterate+1]*1.e9,output.outputs['rli'][:,-1]*1e3,'k')
        plt.plot(samm.t[0:iterate+1]*1.e9,output.outputs['rli'][:,1]*1e3,'r')
        plt.plot(samm.t[0:iterate+1]*1.e9,0.1*output.outputs['Il']*1e-6,'y')
        plt.plot(samm.t[0:iterate+1]*1e9,output.outputs['TgkeV'],'magenta')
        # plt.xlim((2900, 3140))
        # plt.ylim((0, 3))
        plt.legend(['Liner Outer Radius [mm]', 'Liner Inner/ Fuel Radius [mm]', 'Liner Current [1/10 MA]', 'Fuel Temp [keV]'])
        plt.show()
    
    if samm.save_outputs == 1:
        save_list = {}
        i = 0
        for key in output.outputs:
            save_list[key] = output.outputs[key]
            i += 1
        sv_str = samm.param_file.replace('.npz','_outs.npz')
        np.savez(os.path.expanduser(sv_str),  **save_list)
        # sv_str = samm.param_file.replace('.npz','_outs.npz')
        # np.savez(os.path.expanduser(sv_str),  output.outputs)
        
    return output
