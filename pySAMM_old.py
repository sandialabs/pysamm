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

import time

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
    
    # List of parameters that may be set by the user
    permitted_parameters = List(['rhog0','ftrit','fdops','Adops','Zdops','Nr', \
                                 'rl0','rhol0','AR','h','Nls', \
                                 'Bz0','Troom','tmin','tmax', \
                                 'Ezbl','rLEC','rdump','tzbl','tauzbl','lambda_b','xi_b','deltazlec','fsharp','bestfocus','dolew', \
                                 'Taur', \
                                 'Z0','L','L0','C','Rloss'])
                                 
    hot_spot_parameter = Dict({'rphf':[0,1]})
    
    switches = Dict({'nernst':[0,1],'laserabs':[0,1],'n_endloss':[0,1,2]})
    
    drive_mode = Dict({'VI_mode':['V','I']})
    
    VI_mode = Str(default_value='V')
    tmin = CFloat(default_value=2900.e-9)
    tmax = CFloat(default_value=3141.e-9)
    deltat = CFloat(default_value=0.2*1.e-9, help='time step (s)')
    
    def __init__(self, *args, **kwargs):
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
                            else: print('%s not loaded' % (key))
                    
        if len(args) > 1:
            raise SystemExit('Length of args -- string representation of liner material -- cannot be greater than zero.')
        for string in args: liner_material = string
        if len(args)==0 or liner_material=='Be':
            print('Be')
            ## Be Liner ##
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
            drive = np.loadtxt('/Users/thomoor/code/pysamm/pysamm/open_voltage.txt')
            drive_mult_factor=95./80.*(0.99*19.4/27.)
            drive[:,1] = drive[:,1]*drive_mult_factor
        elif self.VI_mode == 'I':
            drive = np.loadtxt('/Users/thomoor/code/pysamm/pysamm/current.txt')
            drive_mult_factor=1.23*18.15/16.44
            drive[:,1] = drive[:,1]*drive_mult_factor
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
    
    # Load open circuit waveform voltage and establish time array
    # path = CFloat(default_value='')
    # filename = CFloat(defaule_value='z2951_open_circuit_voltage.txt', help='incude extension')
    # drive = Property(depends_on=['path','filename'])
    # def _get_drive(self):
    #     return np.loadtxt(self.path+self.filename)
    # VI_mode = Str(default_value='V')
    # drive = np.loadtxt('/Users/thomoor/code/pysam/pysam/open_voltage.txt')
    # tmin = CFloat(default_value=2900.e-9)
    # tmax = CFloat(default_value=3141.e-9)
    # deltat = CFloat(default_value=0.2*1.e-9, help='time step (s)')
    
    # drive = Property(depends_on=['VI_mode'])
    # def _get_drive(self):
    #     if self.VI_mode == 'V':
    #         drive_ = np.loadtxt('/Users/thomoor/code/pysam/pysam/open_voltage.txt')
    #     elif self.VI_mode == 'I':
    #         drive_ = np.loadtxt('/Users/thomoor/code/pysam/pysam/current.txt')
    #     return drive_
    # tmin = CFloat(default_value=2900.e-9)
    # tmax = CFloat(default_value=3141.e-9)
    # deltat = CFloat(default_value=0.2*1.e-9, help='time step (s)')
    # t = Property(depends_on=['tmin','tmax','deltat','drive'])
    # def _get_t(self):
    #     number_points = round((self.tmax - self.tmin)/self.deltat + 1.)
    #     return np.linspace(self.tmin, self.tmax, int(number_points))
    # drive_amp = Property(depends_on=['tmin','tmax','deltat','drive'])
    # def _get_drive_amp(self):
    #     return interp1d(self.drive[:,0], self.drive[:,1], kind='linear', bounds_error=False, fill_value=0)(self.t)
    # drivedot = Property(depends_on=['VI_mode'])
    # def _get_drivedot(self):
    #     drivedot_ = []
    #     if self.VI_mode == 'I':
    #         amp_diff = np.diff(self.drive_amp)
    #         amp_diff = np.append(amp_diff,amp_diff[-1])
    #         time_diff = np.diff(self.t)
    #         time_diff = np.append(time_diff,time_diff[-1])
    #         drivedot_ = amp_diff/time_diff
    #     return drivedot_
    
    # Liner inputs
    AR = CFloat(default_value=6.0, label='AR', help='Aspect ratio AR = rl0/(rl0-rg0)')
    rl0 = CFloat(default_value=0.00279, help='Inintial outer liner radius)')
    rg0 = Property(depends_on=['rl0','AR'], label='rg0', help='Initial fuel-liner interface radius')
    def _get_rg0(self):
        return self.rl0*(self.AR - 1.0)/self.AR
    h = CFloat(default_value=7.5e-3, label='h', help='Height of liner')
    Nls = CFloat(default_value=20, label='Nls', help='Number of concentric shells (Nls>=20)')
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
    
    # From testing dopants, this was easier than calling in run_pySAMM()
    # fdops = CFloat(default_value=np.array([1/100.,.1/100.]), label='fdops', help='Fraction of dopants in fuel')
    # Adops = CFloat(default_value=np.array([9.01218,55.845]), label='Adops', help='Atomic mass number of dopants in fuel')
    # Zdops = CFloat(default_value=np.array([4.,26.]), label='Zdops', help='Atomic number of dopants in fuel')
    # mdops = Property(depends_on=['Adops'], label='mdops', help='Unit mass of dopants in fuel')
    # def _get_mdops(self):
    #     return self.Adops*u
        
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
        return self.rl0 + 4.0e-3

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
            
    # traits_view = View(Item(name='Zlnuc'),
    #                     Item(name='Alu'),
    #                     Item(name='rhol0'),
    #                     Item(name='rhoe'),
    #                     Item(name='A1'),
    #                     Item(name='A2'),
    #                     Item(name='gamma1'),
    #                     Item(name='gamma2'),
    #                     Item(name='Bz0'),
    #                     Item(name='Taur'),
    #                     Item(name='Phizl0',style='readonly'),
    #                     Item(name='dskin',style='readonly'),
    #                     Item(name='beta',style='readonly'),
    #                     Item(name='Tl0'),
    #                     Item(name='rl0'),
    #                     Item(name='rg0'),
    #                     Item(name='h'),
    #                     Item(name='Vl0',style='readonly'),
    #                     Item(name='ml',style='readonly'),
    #                     Item(name='Nls'),
    #                     Item(name='Nli',style='readonly'),
    #                     Item(name='mls',style='readonly'),
    #                     buttons = [OKButton, CancelButton],
    #                     resizable=True)               
                        
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
    Phizg0 = Property(depends_on=['self.rg0','seelf.Bz0'])
    def _get_Phizg0(self):
        return (pi*self.rg0**2)*self.Bz0
        
    # traits_view = View(Item(name='rg0', label='rg0', style='readonly'),
    #                     Item(name='h', label='h', style='readonly'),
    #                     Item(name='Vg0', label='Vg0',style='readonly'),
    #                     Item(name='mg0', label='mg0',style='readonly'),
    #                     Item(name='Bz0', label='Bz0',style='readonly'),
    #                     Item(name='Phizg0', label='Phizg0',style='readonly'),
    #                     buttons = [OKButton, CancelButton],
    #                     resizable=True)
        
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
        
    # def __inti__(self):
    #     set_trace()
    #     if hasattr(self, 'rphf') == False:
    #         self.rphf = self.rb/self.rg0
    
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
        
    Phizv = Property(depends_on=['self.rc','self.rl0','self.Bz0'], label='Phizv', help='Axial magnetic flux in vacuum')
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
        Average magnetic furl pressure in the vacuum
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
        if np.isnan(zbar) == True:
            set_trace()
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
            
        Pph = 0
        if self.laser_abs_trigger == 1:
            Pzbl_ = interp1d(self.t,self.Pzbl, bounds_error=False, fill_value=0)(t_)
            if Pzbl_ > 0:
                if self.laserabs == 1: Pph = self.Preheat(Pzbl_, rli[1], Nd, Nt, Ns, t_)
                elif self.laserabs == 0: Pph = Pzbl_
        
        self.Pc_new, self.Pch, self.Pcs = self.Pc(Phizg,rli[1],Ns,Nd,Nt,Eg)           
        if fmh == 1:
            # set_trace()
            dydt_3[1:self.Nls+1] = dydt_3[1:self.Nls+1] + self.Pc_new/self.Nls
            Egdot = self.PpdV(rli[1],rlidot[1],Eg,Nd,Nt,Ns) + Pph - self.Pr_new - self.Pc_new
        else:
            dydt_3[1:self.Nls+1] = dydt_3[1:self.Nls+1] + self.Pc_new/self.Nls
            Egdot = self.PpdV(rli[1],rlidot[1],Eg,Nd,Nt,Ns) + Pph - self.Pr_new

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
        
        if Eg >= self.Eion and self.Th > self.TB:
            Phizgdot = -2.0*pi*rli[1]*np.append(0,self.Fxe(Phizg,rli[1],Ns,Nd,Nt)*(kb/qe))[-1]*np.append(0,self.dTgdr_rg(rli[1]))[-1]
            Egdot = Egdot + self.nernst*Ndtdot*self.Qalpha*self.falpha(rli[1],Eg,Nd,Nt,Ns,Phizg)
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
                    Ndddot]
                    
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
                    Ndddot]
        
        dydt_ = []
        
        for i in range(0,len(dydt)):
            dydt_ = np.append(dydt_,dydt[i])
                    
        return dydt_
        
class RunSAMM(HasTraits):
    
    outputs = Dict(Str, Array(shape=(None)))
    
    # parm_file = File('/home/thomoor/data/run_parameters/pysamm/default.npz', label='parameter file', help='file to store and read the parameters in/from')
    # 
    # def load(self):
    #     '''
    #     Loads the meta information for the object
    #     '''
    #     data = np.load(os.path.expanduser(self.parm_file))
    #     for attribute, value in data.items():
    #         setattr(self, attribute, value.tolist())
    
    def build_model(self, *args, **kwargs):
        samm_ = OdeSystem(*args, **kwargs)
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
        return yinit
        
    def update_dict(self,dict_key,iter_,val):
        self.outputs[dict_key][iter_] = val
        return
        
def save_data(dictionary, path, filename):
    np.savez(path+filename, dictionary)
    return
        
def load_data(path, filename):
    data = np.load(path+filename+'.npz')
    keys = data.files[0]
    data = data[keys]
    return data.item()
    
def plot_2d():
    from matplotlib.pyplot import contour
    from matplotlib.pyplot import contourf
    from matplotlib.mlab import griddata
    return
 
def print_stuff():
    path = '/Users/thomoor/Desktop/runs/'
    filename = 'z2951_20T_10kJ_2mgcc'
    data=load_data(path, filename)
    print(data['Yddn'].max(),data['rg'][0]/data['rg'].min())

# def parameter_scan():
#     """
#     Make this more general with keyword arguments, dicts, etc
#     """
#     res = 20
#     Ezbl_scan = np.linspace(0.0, 30.0, res)*1.0e3
#     rhog0_scan = np.linspace(0.7, 10.0, res)
#     Bz0_ = 10.0
#     n_endloss_ = 1
#     ddn = np.zeros([res,res])
#     crmax = np.zeros([res,res])
#     path = '/Users/thomoor/Desktop/runs/parameter_scan/'
#     status = open(path+"param_scans_completed.txt", "w")
#     status.close()
#     i_ = 0
#     j_ = 0
#     for i in range(i_,res):
#         for j in range(j_,res):
#         # for j in range(j_,res):
#             filename = 'run_Bz'+str(int(Bz0_))+'_ezbl'+str(int(Ezbl_scan[i]))+'_rhog'+str(rhog0_scan[j])
#             print('Running '+filename),
#             print('(i=%s, j=%s)' % (i,j))
#             try:
#                 run_ = run_pySAMM(Bz0=Bz0_ , Ezbl=Ezbl_scan[i], rhog0=rhog0_scan[j],L0=9.0e-9,n_endloss=n_endloss_)
#                 outs = run_.outputs
#                 ddn[i,j] = run_.outputs['Yddn'].max()
#                 crmax[i,j] = run_.outputs['rg'][0]/run_.outputs['rg'].min()
#             except ValueError as err:
#                 outs = {'ValueError':err[0]}
#                 ddn[i,j] = -999
#                 crmax[i,j] = -999
#             except UnboundLocalError as err:
#                 outs = {'UnboundLocalError':err[0]}
#                 ddn[i,j] = -999
#                 crmax[i,j] = -999
#             save_data(outs, path, filename)
#             results = {'Yddn':ddn,'CR_max':crmax}
#             np.savez(path+'preheat_density_scan', results)
#             status = open(path+"param_scans_completed.txt", "a")
#             status.write('i=%s, j=%s\n' % (i,j))
#             status.close()
#             print('Approximately %3.3s ' % ((1.0*i**2)/(res**2)*100.0) +'% complete')
#             
#     return results

def parameter_scan(path,istart):
    """
    Make this more general with keyword arguments, dicts, etc
    """
    # res = 20
    # Ezbl_scan = np.linspace(0.0, 30.0, res)*1.0e3
    # rhog0_scan = np.linspace(0.7, 10.0, res)
    rg0_ = 0.002325
    h_scan = np.array([5, 7, 10, 15])*1.0e-3
    rhog0_scan = np.array([0.5, 0.7, 1.0, 1.5, 2.0])
    Ezbl_scan  = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])*1.0e3
    Bz0_scan  = np.array([5.0, 10.0, 20.0, 30.0, 40.0])
    rphf_scan  = np.array([0.1, 0.21, 0.25, 0.3])
    n_endloss_ = 1
    laserabs_ = 0
    resi = h_scan.size
    resj = rhog0_scan.size
    resk = Ezbl_scan.size
    resl = Bz0_scan.size
    resm = rphf_scan.size
    # ddn = np.zeros([resi,resj,resk,resl,resm])
    # crmax = np.zeros([resi,resj,resk,resl,resm])
    # path = '/gscratch/thomoor/pysamm_scans/'
    status = open(path+"param_scans_completed.txt", "w")
    status.close()
    i_ = istart
    for i in range(i_,i_):
        for j in range(0,resj):
            for k in range(0,resk):
                for l in range(0,resl):
                    for m in range(0,resm):
                        filename = 'run_h'+str(int(h_scan[i]))+'_rhog'+str(rhog0_scan[j])+'_Ezbl'+str(int(Ezbl_scan[k]))+'_Bz'+str(int(Bz0_scan[l]))+'_rphf'+str(rphf_scan[m])
                        print('Running '+filename),
                        print('(i=%s, j=%s)' % (i,j))
                        try:
                            run_ = run_pySAMM(h=h_scan[i], rhog0=rhog0_scan[j], Ezbl=Ezbl_scan[k], Bz0=Bz0_scan[l], rphf=rphf_scan[m], rLEC=rg0_*rphf_scan[m], n_endloss=n_endloss_, laserabs=laserabs_, VI_mode='V')
                            outs = run_.outputs
                            # ddn[i,j] = run_.outputs['Yddn'].max()
                            # crmax[i,j] = run_.outputs['rg'][0]/run_.outputs['rg'].min()
                        except ValueError as err:
                            outs = {'ValueError':err[0]}
                            # ddn[i,j] = -999
                            # crmax[i,j] = -999
                        except UnboundLocalError as err:
                            outs = {'UnboundLocalError':err[0]}
                            # ddn[i,j] = -999
                            # crmax[i,j] = -999
                        save_data(outs, path, filename)
                        # results = {'Yddn':ddn,'CR_max':crmax}
                        # np.savez(path+'preheat_density_scan', results)
                        status = open(path+"param_scans_completed.txt", "a")
                        status.write('i=%s, j=%s, k=%s, l=%s, m=%s\n' % (i,j,k,l,m))
                        status.close()
                        # print('Approximately %3.3s ' % ((1.0*i**2)/(res**2)*100.0) +'% complete')
            
    return
    
def load_parameter_scan(path):
    # rg0_ = 0.002325
    h_scan = np.array([5, 7, 10, 15])*1.0e-3
    rhog0_scan = np.array([0.5, 0.7, 1.0, 1.5, 2.0])
    Ezbl_scan  = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])*1.0e3
    Bz0_scan  = np.array([5.0, 10.0, 20.0, 30.0, 40.0])
    rphf_scan  = np.array([0.1, 0.21, 0.25, 0.3])
    resi = h_scan.size
    resj = rhog0_scan.size
    resk = Ezbl_scan.size
    resl = Bz0_scan.size
    resm = rphf_scan.size
    ddn = np.zeros([resi,resj,resk,resl,resm])
    crmax = np.zeros([resi,resj,resk,resl,resm])
    num_proc = 0
    null_list = np.array([])
    for i in range(0,resi):
        for j in range(0,resj):
            for k in range(0,resk):
                for l in range(0,resl):
                    for m in range(0,resm):
                        filename = 'run_h'+str((h_scan[i]))+'_rhog'+str(rhog0_scan[j])+'_Ezbl'+str(int(Ezbl_scan[k]))+'_Bz'+str(int(Bz0_scan[l]))+'_rphf'+str(rphf_scan[m])
                        try:
                            data = load_data(path, filename)
                            try:
                                data['Yddn']
                                # set_trace()
                                ddn[i,j,k,l,m] = data['Yddn'].max()
                                # set_trace()
                                index = np.where(data['TgkeV'] == data['TgkeV'].max())
                                # index = np.where(data['Yddn'] == data['Yddn'].max())
                                if len(index[0]) > 1: index = index[0][0]
                                # crmax[i,j] = data['rg'][0]/data['rg'].min()
                                crmax[i,j,k,l,m] = data['rg'][0]/data['rg'][index]
                            except KeyError:
                                ddn[i,j,k,l,m] = -999
                                crmax[i,j,k,l,m] = -999
                        except IOError:
                            np.append(null_list,filename)
                            ddn[i,j,k,l,m] = -999
                            crmax[i,j,k,l,m] = -999
                        num_proc += 1
                        print('%s processed' % (1.0*num_proc/(resi*resj*resk*resl*resm)))
    
    return ddn, crmax, null_list
    
# def make_image(x,y):
#     
#     for i in range(x.size):
#         for j in range(y.size);
#         image(i,j) = 
#     
#     return image
    
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
    path = '/Users/thomoor/Desktop/runs/parameter_scan/'
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
                index = np.where(run_.outputs['Yddn'] == ddn[i,j])[0][0]
                # crmax[i,j] = run_.outputs['rg'][0]/run_.outputs['rg'].min()
                crmax[i,j] = run_.outputs['rg'][0]/run_.outputs['rg'][index]
            except ValueError as err:
                outs = {'ValueError':err[0]}
                ddn[i,j] = -999
                crmax[i,j] = -999
            except UnboundLocalError as err:
                outs = {'UnboundLocalError':err[0]}
                ddn[i,j] = -999
                crmax[i,j] = -999
            save_data(outs, path, filename)
            results = {'Yddn':ddn,'CR_max':crmax}
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
    path = '/Users/thomoor/Desktop/runs/parameter_scan/'
    for i in range(0,resi):
        for j in range(0,resj):
            filename = 'run_Bz'+str(int(Bz_scan[j]))+'_Ezbl'+str(int(Ezbl_scan[i]))
            print('Running '+filename),
            print('(i=%s, j=%s)' % (i,j))
            data = load_data(path, filename)
            try:
                data['Yddn']
                ddn[i,j] = data['Yddn'].max()
                # set_trace()
                index = np.where(data['TgkeV'] == data['TgkeV'].max())
                # index = np.where(data['Yddn'] == data['Yddn'].max())
                if len(index[0]) > 1: index = index[0][0]
                # crmax[i,j] = data['rg'][0]/data['rg'].min()
                crmax[i,j] = data['rg'][0]/data['rg'][index]
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
    plt.plot(Ezbl_scan,crmax[:,4],'-',color='magenta')
    plt.plot(Ezbl_scan,crmax[:,3],'-',color='orange')
    plt.plot(Ezbl_scan,crmax[:,2],'-',color='green')
    plt.plot(Ezbl_scan,crmax[:,1],'-',color='blue')
    plt.plot(Ezbl_scan,crmax[:,0],'-',color='grey')
    plt.legend(['B$_{z0}$ = 30 T','B$_{z0}$ = 10 T','B$_{z0}$ = 3 T','B$_{z0}$ = 1 T','B$_{z0}$ = 0 T'])
    plt.xscale('log')
    plt.xlim([30,1e4])
    plt.ylim([0,200])
    plt.ylabel('Convergence Ratio @ Peak Burn')
    plt.tight_layout()
    plt.show()
    return ddn, crmax, Ezbl_scan, Bz_scan

# def load_parameter_scan():
#     res = 20
#     Ezbl_scan = np.linspace(0.0, 30.0, res)*1.0e3
#     rhog0_scan = np.linspace(0.7, 10.0, res)
#     Bz0_ = 10.0
#     ddn = np.zeros([res,res])
#     crmax = np.zeros([res,res])
#     path = '/Users/thomoor/Desktop/runs/parameter_scan/'
#     # status = open(path+"param_scans_completed.txt", "w")
#     # status.close()
#     for i in range(res):
#         for j in range(res):
#             filename='run_Bz%s_ezbl%s_rhog%s' %(int(Bz0_),int(Ezbl_scan[i]),rhog0_scan[j])
#             print('Running '+filename),
#             print('(i=%s, j=%s)' % (i,j))
#             data = load_data(path, filename)
#             try:
#                 data['Yddn']
#                 ddn[i,j] = data['Yddn'].max()
#                 crmax[i,j] = data['rg'][0]/data['rg'].min()
#             except KeyError:
#                 ddn[i,j] = -999
#                 crmax[i,j] = -999
#             print('Approximately %3.3s ' % ((1.0*i**2)/(res**2)*100.0) +'% complete')
#     plot_parameter_scan(rhog0_scan,Ezbl_scan,ddn,crmax)
#     return ddn, crmax

def plot_parameter_scan(x,y,z1,z2):
    from matplotlib.pyplot import contour
    from matplotlib.pyplot import contourf
    from matplotlib.mlab import griddata
    plt.figure()
    cb_res = np.linspace(0,5e14,40)
    cb_ticks = np.linspace(0,5e14,11)
    contourf(x,y,z1,cb_res,cmap='jet')
    plt.colorbar(ticks=cb_ticks)
    # plt.clim(0,5e14)
    contour_levels = np.log10(np.arange(20,40,5))
    contour_levels = np.append(contour_levels,np.log10(np.array([40,50,70,100])))
    cs=contour(x,y,np.log10(z2),levels=contour_levels,colors='k')
    cs.levels = 10**cs.levels
    plt.clabel(cs,levels=10**cs.levels,fmt='%3.0f')
    plt.xlim([0,x.max()])
    plt.ylim([0,y.max()])
    return
    
def find_files(filename, path):
    import fnmatch
    import os
    
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, filename):
            print(file)
    return

def single_run():
    Bz_scan = np.array([10.0, 15.0, 10.0, 15.0])
    Ezbl_scan = np.array([800.0, 800.0, 1500.0, 1500.0])
    rhog_scan = np.array([0.7, 0.7, 1.05, 1.05])
    ddn = np.zeros(Bz_scan.size)
    crmax = np.zeros(Bz_scan.size)
    pg_stag = np.zeros(Bz_scan.size)
    h = 7.5e-3
    for i in range(Bz_scan.size):
        filename = 'pySAMM_outs_Ezbl'+str(int(Ezbl_scan[i]))+'_rhog'+str(rhog_scan[i])+'_Bz'+str(int(Bz_scan[i]))+'_laserabs_off'
        print('Currently running %s' % (filename))
        run_ = run_pySAMM(Bz0=Bz_scan[i] , Ezbl=Ezbl_scan[i], rhog0=rhog_scan[i],L0=9.05*1.0e-9,laserabs=0)
        outs = run_.outputs
        ddn[i] = outs['Yddn'].max()
        crmax[i] = outs['rg'][0]/outs['rg'].min()
        index_crmax = np.where(outs['rg'] == outs['rg'].min())
        pg_stag[i] = (2.0/3.0)*outs['Ethg'][index_crmax]/(pi*outs['rg'][index_crmax]**2*h)
        path = '/Users/thomoor/Desktop/runs/'
        save_data(outs, path, filename)
    return {'Yddn':ddn,'CRmax':crmax,'pg_stag':pg_stag}
    
def load_single_run():
    results = {}
    Bz_scan = np.array([10.0, 15.0, 10.0, 15.0])
    Ezbl_scan = np.array([800.0, 800.0, 1500.0, 1500.0])
    rhog_scan = np.array([0.7, 0.7, 1.05, 1.05])
    for i in range(Bz_scan.size):
        filename = 'pySAMM_outs_Ezbl'+str(int(Ezbl_scan[i]))+'_rhog'+str(rhog_scan[i])+'_Bz'+str(int(Bz_scan[i]))
        print('Currently loading %s' % (filename))
        path = '/Users/thomoor/Desktop/runs/'
        data = load_data(path, filename)
        results[filename] = {'Yddn':data['Yddn'].max(),'CR_max':data['rg'][0]/data['rg'].min()}
        del data
    return results
                                            
def run_pySAMM(*args, **kwargs):
    """
    A function to run pySAMM
    - *args and **kwargs defined in InputDeck Class
    """
    start_time = time.time()
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    output = RunSAMM()
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
    
        samm.iterate_ = iterate
            
        # if iterate%100 == 0:
        #     # print '.',
        #     print(samm.t[iterate])
    
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
        
        csEOS_sq = (3.0*samm.A1/(2.0*samm.rhol0_zero))*(((samm.gamma1*samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma1-1.0) - (samm.gamma2*samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma2-1.0))*(1 + (3.0/4.0)*(samm.A2 - 4.0)*((samm.rhols[1:]/samm.rhol0_zero)**(2.0/3.0) - 1)) + \
                ((samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma1) - (samm.rhols[1:]/samm.rhol0_zero)**(samm.gamma2))*(1.0/2)*(samm.A2 - 4)*(samm.rhols[1:]/samm.rhol0_zero)**(-1.0/3))
        csEOS = sqrt(abs(csEOS_sq))
        csl = sqrt(csEOS**2 + (5.0/3.0)*(2./3.)*(output.outputs['Els'][iterate,1:]/samm.mls))
        tcsl = np.diff(output.outputs['rli'][iterate,1:])/csl
        
        csg = sqrt((5.0/3.0)*((2./3.)*output.outputs['Ethg'][iterate]/samm.Vg(output.outputs['rg'][iterate]))/(output.outputs['mg'][iterate]/samm.Vg(output.outputs['rg'][iterate])))
        tcsg = output.outputs['rg'][iterate]/csg
        if np.isnan(tcsg) == 1:
            tcsg = 0
    
    showplot = True
    if showplot == True:        
        plt.figure()
        plt.clf()
        plt.plot(samm.t[0:iterate+1]*1.e9,output.outputs['rli'][:,-1]*1e3,'k')
        plt.plot(samm.t[0:iterate+1]*1.e9,output.outputs['rli'][:,1]*1e3,'r')
        plt.plot(samm.t[0:iterate+1]*1.e9,0.1*output.outputs['Il']*1e-6,'y')
        plt.plot(samm.t[0:iterate+1]*1e9,output.outputs['TgkeV'],'magenta')
        plt.xlim((2900, 3140))
        plt.ylim((0, 3))
        plt.legend(['Liner Outer Radius [mm]', 'Liner Inner/ Fuel Radius [mm]', 'Liner Current [1/10 MA]', 'Fuel Temp [keV]'])
        plt.show()
        
    total_runtime = time.time() - start_time
    print("Total runtime: {:6.2f} seconds.".format(total_runtime))
    return output
    
def dimensionless_curve(nernst):
    """
    # For updating rhoc when iteratively solving for rh and TB in Pr() in OdeSystem Class
    # Dimensionless monotonically decreasing function Fxgvp of dimensional parameter xgpv = rh/r*
    # where rh is the radius of the hot spot and r* is the dimensionless radius -- See Mcbride's 2015
    # paper for details
    """
    ### Still haven't quite worked this out ###
    # xgv = nonlinspace(0.,1.,1e3,0.05)
    # inte = integrate.cumtrapz(2.*xgv/(1-xgv**self.xi)**1.0,xgv,initial=1)
    # Fxgv = (1-xgv**self.xi)**1.0/xgv**2.*inte
    # Fxgv[0] = 1
    # Fxgv[-1] = 0
    # xgvp = np.flipud(xgv)
    # Fxgvp = np.flipud(Fxgv)
    # xgvp, indices = np.unique(xgvp,return_index=True)
    # xgvp = np.flipud(xgv)
    # self.xgvp = xgvp[indices]
    # self.Fxgvp = Fxgvp[indices]
    
    ### Loading actual arrays instead ###
    if nernst == 1:
        xgvp=np.array([1.00000000000000000,0.99999999999999967,0.99999999999999956,0.99999999999999944,0.99999999999999933,0.99999999999999922,0.99999999999999911,0.99999999999999900,0.99999999999999889,0.99999999999999878, \
                            0.99999999999999867,0.99999999999999856,0.99999999999999845,0.99999999999999833,0.99999999999999822,0.99999999999999811,0.99999999999999800,0.99999999999999789,0.99999999999999778,0.99999999999999767, \
                            0.99999999999999756,0.99999999999999734,0.99999999999999711,0.99999999999999689,0.99999999999999667,0.99999999999999645,0.99999999999999623,0.99999999999999600,0.99999999999999578,0.99999999999999556, \
                            0.99999999999999534,0.99999999999999500,0.99999999999999467,0.99999999999999434,0.99999999999999400,0.99999999999999367,0.99999999999999334,0.99999999999999289,0.99999999999999245,0.99999999999999201, \
                            0.99999999999999156,0.99999999999999112,0.99999999999999056,0.99999999999999001,0.99999999999998945,0.99999999999998890,0.99999999999998823,0.99999999999998757,0.99999999999998690,0.99999999999998612, \
                            0.99999999999998535,0.99999999999998457,0.99999999999998368,0.99999999999998279,0.99999999999998179,0.99999999999998079,0.99999999999997968,0.99999999999997857,0.99999999999997746,0.99999999999997624, \
                            0.99999999999997491,0.99999999999997358,0.99999999999997213,0.99999999999997058,0.99999999999996902,0.99999999999996736,0.99999999999996558,0.99999999999996370,0.99999999999996170,0.99999999999995970, \
                            0.99999999999995759,0.99999999999995537,0.99999999999995293,0.99999999999995037,0.99999999999994771,0.99999999999994493,0.99999999999994205,0.99999999999993894,0.99999999999993572,0.99999999999993228, \
                            0.99999999999992872,0.99999999999992495,0.99999999999992095,0.99999999999991673,0.99999999999991229,0.99999999999990763,0.99999999999990274,0.99999999999989764,0.99999999999989220,0.99999999999988654, \
                            0.99999999999988054,0.99999999999987421,0.99999999999986755,0.99999999999986056,0.99999999999985323,0.99999999999984546,0.99999999999983724,0.99999999999982858,0.99999999999981948,0.99999999999980993, \
                            0.99999999999979994,0.99999999999978939,0.99999999999977829,0.99999999999976663,0.99999999999975431,0.99999999999974132,0.99999999999972766,0.99999999999971334,0.99999999999969824,0.99999999999968237, \
                            0.99999999999966560,0.99999999999964795,0.99999999999962941,0.99999999999960987,0.99999999999958933,0.99999999999956768,0.99999999999954492,0.99999999999952094,0.99999999999949563,0.99999999999946898, \
                            0.99999999999944100,0.99999999999941158,0.99999999999938061,0.99999999999934797,0.99999999999931355,0.99999999999927736,0.99999999999923928,0.99999999999919920,0.99999999999915701,0.99999999999911260, \
                            0.99999999999906586,0.99999999999901668,0.99999999999896494,0.99999999999891043,0.99999999999885303,0.99999999999879263,0.99999999999872902,0.99999999999866207,0.99999999999859157,0.99999999999851741, \
                            0.99999999999843936,0.99999999999835720,0.99999999999827072,0.99999999999817968,0.99999999999808387,0.99999999999798295,0.99999999999787670,0.99999999999776490,0.99999999999764722,0.99999999999752331, \
                            0.99999999999739297,0.99999999999725575,0.99999999999711131,0.99999999999695921,0.99999999999679912,0.99999999999663058,0.99999999999645317,0.99999999999626643,0.99999999999606992,0.99999999999586309, \
                            0.99999999999564526,0.99999999999541600,0.99999999999517475,0.99999999999492073,0.99999999999465339,0.99999999999437195,0.99999999999407574,0.99999999999376388,0.99999999999343558,0.99999999999309008, \
                            0.99999999999272637,0.99999999999234346,0.99999999999194045,0.99999999999151623,0.99999999999106970,0.99999999999059963,0.99999999999010480,0.99999999998958400,0.99999999998903577,0.99999999998845868, \
                            0.99999999998785116,0.99999999998721167,0.99999999998653855,0.99999999998583000,0.99999999998508415,0.99999999998429911,0.99999999998347266,0.99999999998260281,0.99999999998168709,0.99999999998072320, \
                            0.99999999997970856,0.99999999997864053,0.99999999997751632,0.99999999997633293,0.99999999997508726,0.99999999997377598,0.99999999997239575,0.99999999997094280,0.99999999996941347,0.99999999996780364, \
                            0.99999999996610911,0.99999999996432531,0.99999999996244771,0.99999999996047118,0.99999999995839062,0.99999999995620059,0.99999999995389532,0.99999999995146871,0.99999999994891442,0.99999999994622568, \
                            0.99999999994339539,0.99999999994041622,0.99999999993728017,0.99999999993397903,0.99999999993050426,0.99999999992684652,0.99999999992299626,0.99999999991894339,0.99999999991467725,0.99999999991018651, \
                            0.99999999990545940,0.99999999990048349,0.99999999989524568,0.99999999988973232,0.99999999988392874,0.99999999987781962,0.99999999987138899,0.99999999986461996,0.99999999985749466,0.99999999984999433, \
                            0.99999999984209931,0.99999999983378873,0.99999999982504073,0.99999999981583232,0.99999999980613929,0.99999999979593601,0.99999999978519571,0.99999999977389020,0.99999999976198961,0.99999999974946274, \
                            0.99999999973627651,0.99999999972239628,0.99999999970778553,0.99999999969240583,0.99999999967621667,0.99999999965917541,0.99999999964123720,0.99999999962235486,0.99999999960247876,0.99999999958155650, \
                            0.99999999955953311,0.99999999953635055,0.99999999951194785,0.99999999948626084,0.99999999945922191,0.99999999943075990,0.99999999940079987,0.99999999936926298,0.99999999933606620,0.99999999930112227, \
                            0.99999999926433913,0.99999999922562011,0.99999999918486326,0.99999999914196125,0.99999999909680126,0.99999999904926440,0.99999999899922565,0.99999999894655323,0.99999999889110858,0.99999999883274582, \
                            0.99999999877131129,0.99999999870664347,0.99999999863857203,0.99999999856691790,0.99999999849149246,0.99999999841209730,0.99999999832852349,0.99999999824055097,0.99999999814794838,0.99999999805047191, \
                            0.99999999794786509,0.99999999783985793,0.99999999772616621,0.99999999760649072,0.99999999748051649,0.99999999734791201,0.99999999720832844,0.99999999706139830,0.99999999690673502,0.99999999674393159, \
                            0.99999999657255956,0.99999999639216786,0.99999999620228197,0.99999999600240208,0.99999999579200216,0.99999999557052854,0.99999999533739847,0.99999999509199833,0.99999999483368240,0.99999999456177091, \
                            0.99999999427554831,0.99999999397426131,0.99999999365711711,0.99999999332328116,0.99999999297187492,0.99999999260197359,0.99999999221260372,0.99999999180274068,0.99999999137130591,0.99999999091716407, \
                            0.99999999043912002,0.99999998993591577,0.99999998940622703,0.99999998884866004,0.99999998826174741,0.99999998764394460,0.99999998699362580,0.99999998630907971,0.99999998558850489,0.99999998483000507, \
                            0.99999998403158430,0.99999998319114136,0.99999998230646459,0.99999998137522583,0.99999998039497451,0.99999997936313101,0.99999997827697995,0.99999997713366306,0.99999997593017165,0.99999997466333856, \
                            0.99999997332983004,0.99999997192613688,0.99999997044856503,0.99999996889322629,0.99999996725602758,0.99999996553266057,0.99999996371859001,0.99999996180904205,0.99999995979899159,0.99999995768314898, \
                            0.99999995545594622,0.99999995311152234,0.99999995064370772,0.99999994804600811,0.99999994531158742,0.99999994243324986,0.99999993940342091,0.99999993621412719,0.99999993285697597,0.99999992932313253, \
                            0.99999992560329731,0.99999992168768137,0.99999991756598039,0.99999991322734771,0.99999990866036592,0.99999990385301674,0.99999989879264917,0.99999989346594642,0.99999988785889093,0.99999988195672729, \
                            0.99999987574392346,0.99999986920412987,0.99999986232013671,0.99999985507382805,0.99999984744613479,0.99999983941698389,0.99999983096524614,0.99999982206868010,0.99999981270387373,0.99999980284618284, \
                            0.99999979246966608,0.99999978154701685,0.99999977004949137,0.99999975794683293,0.99999974520719248,0.99999973179704471,0.99999971768109963,0.99999970282221007,0.99999968718127374,0.99999967071713025, \
                            0.99999965338645291,0.99999963514363466,0.99999961594066800,0.99999959572701891,0.99999957444949350,0.99999955205209834,0.99999952847589291,0.99999950365883461,0.99999947753561536,0.99999945003748980, \
                            0.99999942109209450,0.99999939062325727,0.99999935855079702,0.99999932479031262,0.99999928925296067,0.99999925184522176,0.99999921246865442,0.99999917101963620,0.99999912738909069,0.99999908146220073, \
                            0.99999903311810601,0.99999898222958528,0.99999892866272133,0.99999887227654871,0.99999881292268278,0.99999875044492925,0.99999868467887287,0.99999861545144508,0.99999854258046850,0.99999846587417729, \
                            0.99999838513071293,0.99999830013759250,0.99999821067114991,0.99999811649594728,0.99999801736415495,0.99999791301489993,0.99999780317357878,0.99999768755113549,0.99999756584330046,0.99999743772978988, \
                            0.99999730287346300,0.99999716091943469,0.99999701149414177,0.99999685420435969,0.99999668863616809,0.99999651435386105,0.99999633089880102,0.99999613778821150,0.99999593451390678,0.99999572054095442, \
                            0.99999549530626775,0.99999525821712387,0.99999500864960400,0.99999474594695159,0.99999446941784376,0.99999417833457238,0.99999387193112876,0.99999354940118812,0.99999320989598750,0.99999285252209202, \
                            0.99999247633904420,0.99999208035688858,0.99999166353356683,0.99999122477217561,0.99999076291807953,0.99999027675587315,0.99998976500618220,0.99998922632229703,0.99998865928662839,0.99998806240697724, \
                            0.99998743411260760,0.99998677275011327,0.99998607657906657,0.99998534376743842,0.99998457238677729,0.99998376040713399,0.99998290569171999,0.99998200599128417,0.99998105893819378,0.99998006204020395, \
                            0.99997901267389888,0.99997790807778830,0.99997674534504022,0.99997552141583179,0.99997423306929656,0.99997287691504899,0.99997144938426208,0.99996994672027584,0.99996836496871133,0.99996669996706455, \
                            0.99996494733375207,0.99996310245658104,0.99996116048061157,0.99995911629538059,0.99995696452145322,0.99995469949626647,0.99995231525922779,0.99994980553602919,0.99994716372213599,0.99994438286540621, \
                            0.99994145564779602,0.99993837436610100,0.99993513091168518,0.99993171674914227,0.99992812289383393,0.99992433988824625,0.99992035777710131,0.99991616608115919,0.99991175376964125,0.99990710923120130, \
                            0.99990222024336972,0.99989707394038918,0.99989165677935699,0.99988595450458628,0.99987995211009073,0.99987363380009542,0.99986698294746879,0.99985998204996707,0.99985261268417580,0.99984485545702717, \
                            0.99983668995476538,0.99982809468922662,0.99981904704129110,0.99980952320135896,0.99979949810669355,0.99978894537546681,0.99977783723733349,0.99976614446035095,0.99975383627405356,0.99974088028847741, \
                            0.99972724240892352,0.99971288674623526,0.99969777552235284,0.99968186897089772,0.99966512523252393,0.99964750024476201,0.99962894762606525,0.99960941855375285,0.99958886163552929,0.99956722277424137, \
                            0.99954444502551720,0.99952046844791276,0.99949522994517126,0.99946866310018023,0.99944069800018964,0.99941126105283118,0.99938027479245384,0.99934765767626721,0.99931332386975491,0.99927718302079460, \
                            0.99923914002188896,0.99919909475988311,0.99915694185250847,0.99911257037106149,0.99906586354848570,0.99901669847209018,0.99896494576009487,0.99891046922115245,0.99885312549594996,0.99879276367994729, \
                            0.99872922492626026,0.99866234202764237,0.99859193897646559,0.99851783050154264,0.99843982158057110,0.99835770692691694,0.99827127044938624,0.99818028468356446,0.99808451019322575,0.99798369494023764, \
                            0.99787757362130269,0.99776586696979230,0.99764828102083392,0.99752450633771983,0.99739421719759980,0.99725707073431558,0.99711270603612157,0.99696074319591743,0.99680078231149194,0.99663240243314932, \
                            0.99645516045594662,0.99626858995362799,0.99607219995118734,0.99586547363282873,0.99564786698192498,0.99541880734939470,0.99517769194673122,0.99492388625971706,0.99465672237864944,0.99437549724068353, \
                            0.99407947077966685,0.99376786397859662,0.99343985681957536,0.99309458612586876,0.99273114329038814,0.99234857188461911,0.99194586514170424,0.99152196330705700,0.99107575084953359,0.99060605352582476, \
                            0.99011163529034185,0.98959119504246507,0.98904336320259478,0.98846669810799448,0.98785968221894149,0.98722071812520151,0.98654812434231731,0.98584013088664979,0.98509487461752610,0.98431039433423795, \
                            0.98348462561498728,0.98261539538419707,0.98170041619389159,0.98073728020409634,0.97972345284641715,0.97865626615412327,0.97753291174118240,0.97635043341177086,0.97510571938081136,0.97379549408506449, \
                            0.97241630956322578,0.97096453638234292,0.96943635408667672,0.96782774114387016,0.96613446436196859,0.96435206774944060,0.96247586078888481,0.96050090609356287,0.95842200641427666,0.95623369096239641, \
                            0.95393020101304882,0.95150547475057767,0.94895313131639747,0.94626645401726051,0.94343837264974784,0.94046144489447137,0.93732783673102249,0.93402930182212884,0.93055715981276721,0.92690227348712328, \
                            0.92305502472328760,0.91900528918240798,0.91474240966569254,0.91025516806914997,0.90553175586226309,0.90055974301290842,0.89532604527674564,0.88981688976499540,0.88401777869999509,0.87791345126315268, \
                            0.87148784343489749,0.86472404572094463,0.85760425865362588,0.85010974595118505,0.84222078521177368,0.83391661601239331,0.82517538527620349,0.81597408976442465,0.80628851554149961,0.79609317425421011, \
                            0.78536123605706321,0.77406445900743492,0.76217311474466831,0.74965591025754552,0.73647990553425835,0.72261042687816668,0.70801097566122806,0.69264313227497687,0.67646645502629144,0.65943837371188563, \
                            0.64151407759145851,0.62264639746469308,0.60278568154178214,0.58187966478082331,0.55987333134823503,0.53670876984024740,0.51232502088447096,0.48665791672049563,0.45963991233736373,0.43119990772354067, \
                            0.40126306076162166,0.36975059027539114,0.33657956871093797,0.30166270390625044,0.26490810937500037,0.22621906250000029,0.18549375000000023,0.14262500000000017,0.09750000000000011,0.00000000000000000])

        Fxgvp=np.array([0.00000000000000000,0.00000000000002329,0.00000000000003080,0.00000000000003825,0.00000000000004566,0.00000000000005302,0.00000000000006036,0.00000000000006767,0.00000000000007496,0.00000000000008222, \
                            0.00000000000008946,0.00000000000009668,0.00000000000010389,0.00000000000011108,0.00000000000011826,0.00000000000012542,0.00000000000013257,0.00000000000013970,0.00000000000014683,0.00000000000015394, \
                            0.00000000000016105,0.00000000000017522,0.00000000000018936,0.00000000000020347,0.00000000000021754,0.00000000000023158,0.00000000000024560,0.00000000000025959,0.00000000000027356,0.00000000000028750, \
                            0.00000000000030142,0.00000000000032226,0.00000000000034305,0.00000000000036381,0.00000000000038452,0.00000000000040520,0.00000000000042584,0.00000000000045331,0.00000000000048073,0.00000000000050809, \
                            0.00000000000053541,0.00000000000056268,0.00000000000059670,0.00000000000063065,0.00000000000066455,0.00000000000069839,0.00000000000073892,0.00000000000077937,0.00000000000081976,0.00000000000086679, \
                            0.00000000000091373,0.00000000000096059,0.00000000000101405,0.00000000000106741,0.00000000000112733,0.00000000000118714,0.00000000000125348,0.00000000000131970,0.00000000000138580,0.00000000000145838, \
                            0.00000000000153742,0.00000000000161632,0.00000000000170164,0.00000000000179335,0.00000000000188491,0.00000000000198282,0.00000000000208708,0.00000000000219766,0.00000000000231452,0.00000000000243118, \
                            0.00000000000255410,0.00000000000268327,0.00000000000282510,0.00000000000297310,0.00000000000312726,0.00000000000328755,0.00000000000345395,0.00000000000363284,0.00000000000381778,0.00000000000401512, \
                            0.00000000000421846,0.00000000000443411,0.00000000000466204,0.00000000000490220,0.00000000000515453,0.00000000000541899,0.00000000000569555,0.00000000000598415,0.00000000000629101,0.00000000000660981, \
                            0.00000000000694675,0.00000000000730175,0.00000000000767475,0.00000000000806568,0.00000000000847448,0.00000000000890725,0.00000000000936390,0.00000000000984433,0.00000000001034846,0.00000000001087620, \
                            0.00000000001142745,0.00000000001200824,0.00000000001261847,0.00000000001325800,0.00000000001393282,0.00000000001464277,0.00000000001538773,0.00000000001616755,0.00000000001698814,0.00000000001784934, \
                            0.00000000001875699,0.00000000001971091,0.00000000002071093,0.00000000002176282,0.00000000002286640,0.00000000002402741,0.00000000002524562,0.00000000002652673,0.00000000002787641,0.00000000002929438, \
                            0.00000000003078037,0.00000000003234000,0.00000000003397885,0.00000000003570245,0.00000000003751632,0.00000000003942008,0.00000000004141920,0.00000000004351912,0.00000000004572522,0.00000000004804287, \
                            0.00000000005047740,0.00000000005303410,0.00000000005571822,0.00000000005854076,0.00000000006150685,0.00000000006462165,0.00000000006789596,0.00000000007133482,0.00000000007494893,0.00000000007874325, \
                            0.00000000008272835,0.00000000008691477,0.00000000009131294,0.00000000009593325,0.00000000010078601,0.00000000010588707,0.00000000011124658,0.00000000011687463,0.00000000012278681,0.00000000012899860, \
                            0.00000000013551986,0.00000000014237142,0.00000000014956847,0.00000000015713161,0.00000000016507580,0.00000000017342138,0.00000000018218856,0.00000000019139742,0.00000000020106791,0.00000000021122528, \
                            0.00000000022190005,0.00000000023311168,0.00000000024488492,0.00000000025725516,0.00000000027024674,0.00000000028389462,0.00000000029822815,0.00000000031328717,0.00000000032910591,0.00000000034571838, \
                            0.00000000036316898,0.00000000038150182,0.00000000040075537,0.00000000042097845,0.00000000044221952,0.00000000046453195,0.00000000048796877,0.00000000051258260,0.00000000053843619,0.00000000056559180, \
                            0.00000000059411651,0.00000000062407688,0.00000000065554419,0.00000000068859441,0.00000000072330811,0.00000000075976528,0.00000000079806076,0.00000000083827846,0.00000000088052213,0.00000000092488969, \
                            0.00000000097148851,0.00000000102043027,0.00000000107183093,0.00000000112581570,0.00000000118251392,0.00000000124206407,0.00000000130460359,0.00000000137028898,0.00000000143927057,0.00000000151171755, \
                            0.00000000158780283,0.00000000166771294,0.00000000175163299,0.00000000183977147,0.00000000193233527,0.00000000202954452,0.00000000213163241,0.00000000223884508,0.00000000235143655,0.00000000246967845, \
                            0.00000000259385493,0.00000000272425767,0.00000000286120517,0.00000000300502316,0.00000000315604931,0.00000000331465235,0.00000000348120776,0.00000000365611202,0.00000000383978241,0.00000000403266154, \
                            0.00000000423520754,0.00000000444790342,0.00000000467125674,0.00000000490579466,0.00000000515208255,0.00000000541070943,0.00000000568228777,0.00000000596746251,0.00000000626691541,0.00000000658135995, \
                            0.00000000691154099,0.00000000725824827,0.00000000762230669,0.00000000800458052,0.00000000840597752,0.00000000882745760,0.00000000927001855,0.00000000973470922,0.00000001022263807,0.00000001073495886, \
                            0.00000001127289272,0.00000001183771386,0.00000001243076243,0.00000001305344373,0.00000001370723646,0.00000001439369177,0.00000001511443692,0.00000001587117880,0.00000001666570741,0.00000001749990812, \
                            0.00000001837575173,0.00000001929531543,0.00000002026077272,0.00000002127440540,0.00000002233861096,0.00000002345590554,0.00000002462893118,0.00000002586045855,0.00000002715339839,0.00000002851079974, \
                            0.00000002993586974,0.00000003143196736,0.00000003300262289,0.00000003465154432,0.00000003638261512,0.00000003819991747,0.00000004010772973,0.00000004211054931,0.00000004421308984,0.00000004642029943, \
                            0.00000004873737436,0.00000005116976011,0.00000005372318142,0.00000005640363868,0.00000005921743336,0.00000006217117230,0.00000006527179257,0.00000006852657765,0.00000007194316109,0.00000007552956271, \
                            0.00000007929419160,0.00000008324587347,0.00000008739386938,0.00000009174789816,0.00000009631815849,0.00000010111535047,0.00000010615069697,0.00000011143598435,0.00000011698357078,0.00000012280642605, \
                            0.00000012891815879,0.00000013533305111,0.00000014206607691,0.00000014913295892,0.00000015655018965,0.00000016433507138,0.00000017250575538,0.00000018108129204,0.00000019008166075,0.00000019952782618, \
                            0.00000020944178192,0.00000021984660488,0.00000023076650089,0.00000024222686477,0.00000025425433912,0.00000026687687590,0.00000028012379469,0.00000029402585736,0.00000030861532893,0.00000032392605695, \
                            0.00000033999355078,0.00000035685505735,0.00000037454965045,0.00000039311831065,0.00000041260403323,0.00000043305191548,0.00000045450925866,0.00000047702568117,0.00000050065322468,0.00000052544647733, \
                            0.00000055146269204,0.00000057876192398,0.00000060740716393,0.00000063746447502,0.00000066900314541,0.00000070209584433,0.00000073681879316,0.00000077325192292,0.00000081147907045,0.00000085158815473, \
                            0.00000089367138610,0.00000093782545879,0.00000098415178781,0.00000103275671708,0.00000108375177358,0.00000113725390717,0.00000119338576300,0.00000125227594782,0.00000131405931926,0.00000137887729808, \
                            0.00000144687816719,0.00000151821741602,0.00000159305808771,0.00000167157113745,0.00000175393581576,0.00000184034007006,0.00000193098095964,0.00000202606510260,0.00000212580912436,0.00000223044015075, \
                            0.00000234019630144,0.00000245532722578,0.00000257609465611,0.00000270277298912,0.00000283564989013,0.00000297502692592,0.00000312122024375,0.00000327456126230,0.00000343539740155,0.00000360409285323, \
                            0.00000378102938818,0.00000396660718821,0.00000416124572604,0.00000436538470531,0.00000457948500004,0.00000480402969713,0.00000503952512478,0.00000528650198538,0.00000554551651078,0.00000581715167657, \
                            0.00000610201848074,0.00000640075727578,0.00000671403916690,0.00000704256747417,0.00000738707926921,0.00000774834697157,0.00000812718003855,0.00000852442671483,0.00000894097587318,0.00000937775894992, \
                            0.00000983575195931,0.00001031597759921,0.00001081950747140,0.00001134746438879,0.00001190102480095,0.00001248142132935,0.00001308994542025,0.00001372795012719,0.00001439685301140,0.00001509813919229, \
                            0.00001583336451962,0.00001660415892289,0.00001741222988071,0.00001825936607837,0.00001914744122580,0.00002007841805608,0.00002105435250216,0.00002207739806086,0.00002314981038693,0.00002427395206759, \
                            0.00002545229763316,0.00002668743879756,0.00002798208994116,0.00002933909383887,0.00003076142765731,0.00003225220922805,0.00003381470361052,0.00003545232994614,0.00003716866864003,0.00003896746886112, \
                            0.00004085265638742,0.00004282834181726,0.00004489882914601,0.00004706862472885,0.00004934244668073,0.00005172523465993,0.00005422216013896,0.00005683863710748,0.00005958033328556,0.00006245318183555, \
                            0.00006546339359287,0.00006861746987668,0.00007192221584491,0.00007538475448736,0.00007901254121463,0.00008281337913034,0.00008679543497291,0.00009096725578335,0.00009533778631119,0.00009991638720778, \
                            0.00010471285402574,0.00010973743706936,0.00011500086213325,0.00012051435216067,0.00012628964986222,0.00013233904134468,0.00013867538078631,0.00014531211619379,0.00015226331631725,0.00015954369873826, \
                            0.00016716865919287,0.00017515430220210,0.00018351747302904,0.00019227579104550,0.00020144768456899,0.00021105242719663,0.00022111017575490,0.00023164200987544,0.00024266997331476,0.00025421711704525, \
                            0.00026630754423117,0.00027896645713967,0.00029222020607584,0.00030609634043124,0.00032062366191515,0.00033583228008956,0.00035175367025873,0.00036842073384698,0.00038586786134370,0.00040413099792369, \
                            0.00042324771186270,0.00044325726583752,0.00046420069124562,0.00048612086564648,0.00050906259348016,0.00053307269013974,0.00055820006959337,0.00058449583563360,0.00061201337694373,0.00064080846609559, \
                            0.00067093936265647,0.00070246692053608,0.00073545469976806,0.00076996908286305,0.00080607939592829,0.00084385803472431,0.00088338059584691,0.00092472601321692,0.00096797670009600,0.00101321869680819, \
                            0.00106054182438015,0.00111003984434198,0.00116181062486621,0.00121595631351271,0.00127258351680096,0.00133180348684168,0.00139373231529157,0.00145849113487934,0.00152620632875971,0.00159700974798322, \
                            0.00167103893733695,0.00174843736984084,0.00182935469021811,0.00191394696758842,0.00200237695772839,0.00209481437519008,0.00219143617559460,0.00229242684843329,0.00239797872069335,0.00250829227164975, \
                            0.00262357645917757,0.00274404905789588,0.00286993700954025,0.00300147678589284,0.00313891476463763,0.00328250761851089,0.00343252271812239,0.00358923854881172,0.00375294514191248,0.00392394452082318, \
                            0.00410255116224840,0.00428909247298792,0.00448390928267606,0.00468735635281904,0.00489980290254644,0.00512163315140893,0.00535324687963120,0.00559506000615597,0.00584750518485635,0.00611103241924855, \
                            0.00638610969606361,0.00667322363798530,0.00697288017587935,0.00728560524082004,0.00761194547617263,0.00795246897001263,0.00830776600811104,0.00867844984770470,0.00906515751223075,0.00946855060718281, \
                            0.00988931615722018,0.01032816746460003,0.01078584498899467,0.01126311724867299,0.01176078174302688,0.01227966589631479,0.01282062802249202,0.01338455831089656,0.01397237983251536,0.01458504956647511, \
                            0.01522355944633741,0.01588893742567554,0.01658224856235126,0.01730459612078982,0.01805712269146956,0.01884101132672712,0.01965748669187402,0.02050781623048306,0.02139331134260560,0.02231532857450532, \
                            0.02327527081839063,0.02427458852044373,0.02531478089530525,0.02639739714499019,0.02752403768004029,0.02869635534053025,0.02991605661434161,0.03118490284990808,0.03250471146044388,0.03387735711639427, \
                            0.03530477292265281,0.03678895157681318,0.03833194650447488,0.03993587296734444,0.04160290913959714,0.04333529714767423,0.04513534406836828,0.04700542287975280,0.04894797335917853,0.05096550292220924, \
                            0.05306058739604081,0.05523587172056410,0.05749407056987249,0.05983796888663009,0.06227042232132022,0.06479435756799271,0.06741277258770574,0.07012873671044229,0.07294539060584786,0.07586594611268160, \
                            0.07889368591642951,0.08203196306406696,0.08528420030448183,0.08865388924259936,0.09214458929475906,0.09575992643240643,0.09950359170065282,0.10337933949776415,0.10739098560110642,0.11154240492457126, \
                            0.11583752899195972,0.12028034311027920,0.12487488322634435,0.12962523244952884,0.13453551722294174,0.13960990312470933,0.14485259028045916,0.15026780836747788,0.15585981119036307,0.16163287080734412, \
                            0.16759127118572353,0.17373930136417051,0.18008124809879508,0.18662138796910974,0.19336397891907223,0.20031325120742446,0.20747339774046777,0.21484856375924957,0.22244283585182248,0.23026023025979730, \
                            0.23830468044679304,0.24658002389456854,0.25508998809058181,0.26383817566840273,0.27282804865979249,0.28206291181427373,0.29154589493863392,0.30127993420493121,0.31126775237115312,0.32151183785363602, \
                            0.33201442258456859,0.34277745858129488,0.35380259314654328,0.36509114261002823,0.37664406451188559,0.38846192811697344,0.40054488313590636,0.41289262651358966,0.42550436712862705,0.43837878822695847, \
                            0.45151400739001518,0.46490753381106281,0.47855622262264658,0.49245622598248945,0.50660294058397570,0.52099095120953387,0.53561396988960630,0.55046477016512174,0.56553511587571137,0.58081568380735904, \
                            0.59629597942924650,0.61196424482723255,0.62780735779702224,0.64381072088909352,0.65995813899425437,0.67623168381636944,0.69261154328862240,0.70907585364084680,0.72560051140420290,0.74215896212846411, \
                            0.75872196196421826,0.77525730749891753,0.79172952829516596,0.80809953541407320,0.82432421775320308,0.84035597620487401,0.85614218333611913,0.87162455335878974,0.88673840339725829,0.90141178219977902, \
                            0.91556443610444038,0.92910657374808880,0.94193737999269456,0.95394321489739708,0.96499541413742318,0.97494758215871025,0.98363224053554188,0.99085668200452381,0.99639804639804630,1.00000000000000000])

    elif nernst == 0:
        xgvp=np.array([0.99999999999999878,0.99999999999999867,0.99999999999999856,0.99999999999999845,0.99999999999999833,0.99999999999999822,0.99999999999999811,0.99999999999999800,0.99999999999999789,0.99999999999999778, \
                            0.99999999999999767,0.99999999999999756,0.99999999999999734,0.99999999999999711,0.99999999999999689,0.99999999999999667,0.99999999999999645,0.99999999999999623,0.99999999999999600,0.99999999999999578, \
                            0.99999999999999556,0.99999999999999534,0.99999999999999500,0.99999999999999467,0.99999999999999434,0.99999999999999400,0.99999999999999367,0.99999999999999334,0.99999999999999289,0.99999999999999245, \
                            0.99999999999999201,0.99999999999999156,0.99999999999999112,0.99999999999999056,0.99999999999999001,0.99999999999998945,0.99999999999998890,0.99999999999998823,0.99999999999998757,0.99999999999998690, \
                            0.99999999999998612,0.99999999999998535,0.99999999999998457,0.99999999999998368,0.99999999999998279,0.99999999999998179,0.99999999999998079,0.99999999999997968,0.99999999999997857,0.99999999999997746, \
                            0.99999999999997624,0.99999999999997491,0.99999999999997358,0.99999999999997213,0.99999999999997058,0.99999999999996902,0.99999999999996736,0.99999999999996558,0.99999999999996370,0.99999999999996170, \
                            0.99999999999995970,0.99999999999995759,0.99999999999995537,0.99999999999995293,0.99999999999995037,0.99999999999994771,0.99999999999994493,0.99999999999994205,0.99999999999993894,0.99999999999993572, \
                            0.99999999999993228,0.99999999999992872,0.99999999999992495,0.99999999999992095,0.99999999999991673,0.99999999999991229,0.99999999999990763,0.99999999999990274,0.99999999999989764,0.99999999999989220, \
                            0.99999999999988654,0.99999999999988054,0.99999999999987421,0.99999999999986755,0.99999999999986056,0.99999999999985323,0.99999999999984546,0.99999999999983724,0.99999999999982858,0.99999999999981948, \
                            0.99999999999980993,0.99999999999979994,0.99999999999978939,0.99999999999977829,0.99999999999976663,0.99999999999975431,0.99999999999974132,0.99999999999972766,0.99999999999971334,0.99999999999969824, \
                            0.99999999999968237,0.99999999999966560,0.99999999999964795,0.99999999999962941,0.99999999999960987,0.99999999999958933,0.99999999999956768,0.99999999999954492,0.99999999999952094,0.99999999999949563, \
                            0.99999999999946898,0.99999999999944100,0.99999999999941158,0.99999999999938061,0.99999999999934797,0.99999999999931355,0.99999999999927736,0.99999999999923928,0.99999999999919920,0.99999999999915701, \
                            0.99999999999911260,0.99999999999906586,0.99999999999901668,0.99999999999896494,0.99999999999891043,0.99999999999885303,0.99999999999879263,0.99999999999872902,0.99999999999866207,0.99999999999859157, \
                            0.99999999999851741,0.99999999999843936,0.99999999999835720,0.99999999999827072,0.99999999999817968,0.99999999999808387,0.99999999999798295,0.99999999999787670,0.99999999999776490,0.99999999999764722, \
                            0.99999999999752331,0.99999999999739297,0.99999999999725575,0.99999999999711131,0.99999999999695921,0.99999999999679912,0.99999999999663058,0.99999999999645317,0.99999999999626643,0.99999999999606992, \
                            0.99999999999586309,0.99999999999564526,0.99999999999541600,0.99999999999517475,0.99999999999492073,0.99999999999465339,0.99999999999437195,0.99999999999407574,0.99999999999376388,0.99999999999343558, \
                            0.99999999999309008,0.99999999999272637,0.99999999999234346,0.99999999999194045,0.99999999999151623,0.99999999999106970,0.99999999999059963,0.99999999999010480,0.99999999998958400,0.99999999998903577, \
                            0.99999999998845868,0.99999999998785116,0.99999999998721167,0.99999999998653855,0.99999999998583000,0.99999999998508415,0.99999999998429911,0.99999999998347266,0.99999999998260281,0.99999999998168709, \
                            0.99999999998072320,0.99999999997970856,0.99999999997864053,0.99999999997751632,0.99999999997633293,0.99999999997508726,0.99999999997377598,0.99999999997239575,0.99999999997094280,0.99999999996941347, \
                            0.99999999996780364,0.99999999996610911,0.99999999996432531,0.99999999996244771,0.99999999996047118,0.99999999995839062,0.99999999995620059,0.99999999995389532,0.99999999995146871,0.99999999994891442, \
                            0.99999999994622568,0.99999999994339539,0.99999999994041622,0.99999999993728017,0.99999999993397903,0.99999999993050426,0.99999999992684652,0.99999999992299626,0.99999999991894339,0.99999999991467725, \
                            0.99999999991018651,0.99999999990545940,0.99999999990048349,0.99999999989524568,0.99999999988973232,0.99999999988392874,0.99999999987781962,0.99999999987138899,0.99999999986461996,0.99999999985749466, \
                            0.99999999984999433,0.99999999984209931,0.99999999983378873,0.99999999982504073,0.99999999981583232,0.99999999980613929,0.99999999979593601,0.99999999978519571,0.99999999977389020,0.99999999976198961, \
                            0.99999999974946274,0.99999999973627651,0.99999999972239628,0.99999999970778553,0.99999999969240583,0.99999999967621667,0.99999999965917541,0.99999999964123720,0.99999999962235486,0.99999999960247876, \
                            0.99999999958155650,0.99999999955953311,0.99999999953635055,0.99999999951194785,0.99999999948626084,0.99999999945922191,0.99999999943075990,0.99999999940079987,0.99999999936926298,0.99999999933606620, \
                            0.99999999930112227,0.99999999926433913,0.99999999922562011,0.99999999918486326,0.99999999914196125,0.99999999909680126,0.99999999904926440,0.99999999899922565,0.99999999894655323,0.99999999889110858, \
                            0.99999999883274582,0.99999999877131129,0.99999999870664347,0.99999999863857203,0.99999999856691790,0.99999999849149246,0.99999999841209730,0.99999999832852349,0.99999999824055097,0.99999999814794838, \
                            0.99999999805047191,0.99999999794786509,0.99999999783985793,0.99999999772616621,0.99999999760649072,0.99999999748051649,0.99999999734791201,0.99999999720832844,0.99999999706139830,0.99999999690673502, \
                            0.99999999674393159,0.99999999657255956,0.99999999639216786,0.99999999620228197,0.99999999600240208,0.99999999579200216,0.99999999557052854,0.99999999533739847,0.99999999509199833,0.99999999483368240, \
                            0.99999999456177091,0.99999999427554831,0.99999999397426131,0.99999999365711711,0.99999999332328116,0.99999999297187492,0.99999999260197359,0.99999999221260372,0.99999999180274068,0.99999999137130591, \
                            0.99999999091716407,0.99999999043912002,0.99999998993591577,0.99999998940622703,0.99999998884866004,0.99999998826174741,0.99999998764394460,0.99999998699362580,0.99999998630907971,0.99999998558850489, \
                            0.99999998483000507,0.99999998403158430,0.99999998319114136,0.99999998230646459,0.99999998137522583,0.99999998039497451,0.99999997936313101,0.99999997827697995,0.99999997713366306,0.99999997593017165, \
                            0.99999997466333856,0.99999997332983004,0.99999997192613688,0.99999997044856503,0.99999996889322629,0.99999996725602758,0.99999996553266057,0.99999996371859001,0.99999996180904205,0.99999995979899159, \
                            0.99999995768314898,0.99999995545594622,0.99999995311152234,0.99999995064370772,0.99999994804600811,0.99999994531158742,0.99999994243324986,0.99999993940342091,0.99999993621412719,0.99999993285697597, \
                            0.99999992932313253,0.99999992560329731,0.99999992168768137,0.99999991756598039,0.99999991322734771,0.99999990866036592,0.99999990385301674,0.99999989879264917,0.99999989346594642,0.99999988785889093, \
                            0.99999988195672729,0.99999987574392346,0.99999986920412987,0.99999986232013671,0.99999985507382805,0.99999984744613479,0.99999983941698389,0.99999983096524614,0.99999982206868010,0.99999981270387373, \
                            0.99999980284618284,0.99999979246966608,0.99999978154701685,0.99999977004949137,0.99999975794683293,0.99999974520719248,0.99999973179704471,0.99999971768109963,0.99999970282221007,0.99999968718127374, \
                            0.99999967071713025,0.99999965338645291,0.99999963514363466,0.99999961594066800,0.99999959572701891,0.99999957444949350,0.99999955205209834,0.99999952847589291,0.99999950365883461,0.99999947753561536, \
                            0.99999945003748980,0.99999942109209450,0.99999939062325727,0.99999935855079702,0.99999932479031262,0.99999928925296067,0.99999925184522176,0.99999921246865442,0.99999917101963620,0.99999912738909069, \
                            0.99999908146220073,0.99999903311810601,0.99999898222958528,0.99999892866272133,0.99999887227654871,0.99999881292268278,0.99999875044492925,0.99999868467887287,0.99999861545144508,0.99999854258046850, \
                            0.99999846587417729,0.99999838513071293,0.99999830013759250,0.99999821067114991,0.99999811649594728,0.99999801736415495,0.99999791301489993,0.99999780317357878,0.99999768755113549,0.99999756584330046, \
                            0.99999743772978988,0.99999730287346300,0.99999716091943469,0.99999701149414177,0.99999685420435969,0.99999668863616809,0.99999651435386105,0.99999633089880102,0.99999613778821150,0.99999593451390678, \
                            0.99999572054095442,0.99999549530626775,0.99999525821712387,0.99999500864960400,0.99999474594695159,0.99999446941784376,0.99999417833457238,0.99999387193112876,0.99999354940118812,0.99999320989598750, \
                            0.99999285252209202,0.99999247633904420,0.99999208035688858,0.99999166353356683,0.99999122477217561,0.99999076291807953,0.99999027675587315,0.99998976500618220,0.99998922632229703,0.99998865928662839, \
                            0.99998806240697724,0.99998743411260760,0.99998677275011327,0.99998607657906657,0.99998534376743842,0.99998457238677729,0.99998376040713399,0.99998290569171999,0.99998200599128417,0.99998105893819378, \
                            0.99998006204020395,0.99997901267389888,0.99997790807778830,0.99997674534504022,0.99997552141583179,0.99997423306929656,0.99997287691504899,0.99997144938426208,0.99996994672027584,0.99996836496871133, \
                            0.99996669996706455,0.99996494733375207,0.99996310245658104,0.99996116048061157,0.99995911629538059,0.99995696452145322,0.99995469949626647,0.99995231525922779,0.99994980553602919,0.99994716372213599, \
                            0.99994438286540621,0.99994145564779602,0.99993837436610100,0.99993513091168518,0.99993171674914227,0.99992812289383393,0.99992433988824625,0.99992035777710131,0.99991616608115919,0.99991175376964125, \
                            0.99990710923120130,0.99990222024336972,0.99989707394038918,0.99989165677935699,0.99988595450458628,0.99987995211009073,0.99987363380009542,0.99986698294746879,0.99985998204996707,0.99985261268417580, \
                            0.99984485545702717,0.99983668995476538,0.99982809468922662,0.99981904704129110,0.99980952320135896,0.99979949810669355,0.99978894537546681,0.99977783723733349,0.99976614446035095,0.99975383627405356, \
                            0.99974088028847741,0.99972724240892352,0.99971288674623526,0.99969777552235284,0.99968186897089772,0.99966512523252393,0.99964750024476201,0.99962894762606525,0.99960941855375285,0.99958886163552929, \
                            0.99956722277424137,0.99954444502551720,0.99952046844791276,0.99949522994517126,0.99946866310018023,0.99944069800018964,0.99941126105283118,0.99938027479245384,0.99934765767626721,0.99931332386975491, \
                            0.99927718302079460,0.99923914002188896,0.99919909475988311,0.99915694185250847,0.99911257037106149,0.99906586354848570,0.99901669847209018,0.99896494576009487,0.99891046922115245,0.99885312549594996, \
                            0.99879276367994729,0.99872922492626026,0.99866234202764237,0.99859193897646559,0.99851783050154264,0.99843982158057110,0.99835770692691694,0.99827127044938624,0.99818028468356446,0.99808451019322575, \
                            0.99798369494023764,0.99787757362130269,0.99776586696979230,0.99764828102083392,0.99752450633771983,0.99739421719759980,0.99725707073431558,0.99711270603612157,0.99696074319591743,0.99680078231149194, \
                            0.99663240243314932,0.99645516045594662,0.99626858995362799,0.99607219995118734,0.99586547363282873,0.99564786698192498,0.99541880734939470,0.99517769194673122,0.99492388625971706,0.99465672237864944, \
                            0.99437549724068353,0.99407947077966685,0.99376786397859662,0.99343985681957536,0.99309458612586876,0.99273114329038814,0.99234857188461911,0.99194586514170424,0.99152196330705700,0.99107575084953359, \
                            0.99060605352582476,0.99011163529034185,0.98959119504246507,0.98904336320259478,0.98846669810799448,0.98785968221894149,0.98722071812520151,0.98654812434231731,0.98584013088664979,0.98509487461752610, \
                            0.98431039433423795,0.98348462561498728,0.98261539538419707,0.98170041619389159,0.98073728020409634,0.97972345284641715,0.97865626615412327,0.97753291174118240,0.97635043341177086,0.97510571938081136, \
                            0.97379549408506449,0.97241630956322578,0.97096453638234292,0.96943635408667672,0.96782774114387016,0.96613446436196859,0.96435206774944060,0.96247586078888481,0.96050090609356287,0.95842200641427666, \
                            0.95623369096239641,0.95393020101304882,0.95150547475057767,0.94895313131639747,0.94626645401726051,0.94343837264974784,0.94046144489447137,0.93732783673102249,0.93402930182212884,0.93055715981276721, \
                            0.92690227348712328,0.92305502472328760,0.91900528918240798,0.91474240966569254,0.91025516806914997,0.90553175586226309,0.90055974301290842,0.89532604527674564,0.88981688976499540,0.88401777869999509, \
                            0.87791345126315268,0.87148784343489749,0.86472404572094463,0.85760425865362588,0.85010974595118505,0.84222078521177368,0.83391661601239331,0.82517538527620349,0.81597408976442465,0.80628851554149961, \
                            0.79609317425421011,0.78536123605706321,0.77406445900743492,0.76217311474466831,0.74965591025754552,0.73647990553425835,0.72261042687816668,0.70801097566122806,0.69264313227497687,0.67646645502629144, \
                            0.65943837371188563,0.64151407759145851,0.62264639746469308,0.60278568154178214,0.58187966478082331,0.55987333134823503,0.53670876984024740,0.51232502088447096,0.48665791672049563,0.45963991233736373, \
                            0.43119990772354067,0.40126306076162166,0.36975059027539114,0.33657956871093797,0.30166270390625044,0.26490810937500037,0.22621906250000029,0.18549375000000023,0.14262500000000017,0.09750000000000011, \
                            0.40126306076162166,0.36975059027539114,0.33657956871093797,0.30166270390625044,0.26490810937500037,0.22621906250000029,0.18549375000000023,0.14262500000000017,0.09750000000000011,0.00000000000000000])
        
        Fxgvp=np.array([0.00000000000008239,0.00000000000009083,0.00000000000009709,0.00000000000010549,0.00000000000011172,0.00000000000012008,0.00000000000012629,0.00000000000013462,0.00000000000014080,0.00000000000014911, \
                            0.00000000000015527,0.00000000000016355,0.00000000000017796,0.00000000000019232,0.00000000000020666,0.00000000000022096,0.00000000000023523,0.00000000000024947,0.00000000000026369,0.00000000000027789, \
                            0.00000000000029206,0.00000000000030620,0.00000000000032634,0.00000000000034852,0.00000000000036858,0.00000000000039067,0.00000000000041066,0.00000000000043267,0.00000000000046060,0.00000000000048847, \
                            0.00000000000051629,0.00000000000054406,0.00000000000057178,0.00000000000060535,0.00000000000064090,0.00000000000067435,0.00000000000070977,0.00000000000075098,0.00000000000079212,0.00000000000083319, \
                            0.00000000000088000,0.00000000000092875,0.00000000000097540,0.00000000000102977,0.00000000000108404,0.00000000000114599,0.00000000000120582,0.00000000000127330,0.00000000000134065,0.00000000000140789, \
                            0.00000000000148271,0.00000000000156312,0.00000000000164338,0.00000000000172919,0.00000000000182250,0.00000000000191564,0.00000000000201625,0.00000000000212233,0.00000000000223386,0.00000000000235277, \
                            0.00000000000247147,0.00000000000259753,0.00000000000272897,0.00000000000287329,0.00000000000302294,0.00000000000317983,0.00000000000334393,0.00000000000351329,0.00000000000369536,0.00000000000388263, \
                            0.00000000000408445,0.00000000000429143,0.00000000000451095,0.00000000000474297,0.00000000000498745,0.00000000000524432,0.00000000000551356,0.00000000000579512,0.00000000000608894,0.00000000000640043, \
                            0.00000000000672597,0.00000000000706905,0.00000000000742959,0.00000000000780942,0.00000000000820844,0.00000000000862474,0.00000000000906546,0.00000000000953052,0.00000000001001982,0.00000000001053327, \
                            0.00000000001107078,0.00000000001163227,0.00000000001222294,0.00000000001284453,0.00000000001349692,0.00000000001418343,0.00000000001490760,0.00000000001566562,0.00000000001646102,0.00000000001729706, \
                            0.00000000001817360,0.00000000001909932,0.00000000002007041,0.00000000002109031,0.00000000002216221,0.00000000002328592,0.00000000002446999,0.00000000002571060,0.00000000002701626,0.00000000002839185, \
                            0.00000000002983710,0.00000000003135173,0.00000000003294237,0.00000000003461204,0.00000000003636905,0.00000000003821815,0.00000000004015896,0.00000000004219795,0.00000000004433802,0.00000000004658731, \
                            0.00000000004895042,0.00000000005143367,0.00000000005403984,0.00000000005677693,0.00000000005965613,0.00000000006268012,0.00000000006585675,0.00000000006919704,0.00000000007270357,0.00000000007639071, \
                            0.00000000008026095,0.00000000008432509,0.00000000008859561,0.00000000009308317,0.00000000009779668,0.00000000010274666,0.00000000010795188,0.00000000011341931,0.00000000011916265,0.00000000012519530, \
                            0.00000000013153393,0.00000000013818861,0.00000000014518064,0.00000000015252472,0.00000000016024356,0.00000000016835163,0.00000000017686973,0.00000000018581853,0.00000000019521855,0.00000000020509020, \
                            0.00000000021546012,0.00000000022635788,0.00000000023780338,0.00000000024982440,0.00000000026245467,0.00000000027571992,0.00000000028965509,0.00000000030429184,0.00000000031967091,0.00000000033582491, \
                            0.00000000035279106,0.00000000037061398,0.00000000038933961,0.00000000040900571,0.00000000042966225,0.00000000045136041,0.00000000047415404,0.00000000049809820,0.00000000052324437,0.00000000054965916, \
                            0.00000000057740553,0.00000000060655213,0.00000000063716712,0.00000000066932429,0.00000000070309982,0.00000000073857702,0.00000000077583703,0.00000000081497853,0.00000000085608745,0.00000000089926852, \
                            0.00000000094462278,0.00000000099225947,0.00000000104229455,0.00000000109484603,0.00000000115004297,0.00000000120801635,0.00000000126890805,0.00000000133286047,0.00000000140003396,0.00000000147058032, \
                            0.00000000154467524,0.00000000162249592,0.00000000170423128,0.00000000179007339,0.00000000188023593,0.00000000197492954,0.00000000207438066,0.00000000217882838,0.00000000228852435,0.00000000240373132, \
                            0.00000000252472560,0.00000000265179862,0.00000000278525105,0.00000000292540957,0.00000000307260683,0.00000000322719000,0.00000000338953736,0.00000000356003416,0.00000000373908649,0.00000000392712400, \
                            0.00000000412460089,0.00000000433198589,0.00000000454977473,0.00000000477848983,0.00000000501867163,0.00000000527090046,0.00000000553578106,0.00000000581394224,0.00000000610604623,0.00000000641279380, \
                            0.00000000673491549,0.00000000707317842,0.00000000742839369,0.00000000780140647,0.00000000819310667,0.00000000860442683,0.00000000903635100,0.00000000948990646,0.00000000996617094,0.00000001046628137, \
                            0.00000001099142553,0.00000001154285843,0.00000001212188834,0.00000001272989349,0.00000001336831853,0.00000001403868291,0.00000001474258284,0.00000001548169148,0.00000001625776195,0.00000001707263712, \
                            0.00000001792825599,0.00000001882664359,0.00000001976993858,0.00000002076037617,0.00000002180030386,0.00000002289218902,0.00000002403861863,0.00000002524231344,0.00000002650612482,0.00000002783304719, \
                            0.00000002922621893,0.00000003068894336,0.00000003222467647,0.00000003383705288,0.00000003552989245,0.00000003730719150,0.00000003917315588,0.00000004113219194,0.00000004318893002,0.00000004534822820, \
                            0.00000004761518132,0.00000004999514555,0.00000005249372978,0.00000005511683310,0.00000005787064116,0.00000006076164657,0.00000006379665991,0.00000006698283214,0.00000007032767201,0.00000007383904991, \
                            0.00000007752523522,0.00000008139489957,0.00000008545714508,0.00000008972152375,0.00000009419806006,0.00000009889727699,0.00000010383021152,0.00000010900844608,0.00000011444414075,0.00000012015005189, \
                            0.00000012613956346,0.00000013242672452,0.00000013902627834,0.00000014595368453,0.00000015322517749,0.00000016085778822,0.00000016886938514,0.00000017727871789,0.00000018610546321,0.00000019537026146, \
                            0.00000020509477380,0.00000021530172690,0.00000022601496818,0.00000023725951446,0.00000024906161663,0.00000026144881484,0.00000027445000777,0.00000028809551463,0.00000030241714044,0.00000031744825922, \
                            0.00000033322387725,0.00000034978073021,0.00000036715735108,0.00000038539417041,0.00000040453359439,0.00000042462012099,0.00000044570042745,0.00000046782348356,0.00000049104065449,0.00000051540582794, \
                            0.00000054097553208,0.00000056780906527,0.00000059596863329,0.00000062551948882,0.00000065653007900,0.00000068907220118,0.00000072322116767,0.00000075905597353,0.00000079665948223,0.00000083611860456, \
                            0.00000087752450702,0.00000092097280668,0.00000096656379595,0.00000101440266893,0.00000106459974870,0.00000111727075380,0.00000117253704226,0.00000123052590399,0.00000129137083097,0.00000135521182669, \
                            0.00000142219572368,0.00000149247650826,0.00000156621566722,0.00000164358255991,0.00000172475478791,0.00000180991860350,0.00000189926931638,0.00000199301174015,0.00000209136065786,0.00000219454128194, \
                            0.00000230278978416,0.00000241635380406,0.00000253549301044,0.00000266047968508,0.00000279159932809,0.00000292915128922,0.00000307344944126,0.00000322482287990,0.00000338361665636,0.00000355019253849, \
                            0.00000372492982593,0.00000390822618381,0.00000410049853298,0.00000430218396331,0.00000451374072059,0.00000473564920218,0.00000496841303891,0.00000521256019120,0.00000546864412599,0.00000573724504286, \
                            0.00000601897114259,0.00000631445997784,0.00000662437985419,0.00000694943129775,0.00000729034860386,0.00000764790144324,0.00000802289655272,0.00000841617951480,0.00000882863659704,0.00000926119670063, \
                            0.00000971483339464,0.00001019056703959,0.00001068946702123,0.00001121265408200,0.00001176130275978,0.00001233664395489,0.00001293996760503,0.00001357262548857,0.00001423603416258,0.00001493167803948, \
                            0.00001566111260489,0.00001642596778156,0.00001722795146399,0.00001806885320382,0.00001895054807815,0.00001987500073414,0.00002084426962752,0.00002186051145089,0.00002292598576783,0.00002404305987561, \
                            0.00002521421388063,0.00002644204602126,0.00002772927822573,0.00002907876194156,0.00003049348422355,0.00003197657411410,0.00003353130930649,0.00003516112314063,0.00003686961189234,0.00003866054241845, \
                            0.00004053786015656,0.00004250569747691,0.00004456838244579,0.00004673044796332,0.00004899664134209,0.00005137193432735,0.00005386153355709,0.00005647089152949,0.00005920571804798,0.00006207199221169, \
                            0.00006507597494324,0.00006822422208835,0.00007152359812314,0.00007498129047646,0.00007860482451553,0.00008240207920107,0.00008638130347183,0.00009055113335573,0.00009492060987250,0.00009949919773619, \
                            0.00010429680492081,0.00010932380309100,0.00011459104897241,0.00012010990667932,0.00012589227105064,0.00013195059203904,0.00013829790019348,0.00014494783328730,0.00015191466414399,0.00015921332970350, \
                            0.00016685946140534,0.00017486941690202,0.00018326031321904,0.00019205006137141,0.00020125740252429,0.00021090194577613,0.00022100420759660,0.00023158565303044,0.00024266873871124,0.00025427695779334, \
                            0.00026643488684258,0.00027916823481961,0.00029250389419299,0.00030646999431723,0.00032109595713508,0.00033641255532647,0.00035245197300096,0.00036924786902901,0.00038683544313866,0.00040525150487928, \
                            0.00042453454557667,0.00044472481341630,0.00046586439175653,0.00048799728082954,0.00051116948295206,0.00053542909141033,0.00056082638314215,0.00058741391539477,0.00061524662650554,0.00064438194097811, \
                            0.00067487987902262,0.00070680317075090,0.00074021737518850,0.00077519100432892,0.00081179565240077,0.00085010613057149,0.00089020060729187,0.00093216075450987,0.00097607189997922,0.00102202318590097, \
                            0.00107010773414787,0.00112042281830282,0.00117307004281951,0.00122815552951812,0.00128579011174286,0.00134608953645050,0.00140917467451836,0.00147517173960914,0.00154421251587323,0.00161643459484117, \
                            0.00169198162184406,0.00177100355228605,0.00185365691814947,0.00194010510509792,0.00203051864053631,0.00212507549304844,0.00222396138357947,0.00232737010880013,0.00243550387706495,0.00254857365738883, \
                            0.00266679954190007,0.00279041112222019,0.00291964788022262,0.00305475959367013,0.00319600675719980,0.00334366101915972,0.00349800563480463,0.00365933593636695,0.00382795982053699,0.00400419825387072, \
                            0.00418838579670212,0.00438087114608433,0.00458201769833522,0.00479220413176435,0.00501182501013475,0.00524129140747300,0.00548103155478228,0.00573149150927068,0.00599313584668021,0.00626644837731436, \
                            0.00655193288635393,0.00685011389906336,0.00716153747146777,0.00748677200709568,0.00782640910036603,0.00818106440718503,0.00855137854331094,0.00893801801104337,0.00934167615475337,0.00976307414577111, \
                            0.01020296199711377,0.01066211960851728,0.01114135784220065,0.01164151962976271,0.01216348111055650,0.01270815280187696,0.01327648080120365,0.01386944802073349,0.01448807545433553,0.01513342347702739, \
                            0.01580659317698270,0.01650872771999790,0.01724101374626709,0.01800468279921780,0.01880101278603581,0.01963132946942520,0.02049700798999133,0.02139947441852422,0.02234020733729343,0.02332073944931516, \
                            0.02434265921438202,0.02540761251044811,0.02651730431876009,0.02767350043093139,0.02887802917587001,0.03013278316427450,0.03143972104810211,0.03280086929212658,0.03421832395439259,0.03569425247204932, \
                            0.03723089544863892,0.03883056843858380,0.04049566372415343,0.04222865207977034,0.04403208451802602,0.04590859401127452,0.04786089718212296,0.04989179595555041,0.05200417916477262,0.05420102410230367, \
                            0.05648539800694871,0.05886045947672448,0.06132945979688396,0.06389574417136250,0.06656275284507146,0.06933402210346529,0.07221318513479415,0.07520397273934400,0.07831021386879809,0.08153583597761588, \
                            0.08488486516699317,0.08836142610057295,0.09196974166958120,0.09571413238347948,0.09959901546054924,0.10362890359103989,0.10780840334362025,0.11214221318388108,0.11663512107150104,0.12129200160045543, \
                            0.12611781264425778,0.13111759146571389,0.13629645024801165,0.14165957100114587,0.14721219979472519,0.15295964026505887,0.15890724634113290,0.16506041413059655,0.17142457290322585,0.17800517510547806, \
                            0.18480768533572681,0.19183756820551714,0.19910027500778135,0.20660122910831366,0.21434580997200703,0.22233933573035644,0.23058704419155940,0.23909407218920670,0.24786543316007856,0.25690599283596083, \
                            0.26622044292870323,0.27581327268201039,0.28568873815772594,0.29585082911871774,0.30630323336497034,0.31704929837424950,0.32809199009383705,0.33943384872551019,0.35107694134231904,0.36302281117303964, \
                            0.37527242338872230,0.38782610722580640,0.40068349428227479,0.41384345282769669,0.42730401797534595,0.44106231757555359,0.45511449370485735,0.46945561964631827,0.48407961228375918,0.49897913986798637, \
                            0.51414552515797673,0.52956864399650772,0.54523681945013691,0.56113671173062907,0.57725320422226778,0.59356928607098702,0.61006593195176961,0.62672197982599465,0.64351400773731304,0.66041621098132308, \
                            0.67740028133064756,0.69443529041457042,0.71148757985502076,0.72852066136476068,0.74549513073844453,0.76236860053543876,0.77909565729133823,0.79562785033342065,0.81191372074852575,0.82789888079866614, \
                            0.84352615614186854,0.85873580563653218,0.87346583632789132,0.88765243446754072,0.90123053711472401,0.91413457298782075,0.92629940568740887,0.93766151699874212,0.94816047231839817,0.95774071365660618, \
                            0.96635372698529665,0.97396062801248984,0.98053520063263322,0.98606740025068629,0.99056729166267465,0.99406931466582849,0.99663663725038487,0.99836512560942203,0.99938605369177969,0.99986591374070610, \
                            0.97396062801248984,0.98053520063263322,0.98606740025068629,0.99056729166267465,0.99406931466582849,0.99663663725038487,0.99836512560942203,0.99938605369177969,0.99986591374070610,1.00000000000000000,])
    return Fxgvp, xgvp
