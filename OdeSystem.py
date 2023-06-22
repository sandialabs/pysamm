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
global dimensionless_curve

from pySAMM import PhysicalConstants, InputDeck, Liner, Fuel, ZBeamlet, ReactivityParameters, CircuitModel, EnergyYields

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
        
        global dimensionless_curve
        from dimensionless_curve import dimensionless_curve
        dimensionless_curve = self.dimensionless_curve
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