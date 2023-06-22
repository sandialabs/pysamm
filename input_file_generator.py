from pySAMM import *
import os
import sys

DUMP_PATH = os.getenv('DUMP_PATH','/Users/thomoor/data/run_parameters/pysamm/input_files/')
PARAMETER_BASES = os.getenv('PARAMETER_BASES')
PARAMETER_RANGES = os.getenv('PARAMETER_RANGES')

def main():
    global PARAMETER_RANGES
    ensemble = parameter_ensemble()
    ensemble.set_dist('gaussian')
    ensemble.set_path(DUMP_PATH)
    ensemble.set_base_values(rhog0=2.25)
    ensemble.set_ranges(rhog0=[0.5,4.0])
    # ensemble.set_ranges(rhog0=[0.5,4.0],ftrit=[],fdops=[],Adops=[],Zdops=[],Nr=[], \
    #                     rl0=[],rhol0=[],AR=[],h=[],Nls=[], \
    #                     Bz0=[],Troom=[],tmin=[],tmax=[], \
    #                     Ezbl=[],rLEC=[],rdump=[],tzbl=[],tauzbl=[],lambda_b=[],xi_b=[],deltazlec=[],fsharp=[],bestfocus=[],dolew=[], \
    #                     Taur=[], \
    #                     Z0=[],L=[],L0=[],C=[],Rloss=[], \
    #                     VI_mode=[],nernst=[],laserabs=[],n_endloss=[],rphf=[])
    ensemble.set_mu_sigma()
    ensemble.input_file_generator_loop(100)
    
    # set_trace()
    return

class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

class parameter_ensemble(HasTraits):
    parameters = {'rhog0':[],'ftrit':[],'fdops':[],'Adops':[],'Zdops':[],'Nr':[], \
                'rl0':[],'rhol0':[],'AR':[],'h':[],'Nls':[], \
                'Bz0':[],'Troom':[],'tmin':[],'tmax':[], \
                'Ezbl':[],'rLEC':[],'rdump':[],'tzbl':[],'tauzbl':[],'lambda_b':[],'xi_b':[],'deltazlec':[],'fsharp':[],'bestfocus':[],'dolew':[], \
                'Taur':[], \
                'Z0':[],'L':[],'L0':[],'C':[],'Rloss':[], \
                'VI_mode':[],'nernst':[],'laserabs':[],'n_endloss':[],'rphf':[]}
                
    ranges = {'rhog0':[],'ftrit':[],'fdops':[],'Adops':[],'Zdops':[],'Nr':[], \
                'rl0':[],'rhol0':[],'AR':[],'h':[],'Nls':[], \
                'Bz0':[],'Troom':[],'tmin':[],'tmax':[], \
                'Ezbl':[],'rLEC':[],'rdump':[],'tzbl':[],'tauzbl':[],'lambda_b':[],'xi_b':[],'deltazlec':[],'fsharp':[],'bestfocus':[],'dolew':[], \
                'Taur':[], \
                'Z0':[],'L':[],'L0':[],'C':[],'Rloss':[]}
                
    dist_type = Str(default_value='gaussian')
    
    defaults = InputDeck()
    temp_ = OdeSystem()
    defaults.rphf = temp_.rb/temp_.rg0
    del temp_
    for key in parameters:
        parameters[key] = {'base':[],'range':[],'mu':[],'sigma':[],'sample':[]}
        parameters[key]['base'] = getattr(defaults,key)
                
    def set_path(self,path):
        self.path = path
        return
    
    def set_base_values(self, **kwargs):
        for key,value in kwargs.iteritems():
            self.parameters[key]['base'] = value
        return
        
    def set_ranges(self, **kwargs):
        for key,value in kwargs.iteritems():
            self.parameters[key]['range'] = value
        return
        
    def set_mu_sigma(self):
        for key in self.parameters:
            self.parameters[key]['mu'] = self.parameters[key]['base']
            if self.parameters[key]['range'] != []:
                self.parameters[key]['sigma'] = abs(self.parameters[key]['range'][1]-self.parameters[key]['base'])/3.0
            else: self.parameters[key]['sigma'] = self.parameters[key]['base']
        return
        
    def set_dist(self,dist_type):
        if dist_type=='gaussian' or dist_type=='log_normal': #or dist_type=='log_normal':
            self.dist_type = dist_type
        else: print('Only gaussian and linear distributions are currently supported')
        return
    
    def randomize_inputs(self):
        if self.dist_type=='gaussian':
            for key in self.parameters:
                if self.parameters[key]['range'] != []:
                    self.parameters[key]['sample'] = np.random.normal(self.parameters[key]['mu'],self.parameters[key]['sigma'],1)
                else: self.parameters[key]['sample'] = self.parameters[key]['base']
        elif self.dist_type=='linear':
            for key in self.parameters:
                if self.parameters[key]['range'] != []:
                    self.parameters[key]['sample'] = self.parameters[key]['range'][0] + (self.parameters[key]['range'][1]-self.parameters[key]['range'][0])*np.random.random(1)
                else: self.parameters[key]['sample'] = self.parameters[key]['base']
        return

    def input_file_generator_loop(self,iterations):
        f = open(self.path+'file_list.txt', 'w')
        # test = np.array([])
        for i in range(0,iterations):
            self.randomize_inputs()
            ps_instance = InputDeck(param_file=self.path+str(i)+'.npz',\
                            rhog0=self.parameters['rhog0']['sample'],ftrit=self.parameters['ftrit']['sample'],fdops=self.parameters['fdops']['sample'],Adops=self.parameters['Adops']['sample'], \
                            Zdops=self.parameters['Zdops']['sample'],Nr=self.parameters['Nr']['sample'],rl0=self.parameters['rl0']['sample'],rhol0=self.parameters['rhol0']['sample'], \
                            AR=self.parameters['AR']['sample'],h=self.parameters['h']['sample'],Nls=self.parameters['Nls']['sample'],Bz0=self.parameters['Bz0']['sample'], \
                            Troom=self.parameters['Troom']['sample'],tmin=self.parameters['tmin']['sample'],tmax=self.parameters['tmax']['sample'],Ezbl=self.parameters['Ezbl']['sample'], \
                            rLEC=self.parameters['rLEC']['sample'],rdump=self.parameters['rdump']['sample'],tzbl=self.parameters['tzbl']['sample'],tauzbl=self.parameters['tauzbl']['sample'], \
                            lambda_b=self.parameters['lambda_b']['sample'],xi_b=self.parameters['xi_b']['sample'],deltazlec=self.parameters['deltazlec']['sample'],fsharp=self.parameters['fsharp']['sample'], \
                            bestfocus=self.parameters['bestfocus']['sample'],dolew=self.parameters['dolew']['sample'],Taur=self.parameters['Taur']['sample'],Z0=self.parameters['Z0']['sample'], \
                            L=self.parameters['L']['sample'],L0=self.parameters['L0']['sample'],C=self.parameters['C']['sample'],Rloss=self.parameters['Rloss']['sample'], \
                            VI_mode=self.parameters['VI_mode']['sample'],nernst=self.parameters['nernst']['sample'],laserabs=self.parameters['laserabs']['sample'],n_endloss=self.parameters['n_endloss']['sample'], \
                            rphf=self.parameters['rphf']['sample'])
            ps_instance.save()
            # test = np.append(test,self.parameters['rhog0']['sample'])
            f.write(str(i)+'.npz\n')
        # f.close()
        # plt.figure(2)
        # plt.hist(test)
        # plt.show()
        return

if __name__ == "__main__":
    main()