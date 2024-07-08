###########################################################################
# AutoDepGraph for Quantum Inspire Starmon-5
###########################################################################
"""
This version of graph-based tuneup (GBT) is designed specifically for 
Quantum Inspire Starmon-5. 
"""

from importlib import reload    #Leo test

import autodepgraph #Leo test
reload(autodepgraph) #Leo test
from autodepgraph import AutoDepGraph_DAG

# import infinity.calibration
# reload(infinity.calibration) #Leo test
import numpy as np

class inspire_dep_graph(AutoDepGraph_DAG):
    def __init__(self, name: str, device, **kwargs):
        super().__init__(name, **kwargs)
        self.device = device

        qubits = []
        for qubit in self.device.qubits():
            if qubit != 'fakequbit':
                qubits.append(self.device.find_instrument(qubit))
        self.create_dep_graph(Qubit_list=qubits)

    def create_dep_graph(self, Qubit_list):
        print('Creating Graph ...')

        ########################################################
        # GRAPH NODES
        ########################################################
        for Qubit in Qubit_list:
            ################################
            # Qubit Readout Calibration
            ################################
            spectator_list = list([qubit for qubit in self.device.qubits() \
                          if self.device.find_instrument(qubit).instr_LutMan_RO()==Qubit.instr_LutMan_RO()])
            if Qubit.name=='QSE' or Qubit.name=='QSW' or Qubit.name=='QC':
                # spectator_list.pop(spectator_list.index('QSW'))
                spectator_list = list([Qubit.name])


            #self.add_node(Qubit.name + ' Optimal-Weights Calibration',
            #            calibrate_function=self.device.name + '.calibrate_optimal_weights_mux',
            #            calibrate_function_args={'qubits': spectator_list, 'q_target': Qubit.name, 'return_analysis': False})

            #self.add_node(Qubit.name + ' SSRO Calibration',
            #            calibrate_function=Qubit.name + '.calibrate_ssro_fine',
            #            calibrate_function_args={'check_threshold': 0.88, 'optimize_threshold': 0.95, 'nr_shots_per_case': 2**15,})

            ################################
            # Single Qubit Gate Assessment
            ################################
            #self.add_node(Qubit.name + ' T1',
            #               calibrate_function = Qubit.name + '.measure_T1')
            #self.add_node(Qubit.name + ' T2_Star',
            #               calibrate_function = Qubit.name + '.measure_ramsey')
            #self.add_node(Qubit.name + ' T2_Echo',
            #               calibrate_function = Qubit.name + '.measure_echo')
            #self.add_node(Qubit.name + ' ALLXY',
            #              calibrate_function = Qubit.name + '.allxy_GBT')

            ################################
            # Single Qubit Gate Calibration
            ################################
            #self.add_node(Qubit.name + ' Frequency Fine',
            #              calibrate_function=Qubit.name + '.calibrate_frequency_ramsey',
            #              calibrate_function_args={'steps':[10, 30]})
            #              #check_function=Qubit.name + '.check_ramsey', tolerance=0.1e-3)
            self.add_node(Qubit.name + ' Flipping',
                          calibrate_function=Qubit.name + '.flipping_GBT',
                          calibrate_function_args={'nr_sequence': 5})
            #self.add_node(Qubit.name + ' MOTZOI Calibration',
            #              calibrate_function=Qubit.name + '.calibrate_motzoi',
            #              calibrate_function_args={'motzois': np.arange(0,0.16,.01)})
            #self.add_node(Qubit.name + ' Second Flipping',
            #              calibrate_function=Qubit.name + '.flipping_GBT')
            #self.add_node(Qubit.name + ' ALLXY',
            #              calibrate_function=Qubit.name + '.allxy_GBT')
            #self.add_node(Qubit.name + ' RB Fidelity',
            #              calibrate_function=Qubit.name + '.measure_single_qubit_randomized_benchmarking',
            #              calibrate_function_args={'recompile': True, 'nr_seeds': 50, 'nr_cliffords': 2 ** np.arange(11)})

            ###################################################################
            # Qubit Dependencies
            ###################################################################
            # First depends on second being done
            #self.add_edge(Qubit.name + ' Optimal-Weights Calibration',
            #              Qubit.name + ' SSRO Calibration')
            #self.add_edge(Qubit.name + ' T1',
            #              Qubit.name + ' Optimal-Weights Calibration')
            ##self.add_edge(Qubit.name + ' T2_Star',
            ##              Qubit.name + ' T1')
            #self.add_edge(Qubit.name + ' T2_Echo',
            #              Qubit.name + ' T1'), #' T2_Star')
            #self.add_edge(Qubit.name + ' Frequency Fine',
            #              Qubit.name + ' T2_Echo')
            #
            #self.add_edge(Qubit.name + ' Flipping',
            #              Qubit.name + ' Frequency Fine')
            #self.add_edge(Qubit.name + ' MOTZOI Calibration',
            #              Qubit.name + ' Flipping')
            #self.add_edge(Qubit.name + ' Second Flipping',
            #              Qubit.name + ' MOTZOI Calibration')
            #self.add_edge(Qubit.name + ' ALLXY',
            #              Qubit.name + ' MOTZOI Calibration') # Second Flipping')
            #self.add_edge(Qubit.name + ' RB Fidelity',
            #              Qubit.name + ' ALLXY')

        ################################
        # Multiplexed Readout Assessment
        ################################
        #self.add_node('Device SSRO Fidelity',
        #                calibrate_function=self.device.name + '.measure_ssro_multi_qubit',
        #                calibrate_function_args={'qubits': self.device.qubits(), 'initialize': True})

        ################################
        # Two-qubit Calibration
        ################################
        
        # Leo skipping this for now!
        #cardinal = {str(['QNW','QC']):'SE', str(['QNE','QC']):'SW', str(['QC','QSW']):'SW', str(['QC','QSE']):'SE', \
        #      str(['QC','QSW','QSE']):'SW', str(['QC','QSE','QSW']):'SE'}

        #for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
        #   flux_lm = self.device.find_instrument(self.device.find_instrument(pair[0]).instr_LutMan_Flux())
        #   center = flux_lm.parameters['q_amp_center_{}'.format(cardinal[str(pair)])].get()
        #   q_parks = [pair[2]] if len(pair)==3 else ['QNW'] if pair[0]=='QNE' else ['QNE']
        #   self.add_node('{}-{} Gate Calibration'.format(pair[0], pair[1]),
        #                  calibrate_function=self.device.name + '.measure_vcz_A_B_landscape',
        #                  calibrate_function_args={ 'Q0': [pair[0]], 'Q1': [pair[1]], 'update_flux_params': True,
        #                                            'A_points': 21, 'A_ranges': [(.98*center, 1.02*center)], 
        #                                            'B_amps': np.linspace(0, 1, 21),
        #                                            'Q_parks': q_parks})

        #   skip_reverse = False
        #   self.add_node('{}-{} Update Phase Correction'.format(pair[0], pair[1]),
        #                  calibrate_function=self.device.name + '.calibrate_phases',
        #                  calibrate_function_args={'skip_reverse': skip_reverse, 'operation_pairs': list([(pair,cardinal[str(pair)])]) })

        ################################
        # Two-qubit Assessment
        ################################
        #for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
        #   self.add_node('{}-{} Conditional Oscillation'.format(pair[0], pair[1]),
        #                  calibrate_function=self.device.name + '.measure_conditional_oscillation',
        #                  calibrate_function_args={'q0':pair[0], 'q1':pair[1], 'q2':pair[2] if len(pair)==3 else None,
        #                                  'parked_qubit_seq':'ramsey' if len(pair)==3 else None})

        #for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
        #   self.add_node('{}-{} Randomized Benchmarking'.format(pair[0], pair[1]),
        #                  calibrate_function=self.device.name + '.measure_two_qubit_interleaved_randomized_benchmarking',
        #                  calibrate_function_args={'qubits':pair, 'MC':self.device.instr_MC.get_instr(), 'recompile': False, 
        #                                  'measure_idle_flux': False, 'cardinal': cardinal[str(pair)]})

        ################################
        # Device Dependencies
        ################################
        self.add_node('Prep Inspire',
                      calibrate_function=self.device.name + '.prepare_for_inspire')
        self.add_node('Upload Calibration Results',
                      calibrate_function='infinity.calibration.calibration')

        #for Qubit in Qubit_list:
        #   self.add_edge('Device SSRO Fidelity',
        #                   Qubit.name + ' RB Fidelity')

        #for pair in [['QNW','QC'], ['QNE','QC'], ['QC','QSW','QSE'], ['QC','QSE','QSW']]:
        #   self.add_edge('{}-{} Gate Calibration'.format(pair[0], pair[1]),
        #                  'Device SSRO Fidelity')
        #   self.add_edge('{}-{} Update Phase Correction'.format(pair[0], pair[1]),
        #                  '{}-{} Gate Calibration'.format(pair[0], pair[1]))
        #   self.add_edge('{}-{} Conditional Oscillation'.format(pair[0], pair[1]),
        #                  '{}-{} Update Phase Correction'.format(pair[0], pair[1]))
        #   self.add_edge('{}-{} Randomized Benchmarking'.format(pair[0], pair[1]),
        #                  '{}-{} Conditional Oscillation'.format(pair[0], pair[1]))
        #   self.add_edge('Prep Inspire',
        #                  '{}-{} Randomized Benchmarking'.format(pair[0], pair[1]))
        
        #----------------------------------------
        #Leo simplification to skip 2Q gates
        #self.add_edge('Prep Inspire',
        #                  'Device SSRO Fidelity')

        for Qubit in Qubit_list:
           self.add_edge('Prep Inspire',
                           Qubit.name + ' Flipping')

        #----------------------------------------
      
        self.add_edge('Upload Calibration Results', 
                          'Prep Inspire')

        self.cfg_plot_mode = 'svg'
        self.update_monitor()
        self.cfg_svg_filename

        url = self.open_html_viewer()
        print('Dependency Graph Created. URL = ' + url)

class inspire_dep_graph_RO(AutoDepGraph_DAG):
    def __init__(self, 
                  name: str, 
                  device, 
                  doSSRO=False,
                  SSROnumshots=2**16,
                  **kwargs):
        super().__init__(name, **kwargs)
        self.device = device

        # make the list of qubit objects
        qubitobjects = []
        #for qubit in self.device.qubits():
        #    if qubit != 'fakequbit':
        #        qubitobjects.append(self.device.find_instrument(qubit))

        # Prefer to order qubits in the following way
        qubitnames=['NW', 'W', 'NE', 'C', 'E', 'SW', 'SE']
        #qubitnames=['QNE', 'QC', 'QSW', 'QSE']
        for qubitname in qubitnames:
            if qubitname != 'fakequbit':
                qubitobjects.append(self.device.find_instrument(qubitname))

        self.create_dep_graph(Qubit_list=qubitobjects,
                              doSSRO=doSSRO,
                              SSROnumshots=SSROnumshots)

    def create_dep_graph(self, 
                      Qubit_list,
                      doSSRO,
                      SSROnumshots):
        print('Creating dependency graph RO ...')

        ##################
        # Create all nodes
        ##################
        for Qubit in Qubit_list:
        

          if doSSRO:
            self.add_node(Qubit.name + ' SSRO',
                        calibrate_function=Qubit.name + '.calibrate_ssro_fine',
                        calibrate_function_args={'check_threshold': 0.88, 
                                                      'optimize_threshold': 0.95, 
                                                      'nr_shots_per_case': SSROnumshots})


          # Calibration of optimal weights using MUXED readout. 
          # Why is this spectator list so awkward? 
          # for QNW, we cycle through all other qubits in feedline.
          # but for QSW, QSE and for QC, we do not. 
          # ....why did Miguel make this choice? 
          spectator_list = list([qubit for qubit in self.device.qubits() \
                          if self.device.find_instrument(qubit).instr_LutMan_RO()==Qubit.instr_LutMan_RO()])
          if Qubit.name=='QSE' or Qubit.name=='QSW' or Qubit.name=='QC' or Qubit.name=="QNW":
              # spectator_list.pop(spectator_list.index('QSW'))
              spectator_list = list([Qubit.name])

          self.add_node(Qubit.name + ' Optimal Weights',
                        calibrate_function=self.device.name + '.calibrate_optimal_weights_mux',
                        calibrate_function_args={'qubits': [Qubit.name], # for QI put [Qubit.name]
                                                  'q_target': Qubit.name, 
                                                  'return_analysis': False,
                                                  'averages': 2 ** 16,   # this is the number of avgs to use for each transient
                                                  'soft_averaging': 15,
                                                  'update': True,
                                                  'verify': True,
                                                  'disable_metadata': True})

        #NewQubitList=['QSW', 'QSE', 'QC', 'QNE']
        #self.add_node('Cross Fidelity',
        #                calibrate_function=self.device.name + '.measure_ssro_multi_qubit',
        #                calibrate_function_args={'qubits': NewQubitList, 'initialize': True})

        qubit_list = ['NW', 'W', 'NE', 'C', 'E', 'SW', 'SE']

        self.add_node('Cross Fidelity',
                        calibrate_function=self.device.name + '.measure_ssro_multi_qubit',
                        calibrate_function_args={'qubits': qubit_list, 'initialize': True, 'disable_metadata': True})

        #########################
        # Create all dependencies
        #########################

        for Qubit in Qubit_list:

          if doSSRO:
            self.add_edge(Qubit.name + ' Optimal Weights',
                             Qubit.name + ' SSRO')
          self.add_edge('Cross Fidelity',
                           Qubit.name + ' Optimal Weights')
          
        self.add_node('Prep Inspire',
                        calibrate_function=self.device.name + '.prepare_for_inspire')
        self.add_node('Upload Calibration Results',
                        calibrate_function='infinity.calibration.calibration')
        
        self.add_edge('Prep Inspire', 'Cross Fidelity')
        self.add_edge('Upload Calibration Results', 'Prep Inspire')

        self.cfg_plot_mode = 'svg'
        self.update_monitor()
        self.cfg_svg_filename
        url = self.open_html_viewer()
        print('Dependency graph RO created. URL = ' + url)

class inspire_dep_graph_QUICK(AutoDepGraph_DAG):
    def __init__(self, 
                  name: str, 
                  device,
                  epsFlipping=0.001,
                  epsSQP=2, 
                  epsSQPP=3,
                  doSQPs=False,
                  **kwargs):
        super().__init__(name, **kwargs)
        self.device = device

        # make the list of qubit objects
        qubitobjects = []

        #for qubit in self.device.qubits():
        #    if qubit != 'fakequbit':
        #        qubitobjects.append(self.device.find_instrument(qubit))
        
        # I prefer to use this order. Change startup sequence later.
        qubitnames=['NW', 'NE', 'W', 'C', 'E', 'SW', 'SE']
        #qubitnames=['QNE', 'QC', 'QSW', 'QSE'] # used when QNW misbehaves.
        for qubitname in qubitnames:
            if qubitname != 'fakequbit':
                qubitobjects.append(self.device.find_instrument(qubitname))

        # doing this very manually for now
        CZindices=[0, 1, 2, 3, 4, 5, 6, 7]
        #CZindices=[1,3,4]; # used when QNW misbehaves.

        #print(qubitobjects)

        self.create_dep_graph(Qubit_list=qubitobjects,
                              CZindices=CZindices,
                              epsFlipping=epsFlipping,
                              epsSQP=epsSQP,
                              epsSQPP=epsSQPP,
                              doSQPs=doSQPs)

    def create_dep_graph(self, 
                    Qubit_list,
                    CZindices,
                    epsFlipping,
                    epsSQP,
                    epsSQPP,
                    doSQPs):
      
      print('Creating dependency graph QUICK ...')

      ####################
      # Single-qubit nodes
      ####################
      for Qubit in Qubit_list:

 
        self.add_node(Qubit.name + ' Flipping',
                          calibrate_function=Qubit.name + '.flipping_GBT',
                          calibrate_function_args={'nr_sequence': 5, 'eps': epsFlipping, 'disable_metadata': True})


      allCZpairs = [
         ['NW', 'W', 'C'],
         ['NW', 'C', 'W'],
         ['NE', 'C', 'E'],
         ['NE', 'E', 'C'],
         ['W', 'SW'],
         ['C', 'SW', 'SE'],
         ['C', 'SE', 'SW'],
         ['E', 'SE']
      ]

      allCZnames = [
         'CZ_NW_W',
         'CZ_NW_C',
         'CZ_NE_C',
         'CZ_NE_E',
         'CZ_W_SW',
         'CZ_C_SW',
         'CZ_C_SE',
         'CZ_E_SE'
      ]

      #################
      # Two-qubit nodes
      #################
      for CZindex in CZindices:

        CZname=allCZnames[CZindex]
        pair=allCZpairs[CZindex]
        # reverse the first two elements in pair, and determine if there is a qubit to park
        reversedpair=[pair[1], pair[0]]
        parkingqubitexists=False
        if(len(pair)==3):
          reversedpair.append(pair[2])
          parkingqubitexists=True

        if CZindex == 0:
          flux_lm_C = self.device.find_instrument('flux_lm_C')
          flux_lm_C.cfg_awg_channel_amplitude(0.32)
          self.device.prepare_for_timedomain(qubits = ['C'], bypass_flux = False)
        if CZindex == 3:
          flux_lm_C = self.device.find_instrument('flux_lm_C')
          flux_lm_C.cfg_awg_channel_amplitude(0.392)
          self.device.prepare_for_timedomain(qubits = ['C'], bypass_flux = False)  
        
        # Calibrate single-qubit phase of pulsed qubit
        self.add_node(CZname+' SQP Pulsed',
                        calibrate_function=self.device.name + '.calibrate_single_qubit_phase_GBT',
                        calibrate_function_args={'pair': pair, 'eps': epsSQP, 'disable_metadata': True}
                        )
        # Calibrate single-qubit phase of pulsed qubit
        if(doSQPs):
          self.add_node(CZname + ' SQP Static',
                        calibrate_function=self.device.name + '.calibrate_single_qubit_phase_GBT',
                        calibrate_function_args={'pair': reversedpair, 'eps': epsSQP, 'disable_metadata': True}
                        )
        # Calibrate single-qubit phase of parked qubit, if it exists
        if(parkingqubitexists):
          ##### need to add specific routing for parked qubit. 
          ##### the one below does not do the job!!!!!
          self.add_node(CZname +' SQP Parked',
                          calibrate_function=self.device.name + '.calibrate_parking_phase_GBT',
                          calibrate_function_args={'pair': reversedpair, 'eps': epsSQPP, 'disable_metadata': True}
                          )

      ##############
      # Device Nodes
      ##############
      self.add_node('Prep Inspire',
                      calibrate_function=self.device.name + '.prepare_for_inspire')
      self.add_node('Upload Calibration Results',
                      calibrate_function='infinity.calibration.calibration')
      
      ###################
      # Node dependencies
      ###################
      #for Qubit in Qubit_list:
      #  self.add_edge('Prep Inspire',
      #                  Qubit.name + ' Flipping')

      for CZindex in CZindices:
        CZname=allCZnames[CZindex]
        pair=allCZpairs[CZindex]
        # reverse the first two elements in pair, and determine if there is a qubit to park
        reversedpair=[pair[1], pair[0]]
        parkingqubitexists=False
        if(len(pair)==3):
          reversedpair.append(pair[2])
          parkingqubitexists=True

        self.add_edge('Prep Inspire',
                        CZname + ' SQP Pulsed')

        self.add_edge(CZname + ' SQP Pulsed',
                        pair[0] +' Flipping')

        if(doSQPs):
          self.add_edge('Prep Inspire',
                        CZname + ' SQP Static')

          self.add_edge(CZname + ' SQP Static',
                        pair[1] +' Flipping')

        if(parkingqubitexists):
          ##### need to add specific routing for parked qubit. 
          ##### the one below does not do the job!!!!!
          self.add_edge('Prep Inspire',
                        CZname + ' SQP Parked')

          self.add_edge(CZname + ' SQP Parked',
                        pair[2] +' Flipping')

      self.add_edge('Upload Calibration Results', 
                          'Prep Inspire')

      # finishing up
      self.cfg_plot_mode = 'svg'
      self.update_monitor()
      self.cfg_svg_filename
      url = self.open_html_viewer()
      print('Dependency graph QUICK created. URL = ' + url)

class inspire_dep_graph_1Q(AutoDepGraph_DAG):
    def __init__(self, 
                name: str,  
                qubitname: str,
                device,
                doSSRO=False,           # determines whether to include SSRO-related nodes
                SSROnumshots=2**16,     # number of shots to use in SSRO node
                RBnumseeds=50,          # number of seeds to perform in RB
                RBrecompile=True,       # determines whether to always recompile RB sequences
                epsAllXY=0.02,          # threshold for AllXY node
                epsFlipping=0.0005,     # threshold for Flipping node
                **kwargs):
        super().__init__(name, **kwargs)

        self.device = device

        # Make the list of qubit objects.
        # In this case, there will alays be only one object specificed by qubitname.
        # This might look wonky on a first glance: why do a for loop over qubits if there is only one?).
        # The reason for this is that I am reusing code first written by Miguel for a graph with possibly multiple qubits.
        qubitobjects=[self.device.find_instrument(qubitname)]

        self.create_dep_graph(Qubit_list=qubitobjects,
                                doSSRO=doSSRO, 
                                SSROnumshots=SSROnumshots,
                                RBnumseeds=RBnumseeds, 
                                RBrecompile=RBrecompile,
                                epsAllXY=epsAllXY,
                                epsFlipping=epsFlipping)


    def create_dep_graph(self, 
                            Qubit_list,
                            doSSRO,
                            SSROnumshots,
                            RBnumseeds,
                            RBrecompile,
                            epsAllXY,
                            epsFlipping):

        print('Creating dependency graph 1Q ...')

        #############
        # Grah nodes
        #############
        for Qubit in Qubit_list:
        
            spectator_list = list([qubit for qubit in self.device.qubits() \
                          if self.device.find_instrument(qubit).instr_LutMan_RO()==Qubit.instr_LutMan_RO()])
            if Qubit.name=='QSE' or Qubit.name=='QSW' or Qubit.name=='QC':
                # spectator_list.pop(spectator_list.index('QSW'))
                spectator_list = list([Qubit.name])

            if doSSRO:
              self.add_node('SSRO Calibration',
                        calibrate_function=Qubit.name + '.calibrate_ssro_fine',
                        calibrate_function_args={'check_threshold': 0.88, 'optimize_threshold': 0.95, 'nr_shots_per_case': SSROnumshots})

              self.add_node('Optimal Weights',
                        calibrate_function=self.device.name + '.calibrate_optimal_weights_mux',
                        calibrate_function_args={'qubits': spectator_list, 
                                                  'q_target': Qubit.name, 
                                                  'return_analysis': False,
                                                  'averages': 2 ** 19,   # this is the number of avgs to use for each transient
                                                  'update': True,
                                                  'verify': True,
                                                  'disable_metadata': True})

            self.add_node('T1',
                           calibrate_function = Qubit.name + '.measure_T1',
                           calibrate_function_args={'disable_metadata': True})

            #self.add_node(Qubit.name + ' T2_Star',
            #               calibrate_function = Qubit.name + '.measure_ramsey')
            
            self.add_node('T2e',
                           calibrate_function = Qubit.name + '.measure_echo',
                           calibrate_function_args={'disable_metadata': True})
      
            self.add_node('Frequency',
                          calibrate_function=Qubit.name + '.calibrate_frequency_ramsey',
                          calibrate_function_args={'steps':[1, 3, 10, 30],
                                                  'disable_metadata': True})
                          #check_function=Qubit.name + '.check_ramsey', tolerance=0.1e-3)
            
            self.add_node('Flipping',
                          calibrate_function=Qubit.name + '.flipping_GBT',
                          calibrate_function_args={'nr_sequence': 5,
                                                  'disable_metadata': True})
            
            self.add_node('Motzoi',
                          calibrate_function=Qubit.name + '.calibrate_motzoi',
                          calibrate_function_args={'motzois': np.linspace(start = -0.2, stop = 0.2, num = 21),
                                                  'disable_metadata': True})

            self.add_node('AllXY',
                          calibrate_function = Qubit.name + '.measure_allxy',
                          calibrate_function_args={'disable_metadata': True})
            
            self.add_node('RB',
                          calibrate_function=Qubit.name + '.measure_single_qubit_randomized_benchmarking',
                          calibrate_function_args={'recompile': RBrecompile, 'nr_seeds': RBnumseeds, 'nr_cliffords': 2 ** np.arange(10),
                                                  'disable_metadata': True})
            
            

            ####################
            # Node depdendencies
            ####################
            if doSSRO:
              self.add_edge('Optimal Weights',
                          'SSRO Calibration')

              self.add_edge('T1',
                          'Optimal Weights')
            
            self.add_edge('T2e',
                          'T1')
            
            self.add_edge('Frequency',
                          'T2e')
            
            self.add_edge('Flipping',
                          'Frequency')
            
            self.add_edge('Motzoi',
                          'Frequency')
            
            self.add_edge('AllXY',
                          'Flipping')

            self.add_edge('AllXY',
                          'Motzoi')
            
            self.add_edge('RB',
                          'AllXY')
            
            self.add_node('Prep Inspire',
                            calibrate_function=self.device.name + '.prepare_for_inspire')
            self.add_node('Upload Calibration Results',
                            calibrate_function='infinity.calibration.calibration')
            
            self.add_edge('Prep Inspire', 'RB')
            self.add_edge('Upload Calibration Results', 'Prep Inspire')
        

        self.cfg_plot_mode = 'svg'
        self.update_monitor()
        self.cfg_svg_filename

        url = self.open_html_viewer()
        print('Dependency graph 1Q created. URL = ' + url)

class inspire_dep_graph_2Q(AutoDepGraph_DAG):
    def __init__(self, 
                name: str, 
                device, 
                CZindex: int,   # index of the CZ gate, which can be 0,1,3,4. 
                epsTQP=10,      # two-qubit-phase error threshold for TQP node, in degrees.
                epsSQP=2,       # single-qubit phase error threshold for SQP nodes of pulsed and parked qubits, in degrees.
                epsSQPP=5,      # single-qubit phase error threshold for SQP node of parked qubit, in degrees
                RBnumseeds=50,
                RBrecompile=False,
                **kwargs):
        super().__init__(name, **kwargs)
        self.device = device

        self.create_dep_graph(CZindex=CZindex, 
                              epsTQP=epsTQP, 
                              epsSQP=epsSQP, 
                              epsSQPP=epsSQPP,
                              RBnumseeds=RBnumseeds,
                              RBrecompile=RBrecompile)
    
    def create_dep_graph(self, 
              CZindex: int,
              epsTQP,
              epsSQP,
              epsSQPP,
              RBnumseeds,
              RBrecompile):

        print('Creating dependency graph CZ ...')

   
        # convert from CZindex to qubit pairs. The third item, if any, is the parked qubit.
        # This conversion is specific to Quantum Inpire Starmon-5.

        if CZindex==0:
          pair=['NW', 'W', 'C']
          flux_lm_C = self.device.find_instrument('flux_lm_C')
          flux_lm_C.cfg_awg_channel_amplitude(0.32)
          self.device.prepare_for_timedomain(qubits = ['C'], bypass_flux = False)
        elif CZindex==1:
          pair=['NW', 'C', 'W']
        elif CZindex==2:
          pair=['NE', 'C', 'E']
        elif CZindex==3:
          pair=['NE', 'E', 'C']
          flux_lm_C = self.device.find_instrument('flux_lm_C')
          flux_lm_C.cfg_awg_channel_amplitude(0.392)
          self.device.prepare_for_timedomain(qubits = ['C'], bypass_flux = False)
        elif CZindex==4:
          pair=['W', 'SW']
        elif CZindex==5:
          pair=['C', 'SW', 'SE']
        elif CZindex==6:
          pair=['C', 'SE', 'SW']
        elif CZindex==7:
          pair=['E', 'SE']

        # reverse the first two elements in pair, and determine if there is a qubit to park
        reversedpair=[pair[1], pair[0]]
        parkingqubitexists=False
        if(len(pair)==3):
          reversedpair.append(pair[2])
          parkingqubitexists=True

        # get the cardinal direction.
        # here, cardinal is the CZ gate direction of the first element in pair, i.e., the qubit that will be Ramsey'd.
        # for example, QNW goes SE to 'meet' QC.
        # The following is specific to Quantum Inspire Starmon-5.
               
        cardinal = {str(['NW', 'W', 'C']):'SW',
                    str(['NW', 'C', 'W']):'SE',
                    str(['NE', 'C', 'E']):'SW',
                    str(['NE', 'E', 'C']):'SE',
                    str(['W', 'SW']):'SE',
                    str(['C', 'SW', 'SE']):'SW',
                    str(['C', 'SE', 'SW']):'SE',
                    str(['E', 'SE']):'SW'}
        
        #for diagnostics only
        #print(pair,cardinal[str(pair)])
        

        #################
        # Two-qubit Nodes
        #################

        # SNZ Calibration
        #----------------
        # get the flux lutman
        flux_lm = self.device.find_instrument(self.device.find_instrument(pair[0]).instr_LutMan_Flux())
        # get the 'center' corresponding to 11-02 avoided crossing.
        center = flux_lm.parameters['q_amp_center_{}'.format(cardinal[str(pair)])].get()
        #q_parks = [pair[2]] if len(pair)==3 else ['QNW'] if pair[0]=='QNE' else ['QNE']
        q_parks = [pair[2]] if len(pair)==3 else []
        if CZindex in [0, 1, 2, 3]:
          self.add_node('SNZ',
                            calibrate_function=self.device.name + '.measure_vcz_A_B_landscape',
                            calibrate_function_args={ 'Q0': [pair[0]], 'Q1': [pair[1]], 'update_flux_params': True,
                                                      'A_points': 11, 'A_ranges': [(.98*center, 1.02*center)], 
                                                      'B_amps': np.linspace(0, 1, 11),
                                                      'Q_parks': q_parks})
        else:
          self.add_node('SNZ',
                            calibrate_function=self.device.name + '.measure_vcz_A_B_landscape',
                            calibrate_function_args={ 'Q0': [pair[0]], 'Q1': [pair[1]], 'update_flux_params': True,
                                                      'A_points': 11, 'A_ranges': [(.995*center, 1.005*center)], 
                                                      'B_amps': np.linspace(0, 1, 11),
                                                      'Q_parks': q_parks})
        # Assess two-qubit phase
        self.add_node('TQP',
                        calibrate_function=self.device.name + '.measure_two_qubit_phase_GBT',
                        calibrate_function_args={'pair': pair, 'eps': epsTQP, 'updateSQP' : True,
                                                 'disable_metadata': True}
                        )
        # Calibrate single-qubit phase of pulsed qubit
        self.add_node('SQP Pulsed',
                        calibrate_function=self.device.name + '.calibrate_single_qubit_phase_GBT',
                        calibrate_function_args={'pair': pair, 'eps': epsSQP, 'disable_metadata': True}
                        )
        # Calibrate single-qubit phase of static qubit
        self.add_node('SQP Static',
                        calibrate_function=self.device.name + '.calibrate_single_qubit_phase_GBT',
                        calibrate_function_args={'pair': reversedpair, 'eps': epsSQP, 'disable_metadata': True}
                        )
        # Calibrate single-qubit phase of parked qubit, if it exists
        if(parkingqubitexists):
          ##### need to add specific routing for parked qubit. 
          ##### the one below does not do the job!!!!!
          self.add_node('SQP Parked',
                          calibrate_function=self.device.name + '.calibrate_parking_phase_GBT',
                          calibrate_function_args={'pair': reversedpair, 'eps': epsSQPP, 'disable_metadata': True}
                          )

        # perform CZ 2Q IRB
        self.add_node('IRB',
                      calibrate_function=self.device.name + '.measure_two_qubit_interleaved_randomized_benchmarking',
                      calibrate_function_args={'qubits':[pair[0],pair[1]], 
                            'nr_seeds': RBnumseeds, 
                            'MC':self.device.instr_MC.get_instr(), 
                            'recompile': RBrecompile, 
                            'measure_idle_flux': False, 
                            'nr_cliffords':np.array([0., 1., 3., 5., 7., 9., 11., 15., 20., 25., 30., 40., 50.]),
                            'cardinal': cardinal[str(pair)]})
            
        ###############
        # Device nodes
        ###############
        self.add_node('Prep Inspire',
                      calibrate_function=self.device.name + '.prepare_for_inspire')
        self.add_node('Upload Calibration Results',
                      calibrate_function='infinity.calibration.calibration')

        #####################
        # Define dependencies
        #####################

        self.add_edge('TQP',        # depends on
                        'SNZ')
            # self.add_edge('{}-{} Conditional Oscillation'.format(pair[0], pair[1]),
            #               '{}-{} Update Phase Correction'.format(pair[0], pair[1]))
            # self.add_edge('{}-{} TQIRB'.format(pair[0], pair[1]),
            #               '{}-{} Conditional Oscillation'.format(pair[0], pair[1]))
        self.add_edge('IRB', # depends on
                        'TQP')
        self.add_edge('IRB', # depends on
                        'SQP Pulsed')
        self.add_edge('IRB', # depends on
                        'SQP Static')
        if(parkingqubitexists):
          self.add_edge('IRB', # depends on
                        'SQP Parked')

        self.add_edge('Prep Inspire', # depends on
                        'IRB')
        self.add_edge('Upload Calibration Results', # depends on
                         'Prep Inspire')

        self.update_monitor()
        
        self.cfg_plot_mode = 'svg'
        self.cfg_svg_filename
        url = self.open_html_viewer()
        print('Dependency graph CZ created. URL = ' + url)

class inspire_dep_graph_T1T2(AutoDepGraph_DAG):
    def __init__(self, 
                  name: str, 
                  device,
                  **kwargs):
        super().__init__(name, **kwargs)
        self.device = device

        # make the list of qubit objects
        qubitobjects = []

        #for qubit in self.device.qubits():
        #    if qubit != 'fakequbit':
        #        qubitobjects.append(self.device.find_instrument(qubit))
        
        # I prefer to use this order. Change startup sequence later.
        qubitnames=['NW', 'W', 'NE', 'C', 'E', 'SW', 'SE']
        for qubitname in qubitnames:
            if qubitname != 'fakequbit':
                qubitobjects.append(self.device.find_instrument(qubitname))

        self.create_dep_graph(Qubit_list=qubitobjects)

    def create_dep_graph(self, Qubit_list):
      
      print('Creating dependency graph T1T2 ...')

      ####################
      # Single-qubit nodes
      ####################
      for Qubit in Qubit_list:
        self.add_node(Qubit.name+' T1',
                      calibrate_function = Qubit.name + '.measure_T1',
                      calibrate_function_args={'disable_metadata': True})

        #self.add_node(Qubit.name + ' T2_Star',
        #               calibrate_function = Qubit.name + '.measure_ramsey')
            
        self.add_node(Qubit.name+' T2e',
                      calibrate_function = Qubit.name + '.measure_echo',
                      calibrate_function_args={'disable_metadata': True})

        # and their dependencies
        self.add_edge(Qubit.name+' T2e',
                      Qubit.name+' T1')

      ##############
      # Device nodes
      ##############
      self.add_node('Prep Inspire',
                      calibrate_function=self.device.name + '.prepare_for_inspire')
      self.add_node('Upload Calibration Results',
                      calibrate_function='infinity.calibration.calibration')

      for Qubit in Qubit_list:
           self.add_edge('Prep Inspire',
                            Qubit.name+' T2e')

      self.add_edge('Upload Calibration Results', 
                          'Prep Inspire')

      # finishing up
      self.cfg_plot_mode = 'svg'
      self.update_monitor()
      self.cfg_svg_filename
      url = self.open_html_viewer()
      print('Dependency graph T1T2 created. URL = ' + url)


#### next-generation graphs with parallelization
class inspire_dep_graph_T1T2par(AutoDepGraph_DAG):
    def __init__(self, 
                  name: str, 
                  device,
                  **kwargs):
        super().__init__(name, **kwargs)
        self.device = device
        self.create_dep_graph()

    def create_dep_graph(self):
      
      print('Creating dependency graph T1T2par ...')


      # Add all nodes

      #NOTE: The groups must be arranged in a way that matches the feedline numbering.
      # Therefore, we sort the qubits according to [Feedline 1, Feedline 2, ...]
      # as an example, in group B qubit W is in Feedline 1, while qubits NE and SE are in Feedline 2,
      # therefore ['W', 'NE', 'SE']
      # 08/07/2024, MS

      qubit_group_A = {
          'name': 'NW_E_SW',
          'qubit_list': ['NW', 'E', 'SW']
      }
      qubit_group_B = {
          'name': 'W_NE_SE',
          'qubit_list': ['W', 'NE', 'SE']
      }
      qubit_group_C = {
          'name': 'C',
          'qubit_list': ['C']
      }

      qubit_groups = [qubit_group_A, qubit_group_B, qubit_group_C]

      for qubit_group in qubit_groups:
        numqubits=len(qubit_group['qubit_list'])
        timeslist=[np.linspace(0.0, 100.0,71)*1e-6]*numqubits
        self.add_node(f"T1_{qubit_group['name']}",  
                            calibrate_function = self.device.name + '.measure_multi_T1',
                            calibrate_function_args={ 'qubits': qubit_group['qubit_list'], 
                                                      'times' : timeslist, 
                                                      'update': True,
                                                      'disable_metadata': True})

        timeslist=[np.linspace(0.0, 60.0,61)*1e-6]*numqubits
        self.add_node(f"T2_{qubit_group['name']}",  
                            calibrate_function = self.device.name + '.measure_multi_Echo',
                            calibrate_function_args={ 'qubits': qubit_group['qubit_list'], 
                                                      'times' : timeslist, 
                                                      'update': True,
                                                      'disable_metadata': True})

      self.add_node('Prep Inspire',
                      calibrate_function=self.device.name + '.prepare_for_inspire')
      self.add_node('Upload Calibration Results',
                      calibrate_function='infinity.calibration.calibration')

      # Add all dependencies
      for qubit_group in qubit_groups:
        self.add_edge(f"T2_{qubit_group['name']}", f"T1_{qubit_group['name']}")
        self.add_edge('Prep Inspire', f"T2_{qubit_group['name']}")
      self.add_edge('Upload Calibration Results', 'Prep Inspire')

      # finishing up
      self.cfg_plot_mode = 'svg'
      self.update_monitor()
      self.cfg_svg_filename
      url = self.open_html_viewer()
      print('Dependency graph T1T2par created. URL = ' + url)

class inspire_dep_graph_QUICKpar(AutoDepGraph_DAG):
    def __init__(self, 
                  name: str, 
                  device,
                  epsFlipping=0.0005,
                  epsSQP=2, 
                  epsSQPP=3,
                  doSQPs=False,
                  **kwargs):
        super().__init__(name, **kwargs)
        self.device = device


        # doing this very manually for now
        CZindices=[0, 1, 2, 3, 4, 5, 6, 7]
        #CZindices=[1,3,4]; # used when QNW misbehaves.


        self.create_dep_graph(CZindices=CZindices,
                              epsFlipping=epsFlipping,
                              epsSQP=epsSQP,
                              epsSQPP=epsSQPP,
                              doSQPs=doSQPs)

    def create_dep_graph(self, 
                    CZindices,
                    epsFlipping,
                    epsSQP,
                    epsSQPP,
                    doSQPs):
      
      print('Creating dependency graph QUICKpar ...')

      #NOTE: The groups must be arranged in a way that matches the feedline numbering.
      # Therefore, we sort the qubits according to [Feedline 1, Feedline 2, ...]
      # as an example, in group B qubit W is in Feedline 1, while qubits NE and SE are in Feedline 2,
      # therefore ['W', 'NE', 'SE']
      # 08/07/2024, MS


      qubit_group_A = {
          'name': 'NW_E_SW',
          'qubit_list': ['NW', 'E', 'SW']
      }
      qubit_group_B = {
          'name': 'W_NE_SE',
          'qubit_list': ['W', 'NE', 'SE']
      }
      qubit_group_C = {
          'name': 'C',
          'qubit_list': ['C']
      }

      qubit_groups = [qubit_group_A, qubit_group_B, qubit_group_C]

      for qubit_group in qubit_groups:
        self.add_node(f"Flipping_{qubit_group['name']}",
                            calibrate_function=self.device.name + '.multi_flipping_GBT',
                            calibrate_function_args={'qubits': qubit_group['qubit_list'], 
                                                      'number_of_flips': np.arange(0, 31, 2), 
                                                      'eps': epsFlipping,
                                                      'disable_metadata': True})

      allCZpairs = [
         ['NW', 'W', 'C'],
         ['NW', 'C', 'W'],
         ['NE', 'C', 'E'],
         ['NE', 'E', 'C'],
         ['W', 'SW'],
         ['C', 'SW', 'SE'],
         ['C', 'SE', 'SW'],
         ['E', 'SE']
      ]

      allCZnames = [
         'CZ_NW_W',
         'CZ_NW_C',
         'CZ_NE_C',
         'CZ_NE_E',
         'CZ_W_SW',
         'CZ_C_SW',
         'CZ_C_SE',
         'CZ_E_SE'
      ]

      # allCZpairs=[['QNW','QC'], ['QNE','QC'], [], ['QC','QSW','QSE'], ['QC','QSE','QSW']]
      # allCZnames=['CZNW', 'CZNE', 'CZnone', 'CZSW', 'CZSE']

      #################
      # Two-qubit nodes
      #################
      for CZindex in CZindices:

        CZname=allCZnames[CZindex]
        pair=allCZpairs[CZindex]
        # reverse the first two elements in pair, and determine if there is a qubit to park
        reversedpair=[pair[1], pair[0]]
        parkingqubitexists=False
        if(len(pair)==3):
          reversedpair.append(pair[2])
          parkingqubitexists=True

        if CZindex == 0:
          flux_lm_C = self.device.find_instrument('flux_lm_C')
          flux_lm_C.cfg_awg_channel_amplitude(0.32)
          self.device.prepare_for_timedomain(qubits = ['C'], bypass_flux = False)
        if CZindex == 3:
          flux_lm_C = self.device.find_instrument('flux_lm_C')
          flux_lm_C.cfg_awg_channel_amplitude(0.392)
          self.device.prepare_for_timedomain(qubits = ['C'], bypass_flux = False)  
        
        # Calibrate single-qubit phase of pulsed qubit
        self.add_node(CZname+' SQP Pulsed',
                        calibrate_function=self.device.name + '.calibrate_single_qubit_phase_GBT',
                        calibrate_function_args={'pair': pair, 'eps': epsSQP, 'disable_metadata': True}
                        )
        # Calibrate single-qubit phase of pulsed qubit
        if(doSQPs):
          self.add_node(CZname + ' SQP Static',
                        calibrate_function=self.device.name + '.calibrate_single_qubit_phase_GBT',
                        calibrate_function_args={'pair': reversedpair, 'eps': epsSQP, 'disable_metadata': True}
                        )
        # Calibrate single-qubit phase of parked qubit, if it exists
        if(parkingqubitexists):
          ##### need to add specific routing for parked qubit. 
          ##### the one below does not do the job!!!!!
          self.add_node(CZname +' SQP Parked',
                          calibrate_function=self.device.name + '.calibrate_parking_phase_GBT',
                          calibrate_function_args={'pair': reversedpair, 'eps': epsSQPP, 'disable_metadata': True}
                          )

      ##############
      # Device Nodes
      ##############
      self.add_node('Prep Inspire',
                      calibrate_function=self.device.name + '.prepare_for_inspire')
      self.add_node('Upload Calibration Results',
                      calibrate_function='infinity.calibration.calibration')
      
      ###################
      # Node dependencies
      ###################

      for CZindex in CZindices:
        CZname=allCZnames[CZindex]
        pair=allCZpairs[CZindex]
        # reverse the first two elements in pair, and determine if there is a qubit to park
        reversedpair=[pair[1], pair[0]]
        parkingqubitexists=False
        if(len(pair)==3):
          reversedpair.append(pair[2])
          parkingqubitexists=True

        self.add_edge('Prep Inspire',
                        CZname + ' SQP Pulsed')
        
        for qubit_group in qubit_groups:
          if pair[0] in qubit_group:

            self.add_edge(CZname + ' SQP Pulsed',
                            f"Flipping_{qubit_group['name']}")

        if(doSQPs):
          self.add_edge('Prep Inspire',
                        CZname + ' SQP Static')

        for qubit_group in qubit_groups:
          if pair[1] in qubit_group:
            self.add_edge(CZname + ' SQP Static',
                          f"Flipping_{qubit_group['name']}")

        if(parkingqubitexists):
          ##### need to add specific routing for parked qubit. 
          ##### the one below does not do the job!!!!!
          self.add_edge('Prep Inspire',
                        CZname + ' SQP Parked')

          for qubit_group in qubit_groups:
            if pair[2] in qubit_group:
              self.add_edge(CZname + ' SQP Parked',
                            f"Flipping_{qubit_group['name']}")

      self.add_edge('Upload Calibration Results', 'Prep Inspire')

      # finishing up
      self.cfg_plot_mode = 'svg'
      self.update_monitor()
      self.cfg_svg_filename
      url = self.open_html_viewer()
      print('Dependency graph QUICKpar created. URL = ' + url)







class inspire_dep_graph_1Qpar(AutoDepGraph_DAG):
  def __init__(self, 
                name: str, 
                device,
                RBnumseeds = 50,
                RBrecompile = True,
                epsFlipping = 0.0005,
                **kwargs):
      super().__init__(name, **kwargs)
      self.device = device
      self.create_dep_graph(RBnumseeds = RBnumseeds,
                            RBrecompile = RBrecompile,
                            epsFlipping = epsFlipping)

  def create_dep_graph(self,
                       RBnumseeds,
                       RBrecompile,
                       epsFlipping):
    
    print('Creating dependency graph 1Qpar ...')


    # Add all nodes

    #NOTE: The groups must be arranged in a way that matches the feedline numbering.
    # Therefore, we sort the qubits according to [Feedline 1, Feedline 2, ...]
    # as an example, in group B qubit W is in Feedline 1, while qubits NE and SE are in Feedline 2,
    # therefore ['W', 'NE', 'SE']
    # 08/07/2024, MS

    qubit_group_A = {
        'name': 'NW_E_SW',
        'qubit_list': ['NW', 'E', 'SW']
    }
    qubit_group_B = {
        'name': 'W_NE_SE',
        'qubit_list': ['W', 'NE', 'SE']
    }
    qubit_group_C = {
        'name': 'C',
        'qubit_list': ['C']
    }

    qubit_groups = [qubit_group_A, qubit_group_B, qubit_group_C]

    for qubit_group in qubit_groups:
      numqubits=len(qubit_group['qubit_list'])
      timeslist=[np.linspace(0.0, 100.0,71)*1e-6]*numqubits
      self.add_node(f"T1_{qubit_group['name']}",  
                          calibrate_function = self.device.name + '.measure_multi_T1',
                          calibrate_function_args={ 'qubits': qubit_group['qubit_list'], 
                                                    'times' : timeslist, 
                                                    'update': True,
                                                    'disable_metadata': True})

      timeslist=[np.linspace(0.0, 60.0,61)*1e-6]*numqubits
      self.add_node(f"T2e_{qubit_group['name']}",  
                          calibrate_function = self.device.name + '.measure_multi_Echo',
                          calibrate_function_args={ 'qubits': qubit_group['qubit_list'], 
                                                    'times' : timeslist, 
                                                    'update': True,
                                                    'disable_metadata': True})
      timeslist=[np.linspace(0.0, 20.0,61)*1e-6]*numqubits
      self.add_node(f"Frequency_{qubit_group['name']}",
                    calibrate_function = self.device.name + '.calibrate_multi_frequency_fine',
                    calibrate_function_args={ 'qubits': qubit_group['qubit_list'],
                                              'times': timeslist,
                                              'artificial_periods': 2.5,
                                              'update_T2': True,
                                              'update_frequency': True,
                                              'stepsize': 20e-9,
                                              'steps': [1, 3, 10],
                                              'disable_metadata': True})
      self.add_node(f"Flipping_{qubit_group['name']}",
                    calibrate_function = self.device.name + '.multi_flipping_GBT',
                    calibrate_function_args={ 'qubits': qubit_group['qubit_list'],
                                              'eps': epsFlipping,
                                              'disable_metadata': True})
      self.add_node(f"Motzoi_{qubit_group['name']}",
                    calibrate_function = self.device.name + '.measure_multi_motzoi',
                    calibrate_function_args={ 'qubits': qubit_group['qubit_list'],
                                              'amps': np.linspace(start = -0.2, stop = 0.2, num = 21),
                                              'disable_metadata': True})
      self.add_node(f"AllXY_{qubit_group['name']}",
                    calibrate_function = self.device.name + '.measure_multi_AllXY',
                    calibrate_function_args={ 'qubits': qubit_group['qubit_list'],
                                              'disable_metadata': True})
      
    qubit_RB_group_A = {
        'name': 'NW_E',
        'qubit_list': ['NW', 'E']
    }
    qubit_RB_group_B = {
        'name': 'NE_SW',
        'qubit_list': ['NE', 'SW']
    }
    qubit_RB_group_C = {
        'name': 'W_SE',
        'qubit_list': ['W', 'SE']
    }
    qubit_RB_group_D = {
        'name': 'C',
        'qubit_list': ['C']
    }

    qubit_RB_groups = [qubit_RB_group_A, qubit_RB_group_B, qubit_RB_group_C]

    for qubit_RB_group in qubit_RB_groups:
      self.add_node(f"2QRB_{qubit_RB_group['name']}",
                  calibrate_function = self.device.name + '.measure_two_qubit_simultaneous_randomized_benchmarking',
                  calibrate_function_args={ 'qubits': qubit_RB_group['qubit_list'],
                                            'recompile': RBrecompile,
                                            'nr_seeds': RBnumseeds,
                                            'nr_cliffords': 2**np.arange(10),
                                            'disable_metadata': True})
    self.add_node(f"1QRB_{qubit_RB_group_D['name']}",
                        calibrate_function=f"{qubit_RB_group_D['name']}" + '.measure_single_qubit_randomized_benchmarking',
                        calibrate_function_args={'recompile': RBrecompile,
                                                 'nr_seeds': RBnumseeds,
                                                 'nr_cliffords': 2 ** np.arange(10),
                                                'disable_metadata': True})

    self.add_node('Prep Inspire',
                    calibrate_function=self.device.name + '.prepare_for_inspire')
    self.add_node('Upload Calibration Results',
                    calibrate_function='infinity.calibration.calibration')

    # Add all dependencies
    for qubit_group in qubit_groups:
      self.add_edge(f"T2e_{qubit_group['name']}", f"T1_{qubit_group['name']}")
      self.add_edge(f"Frequency_{qubit_group['name']}", f"T2e_{qubit_group['name']}")
      self.add_edge(f"Flipping_{qubit_group['name']}", f"Frequency_{qubit_group['name']}")
      self.add_edge(f"Motzoi_{qubit_group['name']}", f"Frequency_{qubit_group['name']}")
      self.add_edge(f"AllXY_{qubit_group['name']}", f"Flipping_{qubit_group['name']}")
      self.add_edge(f"AllXY_{qubit_group['name']}", f"Motzoi_{qubit_group['name']}")

    self.add_edge(f"2QRB_{qubit_RB_group_A['name']}", f"AllXY_{qubit_group_A['name']}")
    self.add_edge(f"2QRB_{qubit_RB_group_B['name']}", f"AllXY_{qubit_group_A['name']}")
    self.add_edge(f"2QRB_{qubit_RB_group_B['name']}", f"AllXY_{qubit_group_B['name']}")
    self.add_edge(f"2QRB_{qubit_RB_group_C['name']}", f"AllXY_{qubit_group_B['name']}")
    self.add_edge(f"1QRB_{qubit_RB_group_D['name']}", f"AllXY_{qubit_group_C['name']}")

    for qubit_RB_group in qubit_RB_groups:
      self.add_edge('Prep Inspire', f"2QRB_{qubit_RB_group['name']}")
    self.add_edge('Prep Inspire', f"1QRB_{qubit_RB_group_D['name']}")


    self.add_edge('Upload Calibration Results', 'Prep Inspire')

    # finishing up
    self.cfg_plot_mode = 'svg'
    self.update_monitor()
    self.cfg_svg_filename
    url = self.open_html_viewer()
    print('Dependency graph 1Qpar created. URL = ' + url)      