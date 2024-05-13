
from pylatex import Document, Section, Subsection, Command, Figure, Tabular, Itemize
from pylatex.utils import italic, NoEscape, bold
from icecream import ic
import os
from netsquid.util.datacollector import DataCollector
from netsquid.protocols import Signals
import pydynaa
from matplotlib import pyplot as plt


def generate_report(report_info, simulation_data, mode):
    '''
    Generates latex/pdf report
    Input:
     - report_info: Dictionary. Key will be the used value for each iteration. If Fidex mode
     was selected, value will be 0.
     - simulation_data: Dictionary. Key is the request name and value is a dataframe with a row for 
     each value corresponding to the different simulations.
     - mode: 'F' for Fixed and 'E' for Evolution. In the first case will only print values, but
     if evolution mode is selected will include graphs.
    '''
    report = Document('./output/report')
    report.preamble.append(Command('title', 'Simulation report'))
    report.preamble.append(Command('author','UVigo'))
    report.preamble.append(Command('date', NoEscape(r'\today')))
    report.append(NoEscape(r'\maketitle'))

    #Routing Phase section
    with report.create(Section('Routing Protocol')):
        report.append('The next section details information gathered through the route calculation phase\n')

        with report.create(Subsection('Network')):
            with report.create(Figure(position='ht!')) as fig_network:
                image_file = os.path.join(os.path.dirname(__file__), 'output/graf.png')
                fig_network.add_image(image_file,width='180px')
                fig_network.add_caption('Simulated network')

        with report.create(Subsection('Link fidelities')):
            report.append('Fidelity and Cost of each of the links\n')
            if mode == 'F':
                with report.create(Tabular('l|l|l|l')) as table:
                    table.add_hline()
                    table.add_row(('link','cost','fidelity','number of rounds'))
                    table.add_hline()
                    for key, value in report_info[0]['link_fidelities'].items():
                        table.add_row((key,value[0],value[1],value[2]))
        with report.create(Subsection('Path simulation')):  
            if mode == 'F':              
                #report.append('Request fulfillment\n')
                with report.create(Tabular('l|l|l|l|l')) as table:      
                    table.add_hline()
                    table.add_row(['request','result','fidelity','time','purification rounds'])
                    table.add_hline()
                    for reg in report_info[0]['requests_status']:
                        table.add_row(bold(reg['request']),reg['result'],reg['fidelity'],reg['time'],reg['purif_rounds'])
                    table.add_hline()
                report.append('\n')
                report.append('\n')
                with report.create(Tabular('l|p{3.75in}')) as table:
                    table.add_hline()
                    table.add_row('request','shortest path')
                    table.add_hline()
                    for reg in report_info[0]['requests_status']:
                        table.add_row(bold(reg['request']),reg['shortest_path'])

    with report.create(Section('Fase de simulación')):
        report.append('Esta parte está pendiente')

    report.generate_pdf('./output/report',clean_tex=False)
    report.generate_tex()

def dc_setup(protocol):
        '''
        Creates a data collector in order to measure fidelity of E2E entanglement
        Inputs:
            - 
        Outputs:
            - dc: instance of the configured datacollector
        '''
        def record_stats(evexpr):
            #Record an execution
            protocol = evexpr.triggered_events[-1].source
            result = protocol.get_signal_result(Signals.SUCCESS)

            return(result)


        dc = DataCollector(record_stats, include_time_stamp=False, include_entity_name=False)
        dc.collect_on(pydynaa.EventExpression(source = protocol, event_type=Signals.SUCCESS.value))
        return(dc)

def validate_conf(config):
        '''
        Performs validations on the provided configuration file
        Input: 
                - config: Dictionary with parsed yaml file
        Output: -
        '''
   
        #Verify that global mandatory parameters exist
        if 'name' not in config.keys() or 'link_fidel_rounds' not in config.keys() \
            or 'path_fidel_rounds' not in config.keys() or 'nodes' not in config.keys() \
                or 'links' not in config.keys() or 'requests' not in config.keys() \
                    or 'epr_pair' not in config.keys() or 'simulation_duration' not in config.keys(): 
            raise ValueError('Invalid configuration file, check global parameters')

        #Check link sintax
        links = config['links']
            
        #names must be unique
        linknames = [list(link.keys())[0] for link in links]
        set_names = set(linknames)
        if len(set_names) != len(linknames): #there are repeated node names:
            raise ValueError('Invalid configuration file, repeated link names')

        #Valid types
        available_props = {'end1':'string',
                               'end2':'string',
                               'distance':'float',
                               'number_links':'integer',
                               'source_fidelity_sq':'float01',
                               'source_delay':'integer',
                               'photon_speed_fibre':'float',
                               'qchannel_noise_model':'string',
                               'p_depol_init':'float01',
                               'p_depol_length':'float01',
                               'dephase_qchannel_rate':'float',
                               'depolar_qchannel_rate':'float',
                               'p_loss_init':'float01',
                               'p_loss_length':'float01',
                               't1_qchannel_time':'float',
                               't2_qchannel_time':'float',
                               'classical_delay_model':'string',
                               'gaussian_delay_mean':'integer',
                               'gaussian_delay_std':'integer',
                               'qchannel_loss_model':'string'}
        
        #get list of nde names
        nodenames = [list(node.keys())[0] for node in config['nodes']]
        #Get name and type of nodes
        nodenames = []
        #Get nodes type for later usage in link validation section
        #Also get teleport queue memory options, if applicable for usage in request validation section
        nodetypes = {}
        node_tx_memories = {}
        for node in config['nodes']:
            nodename = list(node.keys())[0]
            nodenames.append(nodename)
            nodeprops = list(node.values())[0]
            if 'type' in nodeprops.keys():
                nodetypes[nodename] = nodeprops['type']
                if nodeprops['type'] == 'endNode':
                    tel_q_size = 'not_set' if 'teleport_queue_size' not in nodeprops.keys() else nodeprops['teleport_queue_size']
                    tel_q_tech = 'not_set' if 'teleport_queue_technology' not in nodeprops.keys() else nodeprops['teleport_queue_technology']
                    tel_strat= 'not_set' if 'teleport_strategy' not in nodeprops.keys() else nodeprops['teleport_strategy']
                    node_tx_memories[nodename] = [tel_q_size, tel_q_tech, tel_strat]
            else:
                raise ValueError(f"Node {nodename}: No type specified")

        for link in links:
            link_props = list(link.values())[0]
            link_name = list(link.keys())[0]
            #link names cannot contain hyphens or underscore
            if link_name.find('-') != -1: raise ValueError (f'{link_name}: Link names cannot contain hyphens')       
            if link_name.find('_') != -1: raise ValueError (f'{link_name}: Link names cannot contain underscore')       
            
            #Check that nodes are valid
            if link_props['end1'] not in nodenames:
                raise ValueError(f"link {link_name}: node {link_props['end1']} not defined")
            if link_props['end2'] not in nodenames:
                raise ValueError(f"link {link_name}: node {link_props['end2']} not defined")
            
            #Check that defined properties are valid    
            for prop in link_props.keys():
                #Check that property is valid
                if prop not in available_props.keys():
                    raise ValueError(f'Property {prop} in link {link_name} is not valid')
                if available_props[prop] == 'integer':
                    if not isinstance(link_props[prop],int):
                        raise ValueError(f"link {link_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                    elif link_props[prop]<0:
                        raise ValueError(f"link {link_name} {prop} cannot be negative")
                elif available_props[prop] == 'string':
                    if not isinstance(link_props[prop],str):
                        raise ValueError(f"link {link_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                elif available_props[prop] == 'float':
                    try:
                        val = float(link_props[prop])
                        if val < 0:
                            raise ValueError(f"link {link_name} {prop} cannot be negative")
                    except:
                        raise ValueError(f"link {link_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                elif available_props[prop] == 'float01':
                    try:
                        val = float(link_props[prop])
                        if val < 0 or val > 1:
                            raise ValueError(f"link {link_name} {prop} must be between 0 and 1")
                    except:
                        raise ValueError(f"link {link_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                else:
                    raise ValueError(f"link {link_name} incorrect type for {prop}, it is {type(prop)}")
            
            #Check for definition of mandatory properties
            mandatory = ['end1','end2','distance','source_fidelity_sq','photon_speed_fibre']
            for prop in mandatory:
                if prop not in link_props.keys(): 
                    raise ValueError(f"link {link_name}: missing property {prop}")
       
            #number_links can only be specified between switches
            if (nodetypes[link_props['end1']] == 'endNode' or \
                nodetypes[link_props['end2']] == 'endNode') and \
                'number_links' in link_props.keys() and link_props['number_links'] != 2:
                raise ValueError(f"{link_name}: number_links can only be 2 between node and switch")

            #Check allowed values of noise model
            allowed_qchannel_noise_model = ['DephaseNoiseModel','DepolarNoiseModel','T1T2NoiseModel','FibreDepolarizeModel','FibreDepolGaussModel','None']
            if 'qchannel_noise_model' in link_props.keys() \
                and link_props['qchannel_noise_model'] not in allowed_qchannel_noise_model:
                raise ValueError(f"link {link_name}: Unsupported quantum channel noise model")

            #If quantum channel noise model is FibreDepolarizeModel p_depol_init and p_depol_length must be declared
            if 'qchannel_noise_model' in link_props.keys() and  \
                link_props['qchannel_noise_model'] == 'FibreDepolarizeModel'  \
                and ('p_depol_init' not in link_props.keys() or 'p_depol_length' not in link_props.keys()):
                raise ValueError(f"link {link_name}: When FibreDepolarizeModel is selected for quantum channel, p_depol_init and p_depol_length must be defined")
            
            #If quantum channel noise model is DephaseNoiseModel dephase_qchannel_rate must be declared
            if 'qchannel_noise_model' in link_props.keys() and  \
                link_props['qchannel_noise_model'] == 'DephaseNoiseModel'  \
                'dephase_qchannel_rate' not in link_props.keys():
                raise ValueError(f"link {link_name}: When DephaseNoiseModel is selected for quantum channel, dephase_qchannel_rate must be defined")
            
            #If quantum channel noise model is DepolarNoiseModel depolar_qchannel_rate must be declared
            if 'qchannel_noise_model' in link_props.keys() and  \
                link_props['qchannel_noise_model'] == 'DepolarNoiseModel'  \
                'depolar_qchannel_rate' not in link_props.keys():
                raise ValueError(f"link {link_name}: When DepolarNoiseModel is selected for quantum channel, dephase_qchannel_rate must be defined")

            #If quantum channel noise model is T1T2NoiseModel t1 & t2 times must be declared
            if 'qchannel_noise_model' in link_props.keys() and  \
                link_props['qchannel_noise_model'] == 'T1T2NoiseModel'  \
                and ('t1_qchannel_time' not in link_props.keys() or 't2_qchannel_time' not in link_props.keys()):
                raise ValueError(f"link {link_name}: When T1T2NoiseModel is selected for quantum channel, t1_qchannel_time and t2_qhannel_time must be defined")
    
            #Check allowed values of loss model
            allowed_qchannel_loss_model = ['FibreLossModel','None']
            if 'qchannel_loss_model' in link_props.keys() \
                and link_props['qchannel_loss_model'] not in allowed_qchannel_loss_model:
                raise ValueError(f"link {link_name}: Unsupported quantum channel loss model")
            
            #If quantum channel noise model is FibreLosseModel p_loss_init and p_losslength must be declared
            if 'qchannel_loss_model' in link_props.keys() and  \
                link_props['qchannel_loss_model'] == 'FibreLossModel'  \
                and ('p_loss_init' not in link_props.keys() or 'p_loss_length' not in link_props.keys()):
                raise ValueError(f"link {link_name}: When FibreLossModel is selected for quantum channel, p_loss_init and p_loss_length must be defined")
            
            #Check allowed values of classical channel models
            allowed_classical_model = ['FibreDelayModel','GaussianDelayModel']
            if 'classical_delay_model' in link_props.keys() \
                and link_props['classical_delay_model'] not in allowed_classical_model:
                raise ValueError(f"link {link_name}: Unsupported classical channel delay model")

            #If quantum channel noise model is T1T2NoiseModel t1 & t2 times must be declared
            if 'classical_delay_model' in link_props.keys() and  \
                link_props['classical_delay_model'] == 'GaussianDelayModel'  \
                and ('gaussian_delay_mean' not in link_props.keys() or 'gaussian_delay_std' not in link_props.keys()):
                raise ValueError(f"link {link_name}: When GaussianDelayModel is selected for qclassical channel, gaussian_delay_mean and gaussian_delay_std must be defined")
        

        #Check node sintax
        #No node names are repeated
        set_names = set(nodenames)
        if len(set_names) != len(nodenames): #there are repeated node names:
            raise ValueError('Invalid configuration file, repeated node names')
        
        #Valid types
        available_props = {'type':'string',
                               'num_memories':'integer',
                               'gate_duration':'integer',
                               'gate_duration_X':'integer',
                               'gate_duration_Z':'integer',
                               'gate_duration_CX':'integer',
                               'gate_duration_rotations':'integer',
                               'measurements_duration':'integer',
                               'gate_noise_model':'string',
                               'dephase_gate_rate':'integer',
                               'depolar_gate_rate':'integer',
                               't1_gate_time':'integer',
                               't2_gate_time':'integer',
                               'mem_noise_model':'string',
                               'dephase_mem_rate':'integer',
                               'depolar_mem_rate':'integer',
                               't1_mem_time':'float',
                               't2_mem_time':'float',
                               'teleport_queue_size':'integer',
                               'teleport_queue_technology':'string',
                               'teleport_strategy':'string'}
        
        for node in config['nodes']:
            node_props = list(node.values())[0]
            node_name = list(node.keys())[0]

            #nodenames cannot contain underscore
            if node_name.find('_') != -1: raise ValueError (f'{node_name}: Node names cannot contain underscore')
            
            #Check that defined properties are valid    
            for prop in node_props.keys():
                #Check that property is valid
                if prop not in available_props.keys():
                    raise ValueError(f'Property {prop} in node {node_name} is not valid')
                if available_props[prop] == 'integer':
                    if not isinstance(node_props[prop],int):
                        raise ValueError(f"node {node_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                    elif node_props[prop]<0:
                        raise ValueError(f"node {node_name} {prop} cannot be negative")
                elif available_props[prop] == 'string':
                    if not isinstance(node_props[prop],str):
                        raise ValueError(f"node {node_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                elif available_props[prop] == 'float':
                    try:
                        val = float(node_props[prop])
                        if val < 0:
                            raise ValueError(f"node {node_name} {prop} cannot be negative")
                    except:
                        raise ValueError(f"node {node_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                else:
                    raise ValueError(f"node {node_name} incorrect type for {prop}, it is {type(prop)}")
            
            #Check for definition of mandatory properties
            mandatory = ['type']
            for prop in mandatory:
                if prop not in node_props.keys(): 
                    raise ValueError(f"node {node_name}: missing property {prop}")

            #Only two types are allowed: switch and endNode
            if node_props['type'] not in ['switch','endNode']:
                raise ValueError(f'node {node_name} type can only be switch or endNode')

            #If node is a switch we must define the number of  available memories
            if node_props['type'] == 'switch' and 'num_memories' not in node_props.keys():
                raise ValueError(f"node {node_name}: num_memories must be declared")
            
            #Check allowed values of noise model
            allowed_gate_noise_model = ['DephaseNoiseModel','DepolarNoiseModel','T1T2NoiseModel']
            allowed_mem_noise_model = ['DephaseNoiseModel','DepolarNoiseModel','T1T2NoiseModel']
            if 'gate_noise_model' in node_props.keys() \
                and node_props['gate_noise_model'] not in allowed_gate_noise_model:
                raise ValueError(f"node {node_name}: Unsupported gate noise model")
            if 'mem_noise_model' in node_props.keys() \
                and node_props['mem_noise_model'] not in allowed_mem_noise_model:
                raise ValueError(f"node {node_name}: Unsupported memory noise model")

            #If gate noise model is DephaseNoiseModel the rate must be declared
            if 'gate_noise_model' in node_props.keys() and  \
                node_props['gate_noise_model'] == 'DephaseNoiseModel'  \
                and 'dephase_gate_rate' not in node_props.keys():
                raise ValueError(f"node {node_name}: When DephaseNoiseModel is selected for gate, dephase_gate_rate must be defined")

            #When gate noise model is DepolarNoiseModel the rate must be declared
            if 'gate_noise_model' in node_props.keys() and  \
                node_props['gate_noise_model'] == 'DepolarNoiseModel'  \
                and 'depolar_gate_rate' not in node_props.keys():
                raise ValueError(f"node {node_name}: When DepolarNoiseModel is selected for gate, depolar_gate_rate must be defined")    
                
            #If gate noise model is T1T2NoiseModel t1 & t2 times must be declared
            if 'gate_noise_model' in node_props.keys() and  \
                node_props['gate_noise_model'] == 'T1T2NoiseModel'  \
                and ('t1_gate_time' not in node_props.keys() or 't2_gate_time' not in node_props.keys()):
                raise ValueError(f"node {node_name}: When T1T2NoiseModel is selected for gate, t1_gate_time and t2_gate_time must be defined")

            #If memory noise model is DephaseNoiseModel the rate must be declared
            if 'mem_noise_model' in node_props.keys() and  \
                node_props['mem_noise_model'] == 'DephaseNoiseModel'  \
                and 'dephase_mem_rate' not in node_props.keys():
                raise ValueError(f"node {node_name}: When DephaseNoiseModel is selected for memory, dephase_mem_rate must be defined")

            #When memory noise model is DepolarNoiseModel the rate must be declared
            if 'mem_noise_model' in node_props.keys() and  \
                node_props['mem_noise_model'] == 'DepolarNoiseModel'  \
                and 'depolar_mem_rate' not in node_props.keys():
                raise ValueError(f"node {node_name}: When DepolarNoiseModel is selected for memory, depolar_mem_rate must be defined")    
                
            #If gate noise model is T1T2NoiseModel t1 & t2 times must be declared
            if 'mem_noise_model' in node_props.keys() and  \
                node_props['mem_noise_model'] == 'T1T2NoiseModel'  \
                and ('t1_mem_time' not in node_props.keys() or 't2_mem_time' not in node_props.keys()):
                raise ValueError(f"node {node_name}: When T1T2NoiseModel is selected for memory, t1_mem_time and t2_mem_time must be defined")

            #Check allowed values of teleport memory options
            allowed_teleport_technologies = ['Quantum','Classical']
            allowed_teleport_strategies = ['Oldest','Newest']
            if 'teleport_queue_technology' in node_props.keys() \
                and node_props['teleport_queue_technology'] not in allowed_teleport_technologies:
                raise ValueError(f"node {node_name}: Unsupported teleport queue technology")
            if 'teleport_strategy' in node_props.keys() \
                and node_props['teleport_strategy'] not in allowed_teleport_strategies:
                raise ValueError(f"node {node_name}: Unsupported teleportation strategy")

            #Check that in switch nodes we have > 2*num_links
            if node_props['type'] == 'endNode' and 'num_memories' in node_props.keys() and \
                node_props['num_memories'] != 4:
                raise ValueError(f"node {node_name}: if num_memories declared in endNode, must aways be 2")
            elif node_props['type'] == 'switch':
                #must check than number of memories is greater than connected links
                total_links = 0
                for link in config['links']:
                    link_props = list(link.values())[0]
                    if link_props['end1'] == node_name or link_props['end2'] == node_name:
                        total_links += link_props['number_links'] \
                            if 'number_links' in link_props.keys() else 2
                if total_links > node_props['num_memories']:
                    raise ValueError(f"node {node_name}: not enough memories. Need at least {total_links}")

        #Check requests sintax
        requests = config['requests']
        requestnames = [list(request.keys())[0] for request in requests]
        set_names = set(requestnames)
        if len(set_names) != len(requestnames): #there are repeated request names:
            raise ValueError('Invalid configuration file, repeated request names')
        
        #Check valid properties
        available_props = {'origin': 'string',
            'destination': 'string',
            'minfidelity': 'float01',
            'maxtime': 'integer',
            'path_fidel_rounds': 'integer',
            'application': 'string',
            'teleport': 'list',
            'qber_states': 'list',
            'demand_rate': 'float'}
        
        #Check if a node is in more than one request
        #No need to do so, if this happens, the second request will indicate that no resources are available

        for request in requests:
            request_props = list(request.values())[0]
            request_name = list(request.keys())[0]

            #request names cannot contain underscore
            if request_name.find('_') != -1: raise ValueError (f'{request_name}: Request names cannot contain underscore')
            

            #Check that nodes are valid
            if request_props['origin'] not in nodenames:
                raise ValueError(f"request {request_name}: node {request_props['origin']} not defined")
            if request_props['destination'] not in nodenames:
                raise ValueError(f"request {request_name}: node {request_props['destination']} not defined")
            
            #Check that defined properties are valid    
            for prop in request_props.keys():
                #Check that property is valid
                if prop not in available_props.keys():
                    raise ValueError(f'Property {prop} in request {request_name} is not valid')
                if available_props[prop] == 'integer':
                    if not isinstance(request_props[prop],int):
                        raise ValueError(f"request {request_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                elif available_props[prop] == 'string':
                    if not isinstance(request_props[prop],str):
                        raise ValueError(f"request {request_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                elif available_props[prop] == 'float':
                    try:
                        val = float(request_props[prop])
                        if val < 0:
                            raise ValueError(f"request {request_name} {prop} cannot be negative")
                    except:
                        raise ValueError(f"request {request_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                elif available_props[prop] == 'float01':
                    try:
                        val = float(request_props[prop])
                        if val > 1 or val<0:
                            raise ValueError(f"request {request_name} {prop} must be between 0 and 1")
                    except:
                        raise ValueError(f"request {request_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                elif available_props[prop] == 'list':
                        if not isinstance(request_props[prop],list):
                            raise ValueError(f"request {request_name} {prop} must be of {available_props[prop]} type but is {type(prop)}")
                else:
                    raise ValueError(f"request {request_name}: incorrect type for {prop}, it is {type(prop)}")
            
            #Check for definition of mandatory properties
            mandatory = ['origin','destination','minfidelity','maxtime','application']
            for prop in mandatory:
                if prop not in request_props.keys(): 
                    raise ValueError(f"request {request_name}: missing property {prop}")
            
            #Check for valid applications
            if request_props['application'] not in ['Capacity','QBER','Teleportation','TeleportationWithDemand','CHSH']:
                raise ValueError(f"request {request_name}: Unsupported application")
            
            #If TeleportApplication, teleport parameter must be specified
            if request_props['application'] in ['Teleport','TeleportationWithDemand'] and 'teleport' not in request_props.keys():
                raise ValueError(f"request {request_name}: If application is Teleport, states to teleport must be specified in teleport property")
            node_tx_memories
            #If TeleportWithDemand application, queue technology in origin node must be set
            if request_props['application'] == 'TeleportationWithDemand' and (
                node_tx_memories[request_props['origin']][0] == 'not_set' or
                node_tx_memories[request_props['origin']][1] == 'not_set' or
                node_tx_memories[request_props['origin']][2] == 'not_set' 
            ):
                raise ValueError(f"request {request_name}: If application is TeleportationWithDemand, you must specify queue memory options")
            #If QBER, qber_states parameter must be specified
            if request_props['application'] =='QBER' and 'qber_states' not in request_props.keys():
                raise ValueError(f"request {request_name}: If application is QBER, states to teleport must be specified in qber_states property")
            
            #Of QBER, qber_stater must be [1,0] or [0,1]
            if 'qber_states' in request_props.keys():
                for state in request_props['qber_states']:
                    if state not in [[1,0],[0,1]]:
                        raise ValueError(f"request {request_name}: qber_states can only be 0's or 1's")
                    
def check_parameter(element, parameter):
    '''
    This method verifies that a specified parameter to measure belongs to an elemento
    Input:
        - element:string that indicates type: nodes, links or requests
        - parameter: property that characterizes the element
    Output:
        - result: Bool. False if parameter is incorrect
    '''
    #By default parameter is correct
    result = True

    #Only three type of objects are allowed
    if element not in ['nodes','links','requests']:
        result = False

    #Check that specified parameter allows evolution
    if element == 'nodes' and parameter not in ['gate_duration','gate_duration_X','gate_duration_Z',
                                               'gate_duration_CX','gate_duration_rotations','measurements_duration',
                                               'dephase_gate_rate','depolar_gate_rate','t1_gate_time',
                                               't2_gate_time','dephase_mem_rate','depolar_mem_rate',
                                               't1_mem_time','t2_mem_time','teleport_queue_size']:
        return False
    elif element == 'links' and parameter not in ['endNode_distance','switch_distance',
                                                 'source_fidelity_sq','source_delay','photon_speed_fibre',
                                                 'p_depol_init','p_depol_length','dephase_qchannel_rate',
                                                 'depolar_qchannel_rate','p_loss_init','p_loss_length',
                                                 't1_qchannel_time','t2_qchannel_time','classical_delay_model',
                                                 'gaussian_delay_mean','gaussian_delay_std']:
        result=  False
    elif element == 'requests' and parameter not in ['minfidelity','maxtime','path_fidel_rounds',
                                                    'demand_rate']:
        result = False
    
    return(result)


def load_config(config, element, property, value):
    '''
    Updates property of ALL instances of the specified object with the provided value
    If the property only allows integers, the value will be casted to int
    Input:
        - config: configuration dictionary
        - object: (string) type of oject to update (nodes, links, requests)
        - property: (string) property to update
    Output:
        - config: updated configuration dictionary
    '''

    #Get type of nodes
    nodetypes = {}
    for node in config['nodes']:
        nodename = list(node.keys())[0]
        nodeprops = list(node.values())[0]
        if 'type' in nodeprops.keys():
            nodetypes[nodename] = nodeprops['type']
        else:
            raise ValueError(f"Node {nodename}: No type specified")

    #cast to integer for those that must be int
    if property in ['source_delay','gaussian_delay_mean','gaussian_delay_std','gate_duration','teleport_queue_size',
                    'gate_duration_X','gate_duration_Z','gate_duration_CX','gate_duration_rotations',
                    'measurements_duration','dephase_gate_rate','depolar_gate_rate','t1_gate_time',
                    't2_gate_time','dephase_mem_rate','depolar_mem_rate','maxtime','path_fidel_rounds']:
        value = int(value)
    
    #Update property value
    for instance in config[element]:
        instance_name = list(instance.keys())[0]
        if property not in ['switch_distance','endNode_distance']:
            instance[instance_name][property] = value
        elif property == 'switch_distance' and nodetypes[instance[instance_name]['end1']] == 'switch' \
            and nodetypes[instance[instance_name]['end2']] == 'switch':
                #switch_distance only applies for links between switches
                instance[instance_name]['distance'] = value
        elif property == 'endNode_distance' and (nodetypes[instance[instance_name]['end1']] == 'endNode' \
            or nodetypes[instance[instance_name]['end2']] == 'endNode'):
            instance[instance_name]['distance'] = value

    return(config)


def create_plot(data, request, app):
    """Show a plot of fidelity verus depolarization rate.

    """
    val_name = data.iloc[0]['Parameter']
    if app == 'Capacity':
        fig, axs = plt.subplots(1,3,figsize=(20,20),constrained_layout=True)
        fig.suptitle(request + ' - Capacity', fontsize=14)

        axs[0].plot(data['Value'],data['Generation Rate'],label='Entanglement generation rate (eps)')
        axs[1].plot(data['Value'],data['Mean Fidelity'],label='Mean fidelity')
        axs[2].plot(data['Value'],data['Mean Time'],label='Mean time (ns)')
        for i in [0,1,2]:
            axs[i].legend()
            axs[i].set_xlabel(val_name)
    elif app == 'Teleportation':
        fig, axs = plt.subplots(1,3,figsize=(20,20),constrained_layout=True)
        fig.suptitle(request + ' - Teleportation', fontsize=14)

        axs[0].plot(data['Value'],data['Teleported States'],label='Number of teleported states')
        axs[1].plot(data['Value'],data['Mean Fidelity'],label='Mean fidelity')
        axs[2].plot(data['Value'],data['Mean Time'],label='Mean time (ns)')
        for i in [0,1,2]:
            axs[i].legend()
            axs[i].set_xlabel(val_name)
    elif app == 'QBER':
        fig, axs = plt.subplots(1,3,figsize=(20,20),constrained_layout=True)
        fig.suptitle(request + ' - QBER', fontsize=14)

        axs[0].plot(data['Value'],data['Performed Measurements'],label='Number of measurements')
        axs[1].plot(data['Value'],data['Mean Time'],label='Mean time (ns)')
        axs[2].plot(data['Value'],data['QBER'],label='QBER')
        for i in [0,1,2]:
            axs[i].legend()
            axs[i].set_xlabel(val_name)
    elif app == 'TeleportationWithDemand':
        fig, axs = plt.subplots(3,2,figsize=(20,20),constrained_layout=True)
        fig.suptitle(request + ' - TeleportationWithDemand', fontsize=14)

        axs[0,0].plot(data['Value'],data['Queue Size'],label='Queue Size')
        axs[0,1].plot(data['Value'],data['Discarded Qubits'],label='Discarded Qubits')
        axs[1,0].plot(data['Value'],data['Mean Fidelity'],label='Mean fidelity')
        axs[1,1].plot(data['Value'],data['Mean Time'],label='Mean time (ns)')
        axs[2,0].plot(data['Value'],data['Teleported States'],label='Number of teleported states')
        #We insert a fake subplot, to avoid warning. Later is removed
        axs[2,1].plot(data['Value'],data['Teleported States'],label='Will be deleted')

        for i in [0,1,2]:
            for j in [0,1]:
                axs[i,j].legend()
                axs[i,j].set_xlabel(val_name)

        axs[2,1].remove()
    elif app == 'CHSH':
        fig, axs = plt.subplots(1,2,figsize=(20,20),constrained_layout=True)
        fig.suptitle(request + ' - CHSH', fontsize=14)
        axs[0].plot(data['Value'],data['Wins'],label='% of Wins')
        axs[0].legend()
        axs[0].set_xlabel(val_name)

        axs[1].remove()

    plt.show()