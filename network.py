import yaml
from icecream import ic
import netsquid as ns
import networkx as nx
import numpy as np
from netsquid.nodes import Node, Connection, Network
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel, GaussianDelayModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel, T1T2NoiseModel, QuantumErrorModel, FibreLossModel
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.nodes.connections import DirectConnection
from routing_protocols import LinkFidelityProtocol, PathFidelityProtocol
from netsquid.qubits import ketstates as ks
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_MEASURE, INSTR_X, INSTR_Z,  INSTR_CNOT, IGate
import netsquid.qubits.operators as ops
from utils import dc_setup

class Switch(Node):
    def __init__(self,name,qmemory):
        self._swap_queue = []
        super().__init__(name,qmemory=qmemory)

class EndNode(Node):
    def __init__(self,name,qmemory):
        #TODO: Change queue to quantum memory queue?
        self._transmit_queue = []
        super().__init__(name,qmemory=qmemory)

    def request_teleport(self,state):
        self._transmit_queue.append(state)

    def retrieve_teleport(self):
        if len(self._transmit_queue) > 0:
            return self._transmit_queue.pop(0)
        else:
            return(None)
        
    def get_queue_size(self):
        return(len(self._transmit_queue))
        

class NetworkManager():
    '''
    The only initiallization parameter is the name of the file 
    storing all the network definition
    '''

    def __init__(self, file):
        self.network=""
        self._paths = []
        self._link_fidelities = {}
        self._memory_assignment = {}
        self._available_links = {}
        self._requests_status = []

        with open(file,'r') as config_file:
            self._config = yaml.safe_load(config_file)
        self._validate_file()
        self._create_network()
        self._measure_link_fidelity()
        self._calculate_paths()

    def get_info_report(self):
        '''
        Generates and returns information for the pdf report
        '''
        report_info = {}
        report_info['link_fidelities'] = self._link_fidelities
        report_info['requests_status'] = self._requests_status
        return(report_info)

    def get_config(self, mode, name, property=None):
        '''
        Enables configuration queries
        Input:
            - mode: ['nodes'|'links|'requests']
            - name: name of the element to query
            - property: attribute to query. If None (default), all attributes are returned
        Output:
            - value of required attribute
        '''
        if mode not in ['name','simulation_duration','epr_pair','link_fidel_rounds','path_fidel_rounds','nodes','links','requests']:
            raise ValueError('Unsupported mode')
        else:
            elements = self._config[mode] 
            #Querying for a global property
            if mode in ['name','epr_pair','simulation_duration','link_fidel_rounds','path_fidel_rounds']: 
                return (elements)
            
            #Querying for an element type
            found = False
            for element in elements:
                if list(element.keys())[0] == name:
                    if property:
                        try:  
                            return(list(element.values())[0][property])
                        except:
                            return('NOT_FOUND')
                    else:
                        return(list(element.values())[0])
            
            #name not found
            return('NOT_FOUND')

    def get_mem_position(self, node, link, serial):
        '''
        Maps node and link to memory position.
        If non assigned, creates assignment and stores it in private attribute
        If already assigned, gets memory position
        Input:
            - node: string. Name of node
            - link: string. Name of link
            - serial: integer or string. Index of link
        Output:
            -integer: memory position in the specified node to be used
        ''' 
        serial = str(serial)
        values_list = []

        if node in list(self._memory_assignment.keys()):
            node_links = self._memory_assignment[node]
            #get maximum assigned position
            for node_link in list(node_links.keys()):
                for link_serial in list(self._memory_assignment[node][node_link].keys()):
                    values_list.append(self._memory_assignment[node][node_link][link_serial])
            position = max(values_list) if len(values_list) > 0 else -1

            if link in list(node_links.keys()):
                link_serials = self._memory_assignment[node][link]
                if serial in list(link_serials.keys()):
                    position = link_serials[serial] #This is the assigned position
                else: #serial does not exists
                    #create serial with memory position the maximum for that link plus one
                    position += 1
                    self._memory_assignment[node][link][serial] = position
            else: #link does not exist
                #create link and serial. Position will be 0
                self._memory_assignment[node][link] = {}
                position += 1
                self._memory_assignment[node][link][serial] = position
        else: #node does not exist
            #create node, link, serial and position. Position will be 0
            self._memory_assignment[node] = {}
            self._memory_assignment[node][link] = {}
            self._memory_assignment[node][link][serial] = 0
            position = 0

        return(position)

    def get_paths(self):
        return self._paths
    
    def get_link(self, node1, node2, next_index = False):
        '''
        Obtains the name of the link between two nodes.
        Input:
            - node1, node2: connected nodes by the link
            - next_index: if True returns also the next available index in the link. False by default
        '''
        
        links = self._config['links']
        for link in links:
            link_name = list(link.keys())[0]
            link_props = self.get_config('links',link_name)
            if (link_props['end1'] == node1 and link_props['end2'] == node2) or (link_props['end1'] == node2 and link_props['end2'] == node1):
                if not next_index:
                    return(link_name)
                else:
                    next_index = max(self._available_links[link_name]['occupied']) + 1 \
                        if len(self._available_links[link_name]['occupied']) != 0 else 0
                    #We return the next instance if there are available
                    if self._available_links[link_name]['avail'] > 0:
                        self._available_links[link_name]['occupied'].append(next_index)
                        self._available_links[link_name]['avail'] -= 1
                        return([link_name,next_index])
                    
        #If we haven't returned no direct link between both ends
        return('NOLINK')

    def release_link(self, link_name, index):
        '''
        Returns as available the index of the specified link.
        Input:
            - link_name: link. string
            - index: index in the link to be released. Can be string or integer
        '''
        self._available_links[link_name]['avail'] += 1
        self._available_links[link_name]['occupied'].remove(int(index))

    def _validate_file(self):
        '''
        Performs validations on the provided configuration file
        Input: -
        Output: -
        '''

        #Verify that global mandatory parameters exist
        if 'name' not in self._config.keys() or 'link_fidel_rounds' not in self._config.keys() \
            or 'path_fidel_rounds' not in self._config.keys() or 'nodes' not in self._config.keys() \
                or 'links' not in self._config.keys() or 'requests' not in self._config.keys() \
                    or 'epr_pair' not in self._config.keys() or 'simulation_duration' not in self._config.keys(): 
            raise ValueError('Invalid configuration file, check global parameters')

        #Check link sintax
        links = self._config['links']
            
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
                               'gaussian_delay_std':'integer'}
        
        #get list of nde names
        nodenames = [list(node.keys())[0] for node in self._config['nodes']]

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
            if (self.get_config('nodes',link_props['end1'],'type') == 'endNode' or \
                self.get_config('nodes',link_props['end2'],'type') == 'endNode') and \
                'number_links' in link_props.keys() and link_props['number_links'] != 2:
                raise ValueError(f"{link_name}: number_links can only be 2 between node and switch")

            #Check allowed values of noise model
            allowed_qchannel_noise_model = ['DephaseNoiseModel','DepolarNoiseModel','T1T2NoiseModel','FibreLossModel','FibreDepolarizeModel']
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

            #If quantum channel noise model is FibreLosseModel p_loss_init and p_losslength must be declared
            if 'qchannel_noise_model' in link_props.keys() and  \
                link_props['qchannel_noise_model'] == 'FibreLossModel'  \
                and ('p_loss_init' not in link_props.keys() or 'p_loss_length' not in link_props.keys()):
                raise ValueError(f"link {link_name}: When FibreLossModel is selected for quantum channel, p_loss_init and p_loss_length must be defined")
            
            #If quantum channel noise model is T1T2NoiseModel t1 & t2 times must be declared
            if 'qchannel_noise_model' in link_props.keys() and  \
                link_props['qchannel_noise_model'] == 'T1T2NoiseModel'  \
                and ('t1_qchannel_time' not in link_props.keys() or 't2_qchannel_time' not in link_props.keys()):
                raise ValueError(f"link {link_name}: When T1T2NoiseModel is selected for quantum channel, t1_qchannel_time and t2_qhannel_time must be defined")
    
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
                               't2_mem_time':'float'}
        
        for node in self._config['nodes']:
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

            #Check that in switch nodes we have > 2*num_links
            if node_props['type'] == 'endNode' and 'num_memories' in node_props.keys() and \
                node_props['num_memories'] != 4:
                raise ValueError(f"node {node_name}: if num_memories declared in endNode, must aways be 2")
            elif node_props['type'] == 'switch':
                #must check than number of memories is greater than connected links
                total_links = 0
                for link in self._config['links']:
                    link_props = list(link.values())[0]
                    if link_props['end1'] == node_name or link_props['end2'] == node_name:
                        total_links += link_props['number_links'] \
                            if 'number_links' in link_props.keys() else 2
                if total_links > node_props['num_memories']:
                    raise ValueError(f"node {node_name}: not enough memories. Need at least {total_links}")

        #Check requests sintax
        requests = self._config['requests']
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

    def _create_network(self):
        '''
        Creates network elements as indicated in configuration file: nodes, links and requests
        Input: dictionary with file contents
        Output: -
        '''
        self.network = Network(self._config['name'])
        self._memory_assignment = {}
        self._available_links = {}

        #nodes creation
        switches = [] #List with all switches
        end_nodes = [] # List with all nodes
        for node in self._config['nodes']:
            name = list(node.keys())[0]
            props = list(node.values())[0]
            if props['type'] == 'switch':
                switch = Switch(name, qmemory=self._create_qprocessor(f"qproc_{name}",props['num_memories'], nodename=name))
                switches.append(switch)
            elif props['type'] == 'endNode':
                props['num_memories'] = 4 #In an end node we always have 4 memories (2 por entanglement preparation)
                endnode = EndNode(name, qmemory=self._create_qprocessor(f"qproc_{name}",props['num_memories'], nodename=name))
                end_nodes.append(endnode)
            else:
                raise ValueError('Undefined network element found')

        network_nodes = switches+end_nodes
        self.network.add_nodes(network_nodes)

        #links creation
        for link in self._config['links']:
            link_name = list(link.keys())[0]
            props = list(link.values())[0]
            #store available resources per link
            self._available_links[link_name] = {}
            self._available_links[link_name]['avail'] = props['number_links'] if 'number_links' in props.keys() else 2
            self._available_links[link_name]['occupied'] = []

            nodeA = self.network.get_node(props['end1'])
            nodeB = self.network.get_node(props['end2'])
            # Add Quantum Sources to nodes
            num_qsource = props['number_links'] if 'number_links' in props.keys() else 2
            epr_state = ks.b00 if self._config['epr_pair'] == 'PHI_PLUS' else ks.b01

            state_sampler = StateSampler(
                [epr_state, ks.s00, ks.s01, ks.s10, ks.s11],
                [props['source_fidelity_sq'], (1 - props['source_fidelity_sq'])/4, (1 - props['source_fidelity_sq'])/4,
                 (1 - props['source_fidelity_sq'])/4, (1 - props['source_fidelity_sq'])/4])
            for index_qsource in range(num_qsource):
                if self.get_config('nodes',props['end1'],'type') == 'switch':
                    qsource_origin = nodeA 
                    qsource_dest = nodeB
                else:
                    qsource_origin = nodeB
                    qsource_dest = nodeA
                #Setup QSource
                source = QSource(
                        f"qsource_{qsource_origin.name}_{link_name}_{index_qsource}", state_sampler=state_sampler, num_ports=2, status=SourceStatus.EXTERNAL,
                        models={"emission_delay_model": FixedDelayModel(delay=float(props['source_delay']))})
                qsource_origin.add_subcomponent(source)
                # Setup Quantum Channels
                #get channel noise model from config
                #TODO: All these models apply for a quantum channel?
                if self.get_config('links',link_name,'qchannel_noise_model') == 'FibreDepolarizeModel':
                    qchannel_noise_model = FibreDepolarizeModel(p_depol_init=float(self.get_config('links',link_name,'p_depol_init')),
                                                                p_depol_length=float(self.get_config('links',link_name,'p_depol_length')))
                elif self.get_config('links',link_name,'qchannel_noise_model') == 'DephaseNoiseModel':
                    qchannel_noise_model = DephaseNoiseModel(float(self.get_config('links',link_name,'dephase_qchannel_rate')))
                elif self.get_config('links',link_name,'qchannel_noise_model') == 'DepolarNoiseModel':
                    qchannel_noise_model = DepolarNoiseModel(float(self.get_config('links',link_name,'depolar_qchannel_rate')))
                elif self.get_config('links',link_name,'qchannel_noise_model') == 'FibreLossModel':
                    qchannel_noise_model = FibreLossModel(p_loss_init=float(self.get_config('links',link_name,'p_loss_init')),
                                                           p_loss_length=float(self.get_config('links',link_name,'p_loss_length')))
                elif self.get_config('links',link_name,'qchannel_noise_model') == 'T1T2NoiseModel':
                    qchannel_noise_model = T1T2NoiseModel(T1=float(self.get_config('links',link_name,'t1_qchannel_time')),
                                              T2=float(self.get_config('links',link_name,'t2_qchannel_time')))
                else:
                    qchannel_noise_model = None

                qchannel = QuantumChannel(f"qchannel_{qsource_origin.name}_{qsource_dest.name}_{link_name}_{index_qsource}", 
                        length = props['distance'],
                        models={"quantum_loss_model": qchannel_noise_model, "delay_model": FibreDelayModel(c=float(props['photon_speed_fibre']))})
                port_name_a, port_name_b = self.network.add_connection(
                        qsource_origin, qsource_dest, channel_to=qchannel, 
                        label=f"qconn_{qsource_origin.name}_{qsource_dest.name}_{link_name}_{index_qsource}")
                
                #Setup quantum ports
                qsource_origin.subcomponents[f"qsource_{qsource_origin.name}_{link_name}_{index_qsource}"].ports["qout1"].forward_output(
                    qsource_origin.ports[port_name_a])
                qsource_origin.subcomponents[f"qsource_{qsource_origin.name}_{link_name}_{index_qsource}"].ports["qout0"].connect(
                    qsource_origin.qmemory.ports[f"qin{self.get_mem_position(qsource_origin.name,link_name,index_qsource)}"])
                qsource_dest.ports[port_name_b].forward_input(
                    qsource_dest.qmemory.ports[f"qin{self.get_mem_position(qsource_dest.name,link_name,index_qsource)}"])
                
                # Setup Classical connections: To be done in routing preparation, depends on paths

    def _measure_link_fidelity(self):
        '''
        Performs a simulation in order to estimate fidelity of each link.
        All links between the same two elements are supossed to have the same fidelity, so only one of them
        is measured in the simulation.
        Input: 
            - will work with self._config
        Output: 
            - will store links with fidelities in self._link_fidelities
        '''
        fidelity_values = []
        for link in self._config['links']:
            link_name = list(link.keys())[0]
            props_link = list(link.values())[0]
            origin = self.network.get_node(props_link['end1'])
            dest = self.network.get_node(props_link['end2'])

            protocol = LinkFidelityProtocol(self,origin,dest,link_name,0,self._config['link_fidel_rounds'])
            protocol.start()
            #runtime = props_link['distance']*float(props_link['photon_speed_fibre'])*25
            #will run as many times as specified in config file
            ns.sim_run()
            #We want to minimize the product of the costs, not the sum. log(ab)=log(a)+log(b)
            #so we will work with logarithm
            self._link_fidelities[list(link.keys())[0]]= [-np.log(np.mean(protocol.fidelities)),np.mean(protocol.fidelities),len(protocol.fidelities)]
            ns.sim_stop()
            ns.sim_reset()
            self._create_network() # Network must be recreated for the simulations to work
    
    def _release_path_resources(self, path):
        '''
        Removes classical connections used by a path and releases quantum links for that path
        Input:
            - path: dict. Path dictionary describing calculated path from origin to destination
        '''
        for nodepos in range(len(path['nodes'])-1):
            nodeA = self.network.get_node(path['nodes'][nodepos])
            nodeB = self.network.get_node(path['nodes'][nodepos+1])
            #Delete classical connections
            for i in [1,2]:
                conn = self.network.get_connection(nodeA, nodeB,f"cconn_{nodeA.name}_{nodeB.name}_{path['request']}_{i}")
                self.network.remove_connection(conn)
                #Unable to delete ports. Will remain unconnected
                #TODO: Check how to delete ports
                #nodeA.ports[f"ccon_R_{nodeA.name}_{request_name}_{i}"].remove()
                #nodeB.ports[f"ccon_L_{nodeB.name}_{request_name}_{i}"].remove()
        #remove classical purification connection
        connA = self.network.get_connection(self.network.get_node(path['nodes'][0]), 
                self.network.get_node(path['nodes'][-1]),
                f"cconn_distil_{path['nodes'][0]}_{path['nodes'][-1]}_{path['request']}")
        #Even though classical is bidirectional, only one has to be deleted
        self.network.remove_connection(connA)

        #release quantum channels used by this path
        for link in path['comms']:
            for link_instance in link['links']:
                self.release_link(link_instance.split('-')[0],link_instance.split('-')[1])       

    def _calculate_paths(self):
        first = 1
        for request in self._config['requests']:
            request_name = list(request.keys())[0]
            request_props = list(request.values())[0]

            # Create network graph using available links
            self._graph = nx.Graph()
            for node in self._config['nodes']:
                node_name = list(node.keys())[0]
                node_props = list(node.values())[0]
                if node_props['type'] =='switch':
                    self._graph.add_node(node_name,color='#CF9239',style='filled',fillcolor='#CF9239')
                else:
                    self._graph.add_node(node_name,color='#5DABAB',style='filled',fillcolor='#5DABAB',shape='square')

            for link in self._config['links']:
                link_name = list(link.keys())[0]
                link_props = list(link.values())[0]
                if self._available_links[link_name]['avail']>0:
                    self._graph.add_edge(link_props['end1'],link_props['end2'],weight=self._link_fidelities[link_name][0])

            #Network graph generation, to include in report
            if first:
                '''
                fig = plt.figure()
                nx.draw_networkx(self._graph,ax=fig.add_subplot())
                usematplotlib('Agg')
                fig.savefig('./output/graf.png')
                '''
                
                gr = nx.nx_agraph.to_agraph(self._graph)
                gr.draw('./output/graf.png', prog='fdp')
                first = 0
            
            try:
                shortest_path = nx.shortest_path(self._graph,source=request_props['origin'],target=request_props['destination'], weight='weight')
                purif_rounds = 0
                path = {
                    'request': request_name, 
                    'nodes': shortest_path, 
                    'purif_rounds': purif_rounds,
                    'comms': []}
                for nodepos in range(len(shortest_path)-1):
                    #Get link connecting nodes
                    link = self.get_link(shortest_path[nodepos],shortest_path[nodepos+1],next_index=True)
                    #Determine which of the 2 nodes connected by the link is the source
                    source = shortest_path[nodepos] \
                        if f"qsource_{shortest_path[nodepos]}_{link[0]}_{link[1]}" \
                            in (dict(self.network.get_node(shortest_path[nodepos]).subcomponents)).keys() \
                                else shortest_path[nodepos+1]
                    #Add quantum link to path
                    path['comms'].append({'links': [link[0] + '-' + str(link[1])], 'source': source})

                    #Get classical channel delay model
                    classical_delay_model = None
                    fibre_delay_model = self.get_config('links',link[0], 'classical_delay_model')
                    if fibre_delay_model == 'NOT_FOUND' or fibre_delay_model == 'FibreDelayModel':
                        classical_delay_model = FibreDelayModel(c=float(self.get_config('links',link[0],'photon_speed_fibre')))
                    elif fibre_delay_model == 'GaussianDelayModel':
                        classical_delay_model = GaussianDelayModel(delay_mean=float(self.get_config('links',link[0],'gaussian_delay_mean')),
                                                                        delay_std = float(self.get_config('links',link[0],'gaussian_delay_std')))
                    else: # In case other, we assume FibreDelayModel
                        classical_delay_model = FibreDelayModel(c=float(self.get_config('links',link[0],'photon_speed_fibre')))

                    #Create classical connection. We create channels even if purification is not needed
                    for i in [1,2]:
                        cconn = ClassicalConnection(name=f"cconn_{shortest_path[nodepos]}_{shortest_path[nodepos+1]}_{request_name}_{i}", 
                                                    length=self.get_config('links',link[0],'distance'))
                        cconn.subcomponents['Channel_A2B'].models['delay_model'] = classical_delay_model

                        port_name, port_r_name = self.network.add_connection(
                            self.network.get_node(shortest_path[nodepos]), 
                            self.network.get_node(shortest_path[nodepos+1]), 
                            connection=cconn, label=f"cconn_{shortest_path[nodepos]}_{shortest_path[nodepos+1]}_{request_name}_{i}",
                            port_name_node1=f"ccon_R_{shortest_path[nodepos]}_{request_name}_{i}", 
                            port_name_node2=f"ccon_L_{shortest_path[nodepos+1]}_{request_name}_{i}")

                        #Forward cconn to right most node
                        if f"ccon_L_{path['nodes'][nodepos]}_{request_name}_{i}" in self.network.get_node(path['nodes'][nodepos]).ports:
                            self.network.get_node(path['nodes'][nodepos]).ports[f"ccon_L_{path['nodes'][nodepos]}_{request_name}_{i}"].bind_input_handler(self._handle_message,tag_meta=True)
                        
                    #TODO: DELETE when working  
                    '''
                    if f"ccon_L_{shortest_path[nodepos]}_{request_name}_1" in self.network.get_node(shortest_path[nodepos]).ports:
                        self.network.get_node(shortest_path[nodepos]).ports[f"ccon_L_{shortest_path[nodepos]}_{request_name}_1"].bind_input_handler(
                                lambda message, _node=self.network.get_node(shortest_path[nodepos]): _node.ports[f"ccon_R_{_node.name}_{request_name}_1"].tx_output(message))
                    if f"ccon_L_{shortest_path[nodepos]}_{request_name}_2" in self.network.get_node(shortest_path[nodepos]).ports:
                            self.network.get_node(shortest_path[nodepos]).ports[f"ccon_L_{shortest_path[nodepos]}_{request_name}_2"].bind_input_handler(
                                lambda message, _node=self.network.get_node(shortest_path[nodepos]): _node.ports[f"ccon_R_{_node.name}_{request_name}_2"].tx_output(message))
                    '''

                #Setup classical channel for purification
                #calculate distance from first to last node
                total_distance = 0
                average_photon_speed = 0
                for comm in path['comms']:
                    link_distance = self.get_config('links',comm['links'][0].split('-')[0],'distance')
                    link_photon_speed = float(self.get_config('links',comm['links'][0].split('-')[0],'photon_speed_fibre'))
                    total_distance += link_distance
                    average_photon_speed += link_photon_speed * link_distance
                average_photon_speed = average_photon_speed / total_distance


                conn_purif = DirectConnection(
                    f"ccon_distil_{request_name}",
                    ClassicalChannel(f"cconn_distil_{shortest_path[0]}_{shortest_path[-1]}_{request_name}", 
                                     length=total_distance,
                                     models={'delay_model': FibreDelayModel(c=average_photon_speed)}),
                    ClassicalChannel(f"cconn_distil_{shortest_path[-1]}_{shortest_path[0]}_{request_name}", 
                                     length=total_distance,
                                     models={'delay_model': FibreDelayModel(c=average_photon_speed)})
                )
                self.network.add_connection(self.network.get_node(shortest_path[0]), 
                                           self.network.get_node(shortest_path[-1]), connection=conn_purif,
                                           label=f"cconn_distil_{shortest_path[0]}_{shortest_path[-1]}_{request_name}",
                                           port_name_node1=f"ccon_distil_{shortest_path[0]}_{request_name}",
                                           port_name_node2=f"ccon_distil_{shortest_path[-1]}_{request_name}")
                end_simul = False

                #get measurements to do for average fidelity
                fidel_rounds = request_props['path_fidel_rounds'] \
                    if 'path_fidel_rounds' in request_props.keys() else self._config['path_fidel_rounds']
 
                #Initially no purification
                protocol = PathFidelityProtocol(self,path,fidel_rounds, purif_rounds) #We measure E2E fidelity accordingly to config file times
                
                while end_simul == False:
                    dc = dc_setup(protocol)
                    protocol.start()
                    ns.sim_run()
                    protocol.stop()
                    
                    print(f"Request {request_name} purification rounds {purif_rounds} fidelity {dc.dataframe['Fidelity'].mean()}/{request_props['minfidelity']} in {dc.dataframe['time'].mean()}/{request_props['maxtime']} nanoseconds métricas: {len(dc.dataframe)}")
                    if dc.dataframe["time"].mean() > request_props['maxtime']:
                        #request cannot be fulfilled. Mark as rejected and continue
                        self._requests_status.append({
                            'request': request_name, 
                            'shortest_path': shortest_path,
                            'result': 'rejected', 
                            'reason': 'cannot fulfill time',
                            'purif_rounds': purif_rounds,
                            'fidelity': dc.dataframe["Fidelity"].mean(),
                            'time': dc.dataframe["time"].mean()})
                        
                        #release classical and quantum channels
                        self._release_path_resources(path)

                        end_simul = True
                    elif dc.dataframe["Fidelity"].mean() >= request_props['minfidelity']:
                        #request can be fulfilled
                        self._requests_status.append({
                            'request': request_name, 
                            'shortest_path': shortest_path,
                            'result': 'accepted', 
                            'reason': '-',
                            'purif_rounds': purif_rounds,
                            'fidelity': dc.dataframe["Fidelity"].mean(),
                            'time': dc.dataframe["time"].mean()})
                        path['purif_rounds'] = purif_rounds
                        self._paths.append(path)
                        end_simul=True
                    else: #purification is needed
                        purif_rounds += 1
                        #if first time with purification add second quantum link in path
                        if purif_rounds == 1:
                            #check if we have available link resources for second path
                            available_resources = True
                            for comm in path['comms']:
                                link_name = comm['links'][0].split('-')[0]
                                if self._available_links[link_name]['avail'] == 0:
                                    available_resources = False
                                    break

                            if not available_resources:
                                #No available resources for second link instance, must free path resources
                                self._release_path_resources(path)

                                #return no path
                                shortest_path = 'NOPATH'
                                self._requests_status.append({
                                    'request': request_name, 
                                    'shortest_path': shortest_path,
                                    'result': 'rejected', 
                                    'reason': 'no available resources',
                                    'purif_rounds': 'na',
                                    'fidelity': 0,
                                    'time': 0})
                                
                                end_simul = True

                            else:
                                new_comms = []
                                for nodepos in range(len(shortest_path)-1):
                                    link = self.get_link(shortest_path[nodepos],shortest_path[nodepos+1],next_index=True)
                                    for comm in path['comms']:
                                        if comm['links'][0].split('-')[0] == link[0]:
                                            comm['links'].append(link[0] + '-' + str(link[1]))
                                            new_comms.append(comm)
                                path['comms'] = new_comms   
                                protocol.set_purif_rounds(purif_rounds)

                        else:
                            protocol.set_purif_rounds(purif_rounds)

            except nx.exception.NetworkXNoPath:
                shortest_path = 'NOPATH'
                self._requests_status.append({
                            'request': request_name, 
                            'shortest_path': shortest_path,
                            'result': 'rejected', 
                            'reason': 'no available resources',
                            'purif_rounds': '-',
                            'fidelity': 0,
                            'time': 0})

    def _handle_message(self,msg):
        input_port = msg.meta['rx_port_name']
        forward_port = input_port.replace('ccon_L_','ccon_R_')
        port_elements = input_port.split('_')
        node = self.network.get_node(port_elements[2])
        node.ports[forward_port].tx_output(msg)
        return

    def _create_qprocessor(self,name,num_memories,nodename):
        '''
        Factory to create a quantum processor for each node.

        In an end node it has 4 memory positions. In a swich 2xnum_links.
        Adapted from available example in NetSquid website

        Input:
            - name: name of quantum processor
            - nodename: name of node where it is placed

        Output:
            - instance of QuantumProcessor

        '''

        _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
        _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

        #get gate durations from configuration
        gate_duration = self.get_config('nodes',nodename,'gate_duration') \
            if self.get_config('nodes',nodename,'gate_duration') != 'NOT_FOUND' else 0
        gate_duration_X = self.get_config('nodes',nodename,'gate_duration_X') \
            if self.get_config('nodes',nodename,'gate_duration_X') != 'NOT_FOUND' else gate_duration
        gate_duration_Z = self.get_config('nodes',nodename,'gate_duration_Z') \
            if self.get_config('nodes',nodename,'gate_duration_Z') != 'NOT_FOUND' else gate_duration
        gate_duration_CX = self.get_config('nodes',nodename,'gate_duration_CX') \
            if self.get_config('nodes',nodename,'gate_duration_CX') != 'NOT_FOUND' else gate_duration
        gate_duration_rotations = self.get_config('nodes',nodename,'gate_duration_rotations') \
            if self.get_config('nodes',nodename,'gate_duration_rotations') != 'NOT_FOUND' else gate_duration
        measurements_duration = self.get_config('nodes',nodename,'measurements_duration') \
            if self.get_config('nodes',nodename,'measurements_duration') != 'NOT_FOUND' else gate_duration

        #get gate noise model
        if self.get_config('nodes',nodename,'gate_noise_model') == 'DephaseNoiseModel':
            gate_noise_model = DephaseNoiseModel(float(self.get_config('nodes',nodename,'dephase_gate_rate')))
        elif self.get_config('nodes',nodename,'gate_noise_model') == 'DepolarNoiseModel':
            gate_noise_model = DepolarNoiseModel(float(self.get_config('nodes',nodename,'depolar_gate_rate')))
        elif self.get_config('nodes',nodename,'gate_noise_model') == 'T1T2NoiseModel':
            gate_noise_model = T1T2NoiseModel(T1=float(self.get_config('nodes',nodename,'t1_gate_time')),
                                              T2=float(self.get_config('nodes',nodename,'t2_gate_time')))
        else:
            gate_noise_model = None

        #set memories noise model
        if self.get_config('nodes',nodename,'mem_noise_model') == 'DephaseNoiseModel':
            mem_noise_model = DephaseNoiseModel(float(self.get_config('nodes',nodename,'dephase_mem_rate')))
        elif self.get_config('nodes',nodename,'mem_noise_model') == 'DepolarNoiseModel':
            mem_noise_model = DepolarNoiseModel(float(self.get_config('nodes',nodename,'depolar_mem_rate')))
        elif self.get_config('nodes',nodename,'mem_noise_model') == 'T1T2NoiseModel':
            mem_noise_model = T1T2NoiseModel(T1=float(self.get_config('nodes',nodename,'t1_mem_time')),
                                              T2=float(self.get_config('nodes',nodename,'t2_mem_time')))
        else:
            mem_noise_model = None

        #define available instructions   
        physical_instructions = [
            PhysicalInstruction(INSTR_X, duration=gate_duration_X,
                                quantum_noise_model=gate_noise_model
                                ),
            PhysicalInstruction(INSTR_Z, duration=gate_duration_Z,
                                quantum_noise_model=gate_noise_model
                                ),
            PhysicalInstruction(INSTR_MEASURE_BELL, 
                                duration=measurements_duration,
                                quantum_noise_model=gate_noise_model),
            PhysicalInstruction(INSTR_MEASURE, 
                                duration=measurements_duration,
                                quantum_noise_model=gate_noise_model),
            PhysicalInstruction(INSTR_CNOT, 
                                duration=gate_duration_CX,
                                quantum_noise_model=gate_noise_model),
            PhysicalInstruction(_INSTR_Rx, 
                                duration=gate_duration_rotations,
                                quantum_noise_model=gate_noise_model),
            PhysicalInstruction(_INSTR_RxC, 
                                duration=gate_duration_rotations,
                                quantum_noise_model=gate_noise_model)
        ]

        #build quantum processor
        qproc = QuantumProcessor(name, 
                                 num_positions=num_memories, 
                                 phys_instructions = physical_instructions,
                                 fallback_to_nonphysical=False,
                                 mem_moise_models=[mem_noise_model] * num_memories)
        return qproc


class FibreDepolarizeModel(QuantumErrorModel):
    """Custom non-physical error model used to show the effectiveness
    of repeater chains.

    The default values are chosen to make a nice figure,
    and don't represent any physical system.

    Parameters
    ----------
    p_depol_init : float, optional
        Probability of depolarization on entering a fibre.
        Must be between 0 and 1. Default 0.009
    p_depol_length : float, optional
        Probability of depolarization per km of fibre.
        Must be between 0 and 1. Default 0.025

    """
    def __init__(self, p_depol_init=0.09, p_depol_length=0.025):
        super().__init__()
        self.properties['p_depol_init'] = p_depol_init
        self.properties['p_depol_length'] = p_depol_length
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Uses the length property to calculate a depolarization probability,
        and applies it to the qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Not used.

        """
        for qubit in qubits:
            prob = 1 - (1 - self.properties['p_depol_init']) * np.power(
                10, - kwargs['length']**2 * self.properties['p_depol_length'] / 10)
            ns.qubits.depolarize(qubit, prob=prob)

class ClassicalConnection(Connection):
    """A connection that transmits classical messages in one direction, from A to B.
    Copied from official NetSquid's examples (teleportation)

    Parameters
    ----------
    length : float
        End to end length of the connection [km].
    name : str, optional
       Name of this connection.

    """

    def __init__(self, length, name="ClassicalConnection"):
        super().__init__(name=name)
        self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length,
                                               models={"delay_model": FibreDelayModel()}),
                              forward_input=[("A", "send")],
                              forward_output=[("B", "recv")])