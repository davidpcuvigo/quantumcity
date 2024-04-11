import yaml
from icecream import ic
import pydynaa
import netsquid as ns
import networkx as nx
import matplotlib.pyplot as plt
from netsquid.util.datacollector import DataCollector
import pandas as pd
import numpy as np
from netsquid.nodes import Node, Connection, Network
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.nodes.connections import DirectConnection
from routing_protocols import FidelityProtocol
from netsquid.qubits import ketstates as ks

class NetworkManager():

    def __init__(self, file):
        self.network=""
        self._paths = []
        self._link_fidelities = {}
        self._protocol = ""
        self._memory_assignment = {}
        self._available_links = {}

        with open(file,'r') as config_file:
            self._config = yaml.safe_load(config_file)
        self._validate_file()
        self._create_network()
        self._measure_link_fidelity()
        self._create_graph()
        self._temporal_red() #Temporal mientras no está implementada la etapa 1 de cálculo de rutas


    def get_config(self, mode, name, property):
        '''
        Enables configuration queries
        Input:
            - mode: ['nodes'|'links|'requests']
            - name: name of the element to query
            - property: attribute to query
        Output:
            - value of required attribute
        '''
        if mode not in ['nodes','links']:
            raise ValueError('Unsupported mode')
        else:
            elements = self._config[mode] 
            for element in elements:
                if list(element.keys())[0] == name:  
                    return(list(element.values())[0][property])

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

    def _temporal_red(self):
        '''
        Se utiliza para crear rutas mientas no esté implementada la fase 1
        '''
        self._paths = [(['node1','switch1','switch2','node3'],0),(['node2','switch1','switch3','switch2','node4'],2)]
        node1 = self.network.get_node('node1')
        node2 = self.network.get_node('node2')
        node3 = self.network.get_node('node3')
        node4 = self.network.get_node('node4')
        switch1 = self.network.get_node('switch1')
        switch2 = self.network.get_node('switch2')
        switch3 = self.network.get_node('switch3')
        # Conexiones clásicas path1. No tiene Distil
        conns = [('ccon_R_node1_request1_1','ccon_L_switch1_request1_1'),
                 ('ccon_R_switch1_request1_1','ccon_L_switch2_request1_1'),
                 ('ccon_R_switch2_request1_1','ccon_L_node3_request1_1')]
        for con in conns:
            cconn = ClassicalConnection(name=f"ccon_{con[0].split('_')[2]}_{con[1].split('_')[2]}_{con[0].split('_')[3]}_1", length=5)
            node_origin = self.network.get_node(con[0].split('_')[2])
            node_dest = self.network.get_node(con[1].split('_')[2])
            port_name, port_r_name = self.network.add_connection(
                node_origin, node_dest, connection=cconn, label="classical",
                port_name_node1=con[0], port_name_node2=con[1])
        # Forward cconn to right most node
        switch1.ports['ccon_L_switch1_request1_1'].bind_input_handler(
            lambda message, _node=switch1: _node.ports["ccon_R_switch1_request1_1"].tx_output(message))
        switch2.ports['ccon_L_switch2_request1_1'].bind_input_handler(
            lambda message, _node=switch1: _node.ports["ccon_R_switch2_request1_1"].tx_output(message))
        
        # Conexiones clásicas path2. Tiene Distil
        conns = [('ccon_R_node2_request2_1','ccon_L_switch1_request2_1'),
                 ('ccon_R_node2_request2_2','ccon_L_switch1_request2_2'),
                 ('ccon_R_switch1_request2_1','ccon_L_switch3_request2_1'),
                 ('ccon_R_switch1_request2_2','ccon_L_switch3_request2_2'),
                 ('ccon_R_switch3_request2_1','ccon_L_switch2_request2_1'),
                 ('ccon_R_switch3_request2_2','ccon_L_switch2_request2_2'),
                 ('ccon_R_switch2_request2_1','ccon_L_node4_request2_1'),
                 ('ccon_R_switch2_request2_2','ccon_L_node4_request2_2')]
        for con in conns:
            cconn = ClassicalConnection(name=f"ccon_{con[0].split('_')[2]}_{con[1].split('_')[2]}_{con[0].split('_')[3]}_{con[0].split('_')[4]}", length=5)
            node_origin = self.network.get_node(con[0].split('_')[2])
            node_dest = self.network.get_node(con[1].split('_')[2])
            port_name, port_r_name = self.network.add_connection(
                node_origin, node_dest, connection=cconn, label=f"ccon_{con[0].split('_')[2]}_{con[1].split('_')[2]}_{con[0].split('_')[3]}_{con[0].split('_')[4]}",
                port_name_node1=con[0], port_name_node2=con[1])
        # Forward cconn to right most node
        switch1.ports['ccon_L_switch1_request2_1'].bind_input_handler(
            lambda message, _node=switch1: _node.ports["ccon_R_switch1_request2_1"].tx_output(message))
        switch1.ports['ccon_L_switch1_request2_2'].bind_input_handler(
            lambda message, _node=switch1: _node.ports["ccon_R_switch1_request2_2"].tx_output(message))
        switch3.ports['ccon_L_switch3_request2_1'].bind_input_handler(
            lambda message, _node=switch3: _node.ports["ccon_R_switch3_request2_1"].tx_output(message))
        switch3.ports['ccon_L_switch3_request2_2'].bind_input_handler(
            lambda message, _node=switch3: _node.ports["ccon_R_switch3_request2_2"].tx_output(message))
        switch2.ports['ccon_L_switch2_request2_1'].bind_input_handler(
            lambda message, _node=switch2: _node.ports["ccon_R_switch2_request2_1"].tx_output(message))
        switch2.ports['ccon_L_switch2_request2_2'].bind_input_handler(
            lambda message, _node=switch2: _node.ports["ccon_R_switch2_request2_2"].tx_output(message))
        # Conexión para el Distil
        conn_distil = DirectConnection("Distil_Node2Node4",
            ClassicalChannel("ccon_distil_node1_request2", length=5,
                                models={"delay_model": FibreDelayModel(c=200e3)}),
            ClassicalChannel("ccon_distil_node2_request2", length=5,
                                models={"delay_model": FibreDelayModel(c=200e3)}))
        self.network.add_connection(node2, node4, connection=conn_distil,
                                    port_name_node1="ccon_distil_node1_request2", port_name_node2="ccon_distil_node2_request2")



    def _validate_file(self):
        '''
        Performs validations on the provided configuration file
        Input: -
        Output: -
        '''
        #ic(self._config)
        pass #TODO
        #TODO: realiza validaciones del contenido del fichero 
        '''
        -Por ejemplo, que no haya repetición en los nombres
        -Que no definamos más hijos clolgando de un switch que leafs se hayan definido
        -Comprobar que el número de memorias definidas en los switches es suficiente para la red definida:
            en los end_nodes siempre 4
            en los switches: num_links x 2 + end_nodes x 2
        -Verificar que no hay links entre dos elementos de tipo endNode
        '''

        #links cannot contain hyphens
        links = self._config['links']
        for link in links:
            if list(link.keys())[0].find('-') != -1: raise ValueError ('Links cannot contain hyphens')       

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
                switch = Node(name, qmemory=self._create_qprocessor(f"qproc_{name}",props['num_memories']))
                switches.append(switch)
            elif props['type'] == 'endNode':
                props['num_memories'] = 4 #In an end node we always have 4 memories
                endnode = Node(name, qmemory=self._create_qprocessor(f"qproc_{name}",props['num_memories']))
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
            self._available_links[link_name] = props['number_links'] if 'number_links' in props.keys() else 2

            nodeA = self.network.get_node(props['end1'])
            nodeB = self.network.get_node(props['end2'])
            # Add Quantum Sources to nodes
            num_qsource = props['number_links'] if 'number_links' in props.keys() else 2
            state_sampler = StateSampler(
                [ks.b00, ks.s00],
                [props['source_fidelity_sq'], 1 - props['source_fidelity_sq']])
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
                qchannel = QuantumChannel(f"qchannel_{qsource_origin.name}_{qsource_dest.name}_{link_name}_{index_qsource}", 
                        length = props['distance'],
                        models={"quantum_loss_model": None, "delay_model": FibreDelayModel(c=float(props['photon_speed_fibre']))})
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

            #TODO: Configurar bien los modelos de ruido en los canales cuánticos, según lo que queramos implementar 

    def _measure_link_fidelity(self):
        '''
        Performs a simulation in order to estimate fidelity of each link.
        All links between the same two elements are supossed to have the same fidelity, so only one of them
        is measured in the simulation.
        Input: - will work with self._config
        Output: - will store links with fidelities in self._link_fidelities
        '''
        fidelity_values = []
        for link in self._config['links']:
            link_name = list(link.keys())[0]
            props_link = list(link.values())[0]
            origin = self.network.get_node(props_link['end1'])
            dest = self.network.get_node(props_link['end2'])
            self._protocol = FidelityProtocol(self,origin,dest,link_name,0)
            self._protocol.start()
            runtime = props_link['distance']*float(props_link['photon_speed_fibre'])*25
            ns.sim_run(runtime)
            #self._link_fidelities[list(link.keys())[0]]= [10000*(1-np.mean(self._protocol.fidelities)),np.mean(self._protocol.fidelities),len(self._protocol.fidelities)]
            self._link_fidelities[list(link.keys())[0]]= [1-np.mean(self._protocol.fidelities),np.mean(self._protocol.fidelities),len(self._protocol.fidelities)]
            fidelity_values.append(np.mean(self._protocol.fidelities))
            ns.sim_stop()
            ns.sim_reset()
            self._create_network() # Network must be recreated for the simulations to work

        # calculate fidelity relative to mean fidelities
        #TODO: revisar métodos para ensanchar más las diferencias
        for link in list(self._link_fidelities.keys()):
            self._link_fidelities[link][0] /= (1- np.mean(fidelity_values)) 
            print(link)
        ic(self._link_fidelities)

    def _create_graph(self):
        first = 1
        for request in self._config['requests']:
            request_name = list(request.keys())[0]
            request_props = list(request.values())[0]

            self._graph = nx.Graph()
            for node in self._config['nodes']:
                node_name = list(node.keys())[0]
                node_props = list(node.values())[0]
                self._graph.add_node(node_name)
            for link in self._config['links']:
                link_name = list(link.keys())[0]
                link_props = list(link.values())[0]
                #self._available_links['nodeswitch4']=0
                if self._available_links[link_name]>0:
                    self._graph.add_edge(link_props['end1'],link_props['end2'],weight=self._link_fidelities[link_name][0])
            
            #Esto es temporal, para verificar la red creada cuando todos los recursos están disponibles
            if first:
                #pos = nx.spring_layout(self._graph) 
                #nx.draw_networkx(self._graph,pos)
                nx.draw(self._graph,with_labels=True)
                #labels = nx.get_edge_attributes(self._graph, 'weight')
                #nx.draw_networkx_edge_labels(self._graph, pos, edge_labels = labels)
                plt.show(block=False)

                first = 0
            
            #calculate shotest path
            try:
                path = nx.shortest_path(self._graph,source=request_props['origin'],target=request_props['destination'], weight='weight')
                #TODO: Iniciar simulación E2E con el path obtenido para medir fidelidad y tiempo
                #TODO: recorrer los elementos usados y decrementar contador de links disponibles
                #path = nx.all_simple_paths(self._graph,source=request_props['origin'],target=request_props['destination'])
            except nx.exception.NetworkXNoPath:
                path = 'NOPATH'
            print(f"To go from {request_props['origin']} to {request_props['destination']} shortest path is {path}")

        plt.show()
        #TODO: Para empezar supongo un link siempre (sin distil). Cuando se necesite distil harán falta dos enlaces


    def _create_qprocessor(self,name,num_memories):
        """Factory to create a quantum processor for each node.

        In an end node it has 4 memory positions. In a swich 2xnum_links.

        Parameters
        ----------
        name : str
            Name of the quantum processor.

        Returns
        -------
        :class:`~netsquid.components.qprocessor.QuantumProcessor`
            A quantum processor to specification.

        """

        qproc = QuantumProcessor(name, num_positions=num_memories)
        return qproc



