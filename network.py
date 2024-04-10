import yaml
from icecream import ic
from netsquid.nodes import Node, Connection, Network
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.nodes.connections import DirectConnection
#from pydynaa import EventExpression
import pydynaa
import netsquid as ns
from netsquid.util.datacollector import DataCollector
import pandas as pd
import numpy as np

class NetworkManager():

    def __init__(self, file):
        self.network=""
        self.paths = []
        self._link_fidelities = []
        self._protocol = ""
        self._memory_assignment = {}

        with open(file,'r') as config_file:
            self._config = yaml.safe_load(config_file)
        self._validate_file()
        self._create_network()
        self._measure_link_fidelity()
        self._create_graph()
        self._temporal_red() #Temporal mientras no está implementada la etapa 1 de cálculo de rutas

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

        if node in list(self._memory_assignment.keys()):
            link_serials = self._memory_assignment[node]
            if link in list(link_serials.keys()):
                link_positions = self._memory_assignment[node][link]
                if serial in list(link_positions.keys()):
                    position = link_positions[serial]
                else: #serial does not exists
                    #create serial with memory position the maximum for that link plus one
                    serial_positions = list(link_positions.values())
                    self._memory_assignment[node][link][serial] = max(serial_positions) + 1
                    position = max(serial_positions) + 1
            else: #link does not exist
                #create link and serial. Position will be 0
                self._memory_assignment[node][link] = {}
                self._memory_assignment[node][link][serial] = 0
                position = 0
        else: #node does not exist
            #create node, link, serial and position. Position will be 0
            self._memory_assignment[node] = {}
            self._memory_assignment[node][link] = {}
            self._memory_assignment[node][link][serial] = 0
            position = 0

        return(position)
    
    def _temporal_red(self):
        '''
        Se utiliza para crear rutas mientas no esté implementada la fase 1
        '''
        self.paths = [(['node1','switch1','switch2','node3'],0),(['node2','switch1','switch3','switch2','node4'],2)]
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
        Por ejemplo, que no haya repetición en los nombres
        Que no definamos más hijos clolgando de un switch que leafs se hayan definido
        '''
        '''
        Comprobar que el número de memorias definidas en los switches es suficiente para la red definida:
            en los end_nodes siempre 2
            en los switches: num_links x 2 + end_nodes x 2
        '''

    def _create_network(self):
        '''
        Creates network elements as indicated in configuration file: nodes, links and requests
        Input: dictionary with file contents
        Output: -
        '''
        self.network = Network(self._config['name'])

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
            nodeA = self.network.get_node(props['end1'])
            nodeB = self.network.get_node(props['end2'])
            # Add Quantum Sources to nodes
            num_qsource = props['number_links'] if 'number_links' in props.keys() else 2
            state_sampler = StateSampler(
                [ks.b00, ks.s00],
                [props['source_fidelity_sq'], 1 - props['source_fidelity_sq']])
            for index_qsource in range(num_qsource):
                source = QSource(
                    f"qsource_{props['end1']}_{index_qsource}", state_sampler=state_sampler, num_ports=2, status=SourceStatus.EXTERNAL,
                    models={"emission_delay_model": FixedDelayModel(delay=float(props['source_delay']))})
                nodeA.add_subcomponent(source)
            
                # Setup Classical connections: To be done in protocol Preparation, depends on paths

                # Setup Quantum Channels
                qchannel = QuantumChannel(f"qchannel_{props['end1']}_{props['end2']}_{link_name}_{index_qsource}", 
                                            length = props['distance'],
                                            models={"quantum_loss_model": None, "delay_model": FibreDelayModel(c=float(props['photon_speed_fibre']))})
                port_name_a, port_name_b = self.network.add_connection(
                    nodeA, nodeB, channel_to=qchannel, 
                    label=f"qconn_{props['end1']}_{props['end2']}_{link_name}_{index_qsource}")

                #Setup quantum ports
                nodeA.subcomponents[f"qsource_{props['end1']}_{index_qsource}"].ports["qout1"].forward_output(
                    nodeA.ports[port_name_a])
                nodeA.subcomponents[f"qsource_{props['end1']}_{index_qsource}"].ports["qout0"].connect(
                    nodeA.qmemory.ports[f"qin{self.get_mem_position(props['end1'],link_name,index_qsource)}"])
                nodeB.ports[port_name_b].forward_input(
                    nodeB.qmemory.ports[f"qin{self.get_mem_position(props['end2'],link_name,index_qsource)}"])


            #TODO: Configurar bien los modelos de ruido en los canales cuánticos, según lo que queramos implementar 

    def _measure_link_fidelity(self):
        '''
        Performs a simulation in order to estimate fidelity of each link.
        All links between the same two elements are supossed to have the same fidelity, so only one of them
        is measured in the simulation.
        Input: - will work with self._config
        Output: - will store links with fidelities in self._link_fidelities
        '''

        class FidelityProtocol(LocalProtocol):
            
            def __init__(self, networkmanager, origin, dest, link, qsource_index, name=None):
                self._origin = origin
                self._dest = dest
                self._link = link
                self._qsource_index = qsource_index
                name = name if name else f"FidelityEstimator_{origin.name}_{dest.name}"
                self._memory_left = networkmanager.get_mem_position(self._origin.name, self._link, self._qsource_index)
                self._memory_right = networkmanager.get_mem_position(self._dest.name, self._link, self._qsource_index)
                self._portleft = self._origin.qmemory.ports[f"qin{self._memory_left}"]
                self._portright = self._dest.qmemory.ports[f"qin{self._memory_right}"]
                self.fidelities = []
                super().__init__(nodes={"A": origin, "B": dest}, name=name)

            def run(self):
                #Signal Qsource to start
                self._origin.subcomponents[f"qsource_{self._origin.name}_0"].trigger()
                while True:
                    #ic('Entro yield')
                    expr = yield (self.await_port_input(self._portleft) & self.await_port_input(self._portright))
                    #ic('Salgo del yield')
                    qubit_a, = self._origin.qmemory.peek([self._memory_left])
                    qubit_b, = self._dest.qmemory.peek([self._memory_right])
                    self.fidelities.append(ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True))
                    #self.send_signal(Signals.SUCCESS, 0)
                    self._origin.subcomponents[f"qsource_{self._origin.name}_0"].trigger()
        
        for link in self._config['links']:
            link_name = list(link.keys())[0]
            props_link = list(link.values())[0]
            #if list(link.keys())[0] not in ['nodeswitch4']: continue
            origin = self.network.get_node(props_link['end1'])
            dest = self.network.get_node(props_link['end2'])
            self._protocol = FidelityProtocol(self,origin,dest,link_name,0)
            self._protocol.start()
            ns.sim_run(100000000)
            #print(f"La fidelidad media del enlace {list(link.keys())[0]} es {np.mean(self._protocol.fidelities)} sobre un total de {len(self._protocol.fidelities)} muestras")
            self._link_fidelities.append([list(link.keys())[0], np.mean(self._protocol.fidelities),len(self._protocol.fidelities)])
            ns.sim_stop()
            ns.sim_reset()
            #ic(self._protocol)
            self._create_network() # Si no recreo la red la simulación de las posteriores a la primera no funciona

        ic(self._link_fidelities)

    def _create_graph(self):
        pass

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



