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
from netsquid.examples.teleportation import ClassicalConnection
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.nodes.connections import DirectConnection
from routing_protocols import LinkFidelityProtocol, PathFidelityProtocol, SwapProtocol, CorrectProtocol
from netsquid.qubits import ketstates as ks
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols import LocalProtocol, Signals

class NetworkManager():

    def __init__(self, file):
        self.network=""
        self._paths = []
        self._link_fidelities = {}
        #self._protocol = ""
        self._memory_assignment = {}
        self._available_links = {}

        with open(file,'r') as config_file:
            self._config = yaml.safe_load(config_file)
        self._validate_file()
        self._create_network()
        self._measure_link_fidelity()
        self._create_graph()
        #self._temporal_red() #Temporal mientras no está implementada la etapa 1 de cálculo de rutas


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
        if mode not in ['nodes','links','requests']:
            raise ValueError('Unsupported mode')
        else:
            elements = self._config[mode] 
            for element in elements:
                if list(element.keys())[0] == name:
                    if property:  
                        return(list(element.values())[0][property])
                    else:
                        return(list(element.values())[0])

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
                    self._available_links[link_name]['occupied'].append(next_index)
                    self._available_links[link_name]['avail'] -= 1
                    return([link_name,next_index])
                    #OJO JUAN: Ver cómo lo libero si al final no uso ese índice 
        #If we haven't returned no direct link between both ends
        return('NOLINK')
                

    def _temporal_red(self):
        '''
        Se utiliza para crear rutas mientas no esté implementada la fase 1
        '''
        self._paths = [{
            'request': 'request1',
            'nodes': ['node1','switch1','switch2','node3'],
            'comms': [
                    {'links': ['nodeswitch1-0'], 'source': 'switch1'},
                    {'links': ['interwitch1-0'], 'source': 'switch1'},
                    {'links': ['nodeswitch3-0'], 'source': 'switch2'}],
        'purif_rounds': 0
        },
        {
            'request': 'request2',
            'nodes': ['node2','switch1','switch3','switch2','node4'],
            'comms': [
                    {'links': ['nodeswitch2-0','nodeswitch2-1'], 'source': 'switch1'},
                    {'links': ['interwitch2-0','interwitch2-1'], 'source': 'switch1'},
                    {'links': ['interwitch3-0','interwitch3-1'], 'source': 'switch2'},
                    {'links': ['nodeswitch4-0','nodeswitch4-1'], 'source': 'switch2'}],
            'purif_rounds': 2
        }]
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
            self._available_links[link_name] = {}
            self._available_links[link_name]['avail'] = props['number_links'] if 'number_links' in props.keys() else 2
            self._available_links[link_name]['occupied'] = []

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
            protocol = LinkFidelityProtocol(self,origin,dest,link_name,0)
            protocol.start()
            runtime = props_link['distance']*float(props_link['photon_speed_fibre'])*25
            ns.sim_run(runtime)
            self._link_fidelities[list(link.keys())[0]]= [1-np.mean(protocol.fidelities),np.mean(protocol.fidelities),len(protocol.fidelities)]
            fidelity_values.append(np.mean(protocol.fidelities))
            ns.sim_stop()
            ns.sim_reset()
            self._create_network() # Network must be recreated for the simulations to work

        # calculate fidelity relative to mean fidelities
        #TODO: revisar métodos para ensanchar más las diferencias entre las fidelidades
        for link in list(self._link_fidelities.keys()):
            self._link_fidelities[link][0] /= (1- np.mean(fidelity_values)) 
        ic(self._link_fidelities)

    def dc_setup(self, protocol, nodeA,nodeB,posA=0,posB=0):
        '''
        Creates a data collector in order to measure fidelity of qubits stores en nodeA and nodeB
        against a b00 bell pair
        Inputs:
            - nodeA: node in one end point
            - nodeB: node in the other end point
            - posA: position of memory that stores qubit in nodeA
            - posB: position of qubit that storea quit in nodeB
        Outputs:
            - dc: instance of the configured datacollector
        '''
        def record_stats(evexpr):
            #Record an execution
            protocol = evexpr.triggered_events[-1].source
            result = protocol.get_signal_result(Signals.SUCCESS)
            # Get statistics
            #JUAN: En el ejemplo del purification la posición de memoria la devuelve el protocolo
            #qa, = nodeA.qmemory.pop(positions=[result["posA"]])
            #qb, = nodeB.qmemory.pop(positions=[result["posB"]])
            #fid = qapi.fidelity([qa, qb], ks.b00, squared=True)
            
            return {
                'Fidelity': result['fid'],
                'pairsA': result['pairsA'],
                'pairsB': result['pairsB'],
                'time': result['time']
            }

        dc = DataCollector(record_stats, include_time_stamp=False, include_entity_name=False)
        dc.collect_on(pydynaa.EventExpression(source = protocol, event_type=Signals.SUCCESS.value))
        return(dc)
    
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
                if self._available_links[link_name]['avail']>0:
                    self._graph.add_edge(link_props['end1'],link_props['end2'],weight=self._link_fidelities[link_name][0])
            
            #Esto es temporal, para verificar la red creada cuando todos los recursos están disponibles
            '''
            if first:
                #pos = nx.spring_layout(self._graph) 
                #nx.draw_networkx(self._graph,pos)
                nx.draw(self._graph,with_labels=True)
                #labels = nx.get_edge_attributes(self._graph, 'weight')
                #nx.draw_networkx_edge_labels(self._graph, pos, edge_labels = labels)
                plt.show(block=False)

                first = 0
            '''

            #calculate shotest path
            try:
                shortest_path = nx.shortest_path(self._graph,source=request_props['origin'],target=request_props['destination'], weight='weight')
                path = {
                    'request': request_name, 
                    'nodes': shortest_path, 
                    'purif_rounds': 0,
                    'comms': []}
                for nodepos in range(len(shortest_path)-1):
                    link = self.get_link(shortest_path[nodepos],shortest_path[nodepos+1],next_index=True)
                    source = shortest_path[nodepos] \
                        if f"qsource_{shortest_path[nodepos]}_{link[0]}_{link[1]}" \
                            in (dict(self.network.get_node(shortest_path[nodepos]).subcomponents)).keys() \
                                else shortest_path[nodepos+1]
                    path['comms'].append({'links': [link[0] + '-' + str(link[1])], 'source': source})

                    #Create classical connection
                    cconn = ClassicalConnection(name=f"cconn_{shortest_path[nodepos]}_{shortest_path[nodepos+1]}_{request_name}_1", length=self.get_config('links',link[0],'distance'))
                    #print(f"Conexión: cconn_{shortest_path[nodepos]}_{shortest_path[nodepos+1]}_{request_name}_1")
                    port_name, port_r_name = self.network.add_connection(
                        self.network.get_node(shortest_path[nodepos]), 
                        self.network.get_node(shortest_path[nodepos+1]), 
                        connection=cconn, label="classical",
                        port_name_node1=f"ccon_R_{shortest_path[nodepos]}_{request_name}_1", 
                        port_name_node2=f"ccon_L_{shortest_path[nodepos+1]}_{request_name}_1")
                    
                    # Forward cconn to right most node
                    if f"ccon_L_{shortest_path[nodepos]}_{request_name}_1" in self.network.get_node(shortest_path[nodepos]).ports:
                        self.network.get_node(shortest_path[nodepos]).ports[f"ccon_L_{shortest_path[nodepos]}_{request_name}_1"].bind_input_handler(
                            lambda message, _node=self.network.get_node(shortest_path[nodepos]): _node.ports[f"ccon_R_{_node.name}_{request_name}_1"].tx_output(message))
                        #self.network.get_node(shortest_path[nodepos]).ports[f"ccon_L_{shortest_path[nodepos]}_{request_name}_1"].bind_input_handler(
                        #    self.network.get_node(shortest_path[nodepos]).ports[f"ccon_R_{shortest_path[nodepos]}_{request_name}_1"].tx_output(message))

                #JUAN. Añadir los canales clásicos necesarios. Hecho arriba
                protocol = PathFidelityProtocol(self,path,100) #We measure E2E fidelity 100 times
                dc = self.dc_setup(protocol, self.network.get_node(path['nodes'][0]), self.network.get_node(path['nodes'][-1]))
                protocol.start()
                ns.sim_run()
                #ic(dc.dataframe)
                print("Average fidelity of generated entanglement via a repeater and with filtering: {}, with average time: {}"\
                        .format(dc.dataframe["Fidelity"].mean(),dc.dataframe["time"].mean()))
                protocol.stop()
                #JUAN. CURIOSO aquí. Si no haga el sim_reset puedo relanzar la simulación y funciona.
                #While fidelidad y tiempo no cumplan los requisitos de la request
                #TODO: Comprobar fidelidad y tiempo. Si ambos OK --> continuar a siguiente request
                #TODO: Si no OK
                    #Si tiempo superior al demandado: Si KO abortar request y procesar siguiente
                    #Si tiempo OK aumentar distil en 1 y volver a medir fidelidad E2E

                #path = nx.all_simple_paths(self._graph,source=request_props['origin'],target=request_props['destination'])
                ic(self._available_links)
            except nx.exception.NetworkXNoPath:
                shortest_path = 'NOPATH'
            print(f"To go from {request_props['origin']} to {request_props['destination']} shortest path is {shortest_path}")

        #plt.show()

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

        qproc = QuantumProcessor(name, num_positions=num_memories, fallback_to_nonphysical=True)
        return qproc



