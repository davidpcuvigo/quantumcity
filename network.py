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


class NetworkManager():
    network=""

    def __init__(self, file):
        with open('network_config.yaml','r') as config_file:
            self._config = yaml.safe_load(config_file)
        self._validate_file()
        self._create_network()

    def _validate_file(self):
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

        #ic(config)
        #ic(network_nodes)

        #links creation
        for link in self._config['links']:
            name = list(link.keys())[0]
            props = list(link.values())[0]
            nodeA = self.network.get_node(props['end1'])
            nodeB = self.network.get_node(props['end2'])
            # Add Quantum Sources to nodes
            num_qsource = props['number_links'] if 'number_links' in props.keys() else 2
            state_sampler = StateSampler(
                qs_reprs=[ks.b01, ks.s00],
                probabilities=[props['source_fidelity_sq'], 1 - props['source_fidelity_sq']])
            for index_qsource in range(num_qsource):
                source = QSource(
                    f"qsource_{props['end1']}_{index_qsource}", state_sampler=state_sampler, num_ports=2, status=SourceStatus.EXTERNAL,
                    models={"emission_delay_model": FixedDelayModel(delay=float(props['source_delay']))})
                nodeA.add_subcomponent(source)
            
                # Setup Classical connections: To be done in protocol Preparation, depends on paths

                # Setup Quantum Channels
                qchannel = QuantumChannel(f"qchannel_{props['end1']}_{props['end2']}_{name}_{index_qsource}", 
                                            length = props['distance'],
                                            models={"quantum_loss_model": None, "delay_model": FibreDelayModel(c=props['photon_speed_fibre'])})
                port_name_a, port_name_b = self.network.add_connection(
                    nodeA, nodeB, channel_to=qchannel, 
                    label=f"qconn_{props['end1']}_{props['end2']}_{name}_{index_qsource}")

                #Setup quantum ports
                nodeA.subcomponents[f"qsource_{props['end1']}_{index_qsource}"].ports["qout1"].forward_output(
                    nodeA.ports[port_name_a])
                nodeA.subcomponents[f"qsource_{props['end1']}_{index_qsource}"].ports["qout0"].connect(
                    nodeA.qmemory.ports[f"qin{str(2*index_qsource)}"])
                nodeB.ports[port_name_b].forward_input(nodeB.qmemory.ports[f"qin{str(2*index_qsource+1)}"])


            #TODO: Configurar bien los modelos de ruido en los canales cuánticos, según lo que queramos implementar 
        
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



