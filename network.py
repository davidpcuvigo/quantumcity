import yaml
from icecream import ic
from netsquid.nodes import Node, Connection, Network
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection

with open('network_config.yaml','r') as config_file:
    config = yaml.safe_load(config_file)

#TODO: encapsular lectura de parámetros con lógica que implemente validaciones. 
    '''
    Por ejemplo, que no haya repetición en los nombres
    Que no definamos más hijos clolgando de un switch que leafs se hayan definido
    '''
    '''
    Comprobar que el número de memorias definidas en los switches es suficiente para la red definida:
        en los end_nodes siempre 2
        en los switches: num_links x 2 + end_nodes x 2
    '''

def create_qprocessor(name,num_leafs):
    """Factory to create a quantum processor for each node.

    In an end node it has 2 memory positions. In a swich 2xnum_links.

    Parameters
    ----------
    name : str
        Name of the quantum processor.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """

    qproc = QuantumProcessor(name, num_positions=num_leafs+1)
    return qproc


network = Network(config['name'])

#nodes creation
switches = [] #List with all switches
end_nodes = [] # List with all nodes
for node in config['nodes']:
    name = list(node.keys())[0]
    props = list(node.values())[0]
    if props['type'] == 'switch':
        switch = Node(name, qmemory=create_qprocessor(f"qproc_{name}",props['num_memories']))
        switches.append(switch)
    elif props['type'] == 'endNode':
        props['num_memories'] = 2 #In an end node we always have 2 memories
        endnode = Node(name, qmemory=create_qprocessor(f"qproc_{name}",props['num_memories']))
        end_nodes.append(endnode)
    else:
        raise ValueError('Undefined network element found')

network_nodes = switches+end_nodes
network.add_nodes(network_nodes)

ic(config)
ic(network_nodes)
#links creation
for link in config['links']:
    name = list(link.keys())[0]
    props = list(link.values())[0]
    nodeA = network.get_node(props['end1'])
    nodeB = network.get_node(props['end2'])
    # Setup entangling connection between nodes:
    qconn1 = EntanglingConnection(name=f"qconn1_{name}_{props['end1']}-{props['end2']}", length=props['distance'],
                                     source_frequency=props['source_frequency'])
    qconn2 = EntanglingConnection(name=f"qconn2_{name}_{props['end1']}-{props['end2']}", length=props['distance'],
                                     source_frequency=props['source_frequency'])
    #TODO: Añadir modelo de ruido al canal. Admás, de momento solo hay 1 par de entangling connection, hacen falta las que se indiquen
    #Connect entangling souce to both ends  
    port_name1, port_r_name1 = network.add_connection(
            nodeA, nodeB, connection=qconn1, label=f"qconn1_{name}_{props['end1']}-{props['end2']}")
    port_name2, port_r_name2 = network.add_connection(
            nodeA, nodeB, connection=qconn2, label=f"qconn2_{name}_{props['end1']}-{props['end2']}")
    # Forward qconn directly to quantum memories for right and left inputs:
    nodeA.ports[port_name1].forward_input(nodeA.qmemory.ports["qin0"])  # left node
    nodeA.ports[port_name2].forward_input(nodeA.qmemory.ports["qin1"])  # left node
    nodeB.ports[port_r_name1].forward_input(nodeB.qmemory.ports["qin0"])  # right node
    nodeB.ports[port_r_name2].forward_input(nodeB.qmemory.ports["qin1"])  # right node
    #TODO: Cuando sea un switch y haya varios links, los índices no siempre son 0/1

    # Create classical connection from end1 to end2 end viceversa
    cconn = ClassicalConnection(name=f"cconn_{name}_{props['end1']}-{props['end2']}", length=props['distance'])
    port_name, port_r_name = network.add_connection(nodeA, nodeB, connection=cconn, label="classical",
            port_name_node1=f"ccon_R_{name}_{props['end1']}_{props['end2']}", port_name_node2=f"ccon_L_{name}_{props['end2']}_{props['end1']}")
    cconn2 = ClassicalConnection(name=f"cconn_{name}_{props['end2']}-{props['end1']}", length=props['distance'])
    port_name, port_r_name = network.add_connection(nodeB, nodeA, connection=cconn2, label="classical2",
            port_name_node1=f"ccon_R_{name}_{props['end2']}_{props['end1']}", port_name_node2=f"ccon_L_{name}_{props['end1']}_{props['end2']}")
    #TODO: Ver las conexiones clásicas necesarias para implementar Distil

print('Salida temporal para verificar red creada')
for node in network.nodes.values():
    ic(node.name, node.qmemory)
for conn in network.connections.values():
    ic(conn.name)
