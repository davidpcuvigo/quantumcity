from network import NetworkManager
from icecream import ic

net = NetworkManager('./network_config.yaml')

print('Salida temporal para verificar red creada')
for node in net.network.nodes.values():
    ic(node.name, node.qmemory)
for conn in net.network.connections.values():
    ic(conn.name)