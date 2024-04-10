from network import NetworkManager
from icecream import ic
import logging
from netsquid.util import simlog

'''
logger = logging.getLogger('netsquid')
simlog.logger.setLevel(logging.DEBUG)
# Create a file handler and set the filename
log_file_path = 'simulation.log'
file_handler = logging.FileHandler(log_file_path)

# Set the logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
'''

net = NetworkManager('./network_config.yaml')

'''
print('Salida temporal para verificar red creada')
for node in net.network.nodes.values():
    ic(node.name, node.qmemory,node.qmemory.ports,node.ports)
for conn in net.network.connections.values():
    ic(conn.name)
'''