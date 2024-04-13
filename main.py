from network import NetworkManager
from icecream import ic
import logging
from netsquid.util import simlog
from utils import generate_report

try:
    from pylatex import Document
    print_report = True
except:
    print_report = False

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
if print_report: 
    generate_report(net.get_info_report())
else:
    #Imprimir mensajes por pantalla
    #print(f"To go from {request_props['origin']} to {request_props['destination']} shortest path is {shortest_path}")
    pass

print('Salida temporal para verificar red creada')
ic(net.get_paths())
'''
for node in net.network.nodes.values():
    ic(node.name, node.qmemory,node.qmemory.ports,node.ports)
for conn in net.network.connections.values():
    ic(conn.name)
'''