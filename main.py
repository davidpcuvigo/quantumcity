from network import NetworkManager
from icecream import ic
import logging
import netsquid as ns
from netsquid.util import simlog
from utils import generate_report
from applications import CapacityApplication, TeleportationApplication, CircTeleportationApplication

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


dc={}
for path in net.get_paths():
    application = net.get_config('requests',path['request'],'application')
    if application == 'Capacity':
        app = CapacityApplication(path, net, f"CapacityApplication_{path['request']}")
    elif application == 'Teleport':
        app = TeleportationApplication(path, net, f"TeleportationApplication_{path['request']}")
    elif application == 'TeleportCirc':
        app = CircTeleportationApplication(path, net, f"CircTeleportationApplication_{path['request']}")
    else:
        raise ValueError('Unsupported application')

    app.start()
    dc[path['request']] = app.dc

#Run simulation
duration = net.get_config('simulation_duration','simulation_duration')
ns.sim_run(duration=duration)

for key,value in dc.items():
    print(f"Request {key} fidelity {value.dataframe['Fidelity'].mean()} in {value.dataframe['time'].mean()} nanoseconds métricas: {len(value.dataframe)}")


'''
print('Salida temporal para verificar resultados en detalle')
ic(net.get_paths())
ic(net._requests_status)
ic(net._available_links)
ic(net._link_fidelities)
ic(net._memory_assignment)
'''