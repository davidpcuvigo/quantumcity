from network import NetworkManager
from icecream import ic
import logging
import netsquid as ns
from netsquid.util import simlog
from utils import generate_report
from applications import CapacityApplication, TeleportationApplication, DemandGeneratorProtocol

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
    elif application == 'Teleportation':
        qubits = net.get_config('requests',path['request'],'teleport')
        epr_pair = net.get_config('epr_pair','epr_pair')
        app = TeleportationApplication(path, net, qubits, epr_pair, 'Teleportation', name = f"TeleportationApplication_{path['request']}")
    elif application == 'TeleportationWithDemand':
        qubits = net.get_config('requests',path['request'],'teleport')
        demand_rate = net.get_config('requests',path['request'],'demand_rate')
        epr_pair = net.get_config('epr_pair','epr_pair')
        app = TeleportationApplication(path, net, qubits, epr_pair, 'TeleportationWithDemand', rate=demand_rate, name=f"TeleportationWithDemandApplication_{path['request']}")
    elif application == 'QBER':
        qubits = net.get_config('requests',path['request'],'qber_states')
        epr_pair = net.get_config('epr_pair','epr_pair')
        app = TeleportationApplication(path, net, qubits, epr_pair, 'QBER', name = f"QBERApplication_{path['request']}")
    else:
        raise ValueError('Unsupported application')

    app.start()
    dc[path['request']] = [application, app.dc]

#Run simulation
duration = net.get_config('simulation_duration','simulation_duration')
ns.sim_run(duration=duration)

for key,value in dc.items():
    print(f"----Request {key}: Application: {value[0]} --------------------------------------")
    if value[0] == 'Capacity':
        #print(f"Request {key} was able to generate {len(value[1].dataframe)} entanglements with mean fidelity {value[1].dataframe['Fidelity'].mean()} in {value[1].dataframe['time'].mean()} nanoseconds")
        #print(f"Generation rate was {1e9*len(value[1].dataframe)/float(net.get_config('simulation_duration','simulation_duration'))} entanglements per second")
        print(f"Request: {key}")
        print(f"         Generated entanglements: {len(value[1].dataframe)}")
        print(f"         Mean fidelity: {value[1].dataframe['Fidelity'].mean()}")
        print(f"         STD fidelity: {value[1].dataframe['Fidelity'].std()}")
        print(f"         Mean time: {value[1].dataframe['time'].mean()} nanoseconds")
        print(f"         STD time: {value[1].dataframe['time'].std()} nanoseconds")
        print(f"Entanglement generation rate: {1e9*len(value[1].dataframe)/float(net.get_config('simulation_duration','simulation_duration'))} entanglements per second")    
    elif value[0] == 'Teleportation':
        #print(f"Request {key} performed {len(value[1].dataframe)} teleportations with a mean fidelity of {value[1].dataframe['Fidelity'].mean()} and a mean time of {value[1].dataframe['time'].mean()} nanoseconds")
        print(f"Request: {key}")
        print(f"         Teleported states: {len(value[1].dataframe)}")
        print(f"         Mean fidelity: {value[1].dataframe['Fidelity'].mean()}")
        print(f"         STD fidelity: {value[1].dataframe['Fidelity'].std()}")
        print(f"         Mean time: {value[1].dataframe['time'].mean()} nanoseconds")
        print(f"         STD time: {value[1].dataframe['time'].std()} nanoseconds")
    elif value[0] == 'QBER':
        ok = value[1].dataframe['error'].value_counts().loc[0]
        total = value[1].dataframe['error'].count()
        #print(f"Request {key} performed {len(value[1].dataframe)} measurements with a QBER of {(total - ok) / total}%")
        print(f"Request: {key}")
        print(f"         Performed measurements: {len(value[1].dataframe)}")
        print(f"         Mean time: {value[1].dataframe['time'].mean()} nanoseconds")
        print(f"         STD time: {value[1].dataframe['time'].std()} nanoseconds")
        print(f"QBER: {(total - ok) / total}%")   
    elif value[0] == 'TeleportationWithDemand':
        nodename = net.get_config('requests',key,'origin')
        node = net.network.get_node(nodename)
        queue_size = node.get_queue_size()
        print(f"Request: {key}")
        print(f"         Teleported states: {len(value[1].dataframe)}")
        print(f"         Mean fidelity: {value[1].dataframe['Fidelity'].mean()}")
        print(f"         STD fidelity: {value[1].dataframe['Fidelity'].std()}")
        print(f"         Mean time: {value[1].dataframe['time'].mean()} nanoseconds")
        print(f"         STD time: {value[1].dataframe['time'].std()} nanoseconds")
        print(f"Size of queue after simlation time: {queue_size}")
    else:
        pass
    print()


#TODO: Implementar código para representar evolución. Por ejemplo, si se especifica en demand_rate 
# [0,100,1000] implicará medir comenzando con una tasa de 0 e ir incrementando de 100 en 100 hasta 1000
'''
print('Salida temporal para verificar resultados en detalle')
ic(net.get_paths())
ic(net._requests_status)
ic(net._available_links)
ic(net._link_fidelities)
ic(net._memory_assignment)
'''