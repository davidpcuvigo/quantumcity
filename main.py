from network import NetworkManager
from icecream import ic
import pandas as pd
import logging
import os
import netsquid as ns
from netsquid.util import simlog
from utils import generate_report, validate_conf, check_parameter, load_config, create_plot
import yaml
from applications import CapacityApplication, TeleportationApplication, CHSHApplication

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

file = './network_config.yaml'

#Read configuration file
with open(file,'r') as config_file:
    config = yaml.safe_load(config_file)

#TODO: Ask for execution mode: fixed or evolution
# Si evolution pedir parámetro, valor de inicio, paso y valor de fin
mode = input('Do you want to perform fixed parameter simulation or evolution? (F: Fixed, E: Evolution): ')
if mode == 'F':
    steps = 1
    element = 'FixedSimul'
    property = 'FixedSimul'
    value = 0
    #config_list = [config]
    validate_conf(config)
elif mode == 'E':
    element = input('Enter object (nodes/links/requests). Parameter will be set in ALL instances: ')
    property = input('Enter property: ')
    if not check_parameter(element, property):
        raise ValueError("Evolution for that parameter not supported")
    
    min_val = float(input('Enter minimum value: '))

    max_val = float(input('Enter maximum value: '))
    if max_val <= min_val: raise ValueError('Maximum must be greater than minimum')

    steps = int(input('Enter number of steps (minimum 2): '))
    if steps <= 1: raise ValueError('Minumum of 2 steps needed')

    step_size = (max_val - min_val) / (steps - 1)
else:
    raise ValueError('Unsupported operation. Valid: E or F')

results = {} #This list will store data of the different sumulations
for sim in range(steps):
    #reset simulation to start over
    ns.sim_reset()

    #If we are simulating with evolution we load the configuration parameters
    if steps > 1:
        value = min_val + sim*step_size
        #Update configuration object with each value to simulate with
        config = load_config(config, element, property, value)
        #Check configuration file sintax
        validate_conf(config)

    #Instantiate NetWorkManager based on configuration. Will launch routing protocol
    net = NetworkManager(config)

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
        elif application == 'CHSH':
            app = CHSHApplication(path, net, name = f"QBERApplication_{path['request']}")
        else:
            raise ValueError('Unsupported application')

        app.start()
        dc[path['request']] = [application, app.dc]

    #Run simulation
    duration = net.get_config('simulation_duration','simulation_duration')
    ns.sim_run(duration=duration)

    print('----------------')

    #Acumulate results in general dataframe in case we want evolution
    for key,detail in dc.items():
        if detail[0] == 'Capacity':
            sim_result = {'Application':detail[0],
                          'Request': key,
                            'Parameter': element + '$' + property, 
                            'Value': value,
                            'Generated Entanglements': len(detail[1].dataframe),
                            'Mean Fidelity': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['Fidelity'].mean(),
                            'STD Fidelity': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['Fidelity'].std(),
                            'Mean Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].mean(),
                            'STD Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].std(),
                            'Generation Rate': 0 if len(detail[1].dataframe) == 0 else 1e9*len(detail[1].dataframe)/float(config['simulation_duration'])
                            }
        elif detail[0] == 'Teleportation':
            sim_result = {'Application':detail[0],
                          'Request': key,
                            'Parameter': element + '$' + property, 
                            'Value': value,
                            'Teleported States': len(detail[1].dataframe),
                            'Mean Fidelity': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['Fidelity'].mean(),
                            'STD Fidelity': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['Fidelity'].std(),
                            'Mean Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].mean(),
                            'STD Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].std()
                            }
        elif detail[0] == 'QBER':
            ok = 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['error'].value_counts().loc[0]
            total = 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['error'].count()
            sim_result = {'Application':detail[0],
                          'Request': key,
                            'Parameter': element + '$' + property, 
                            'Value': value,
                            'Performed Measurements': len(detail[1].dataframe),
                            'Mean Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].mean(),
                            'STD Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].std(),
                            'QBER': 100 if len(detail[1].dataframe) == 0 else (total - ok) / total
                            }
        elif detail[0] == 'TeleportationWithDemand':
            nodename = net.get_config('requests',key,'origin')
            node = net.network.get_node(nodename)
            #queue_size = node.get_queue_size()
            sim_result = {'Application':detail[0],
                          'Request': key,
                            'Parameter': element + '$' + property, 
                            'Value': value,
                            'Teleported States': len(detail[1].dataframe),
                            'Mean Fidelity': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['Fidelity'].mean(),
                            'STD Fidelity': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['Fidelity'].std(),
                            'Mean Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].mean(),
                            'STD Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].std(),
                            'Queue Size': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['queue_size'].max(),
                            'Discarded Qubits': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['discarded_qubits'].max()
                            }
        elif application == 'CHSH':
            wins = 0 if len(detail[1].dataframe) == 0 else len(detail[1].dataframe[detail[1].dataframe['wins']==1])
            total = 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['wins'].count()
            sim_result = {'Application':detail[0],
                          'Request': key,
                            'Parameter': element + '$' + property, 
                            'Value': value,
                            'Measurements': len(detail[1].dataframe),
                            'Wins': 0 if total == 0 else (wins)/total,
                            'Mean Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].mean(),
                            'STD Time': 0 if len(detail[1].dataframe) == 0 else detail[1].dataframe['time'].std(),
                            }
        else:
            raise ValueError('Unsupported application')

        if key not in results.keys(): results[key] = [] #Initialize list
        results[key].append(sim_result)

#Al this point in results we have de simulation data
simulation_data = {}
for key in results.keys():
    df_sim_result = pd.DataFrame(results[key])
    simulation_data[key] = df_sim_result
    #df_sim_result.set_index('Value', inplace = True)

#Print results
try:
    os.remove('./output/results.csv')
except:
    pass
for key,value in simulation_data.items():
    print(f"----Request {key}: Application: {value.iloc[0]['Application']} --------------------------------------")
    if value.iloc[0]['Application'] == 'Capacity':
        print(f"         Generated entanglements: {value['Generated Entanglements'].tolist()}")
        print(f"         Mean fidelity: {value['Mean Fidelity'].tolist()}")
        print(f"         STD fidelity: {value['STD Fidelity'].tolist()}")
        print(f"         Mean time: {value['Mean Time'].tolist()} nanoseconds")
        print(f"         STD time: {value['STD Time'].tolist()} nanoseconds")
        print(f"Entanglement generation rate: {value['Generation Rate'].tolist()} entanglements per second")    
    elif value.iloc[0]['Application'] == 'Teleportation':
        print(f"         Teleported states: {value['Teleported States'].tolist()}")
        print(f"         Mean fidelity: {value['Mean Fidelity'].tolist()}")
        print(f"         STD fidelity: {value['STD Fidelity'].tolist()}")
        print(f"         Mean time: {value['Mean Time'].tolist()} nanoseconds")
        print(f"         STD time: {value['STD Time'].tolist()} nanoseconds")
    elif value.iloc[0]['Application'] == 'QBER':
        print(f"         Performed measurements: {value['Performed Measurements'].tolist()}")
        print(f"         Mean time: {value['Mean Time'].tolist()} nanoseconds")
        print(f"         STD time: {value['STD Time'].tolist()} nanoseconds")
        print(f"QBER: {value['QBER'].tolist()}%")
    elif value.iloc[0]['Application'] == 'TeleportationWithDemand':
        print(f"         Teleported states: {value['Teleported States'].tolist()}")
        print(f"         Mean fidelity: {value['Mean Fidelity'].tolist()}")
        print(f"         STD fidelity: {value['STD Fidelity'].tolist()}")
        print(f"         Mean time: {value['Mean Time'].tolist()} nanoseconds")
        print(f"         STD time: {value['STD Time'].tolist()} nanoseconds")
        print(f"Queue size at end of simulation: {value['Queue Size'].tolist()}")
        print(f"Discarded qubits: {value['Discarded Qubits'].tolist()}")
    elif value.iloc[0]['Application'] == 'CHSH':
        print(f"        Measurements: {value['Measurements'].tolist()}")
        print(f"         Mean time: {value['Mean Time'].tolist()} nanoseconds")
        print(f"         STD time: {value['STD Time'].tolist()} nanoseconds")
        print(f"Wins: {value['Wins'].tolist()}")
    print()
    #If evolution, plot graphs
    if mode == 'E': create_plot(value,key,value.iloc[0]['Application'])

    #Save data to disk
    value.to_csv('./output/results.csv', mode='a', index=False, header=False)

if print_report: 
    generate_report(net.get_info_report())

'''
print('Salida temporal para verificar resultados en detalle')
ic(net.get_paths())
ic(net._requests_status)
ic(net._available_links)
ic(net._link_fidelities)
ic(net._memory_assignment)
'''