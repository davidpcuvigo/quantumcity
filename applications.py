from utils import dc_setup
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.components.component import Message
from icecream import ic
import numpy as np
from netsquid.util.simtools import sim_time
from netsquid.qubits import qubitapi as qapi, create_qubits, assign_qstate
from netsquid.qubits import ketstates as ks
from protocols import RouteProtocol
from netsquid.qubits import set_qstate_formalism, QFormalism
from netsquid.components.instructions import INSTR_MEASURE_BELL
from netsquid.components import QuantumProgram
from protocols import SwapCorrectProgram
from network import ClassicalConnection
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel, GaussianDelayModel
import cmath

class GeneralApplication(LocalProtocol):

    def __init__(self, path, networkmanager, name=None):
        #Signal that asks for entanglement
        self._ent_request = 'START_ENTANGLEMENT'
        self.add_signal(self._ent_request)

        self.name = name if name else f"Application_Unidentified"
        super().__init__(nodes=networkmanager.network.nodes, name=name)
        self._path = path
        self._networkmanager = networkmanager

        ent_start_expression = self.await_signal(self, self._ent_request)
        self.add_subprotocol(RouteProtocol(networkmanager,path,ent_start_expression,path['purif_rounds']))

        #Initialize data collector that will gather metrics
        self.dc = dc_setup(self)
    
class CapacityApplication(GeneralApplication):

    def __init__(self, path, networkmanager, name=None):
        name = name if name else f"CapacityApplication_Unidentified"
        super().__init__(path, networkmanager)
    
    def run(self):
        self.start_subprotocols()

        #Get type of EPR to use
        epr_state = ks.b00 if self._networkmanager.get_config('epr_pair','epr_pair') == 'PHI_PLUS' else ks.b01

        #Though in this simulations positions in nodes are always 0, we query in case this is changed in the future
        first_link = self._path['comms'][0]['links'][0]
        last_link = self._path['comms'][-1]['links'][0]
        mem_posA_1 = self._networkmanager.get_mem_position(self._path['nodes'][0],first_link.split('-')[0],first_link.split('-')[1])
        mem_posB_1 = self._networkmanager.get_mem_position(self._path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])

        while True:
            start_time = sim_time()
            #Send signal for entanglement generation
            self.send_signal(self._ent_request)

            #Wait for  entanglement to be generated on both ends
            yield self.await_signal(self.subprotocols[f"RouteProtocol_{self._path['request']}"],Signals.SUCCESS)

            #Measure fidelity and send metrics to datacollector
            qa, = self._networkmanager.network.get_node(self._path['nodes'][0]).qmemory.pop(positions=[mem_posA_1])
            qb, = self._networkmanager.network.get_node(self._path['nodes'][-1]).qmemory.pop(positions=[mem_posB_1])
            fid = qapi.fidelity([qa, qb], epr_state, squared=True)
            result = {
                'posA': mem_posA_1,
                'posB': mem_posB_1,
                'fid': fid,
                'time': sim_time() - start_time
            }
            #send result to datacollector
            self.send_signal(Signals.SUCCESS, result)


class TeleportationApplication(GeneralApplication):

    def __init__(self, path, networkmanager, qubits, epr_pair, name=None):
        name = name if name else f"TeleportApplication_Unidentified"
        super().__init__(path, networkmanager)

        self._qubits = qubits

        mem_posB_1 = self._networkmanager.get_mem_position(self._path['nodes'][-1],self._path['comms'][-1]['links'][0].split('-')[0],0)
        #mem_posB_1=0
        self.add_subprotocol(TeleportCorrectProtocol(networkmanager.network.get_node(path['nodes'][-1]),mem_posB_1,f"TeleportCorrectProtocol_{path['request']}",path['request'],epr_pair))

        self._build_teleport_classic()

    def _build_teleport_classic(self):
        '''
        Adds classical channels for Teleportation protocol
        '''
        for nodepos in range(len(self._path['nodes'])-1):
            link = self._networkmanager.get_link(self._path['nodes'][nodepos],self._path['nodes'][nodepos+1],next_index=False)

            #Get classical channel delay model
            classical_delay_model = None
            fibre_delay_model = self._networkmanager.get_config('links',link, 'classical_delay_model')
            if fibre_delay_model == 'NOT_FOUND' or fibre_delay_model == 'FibreDelayModel':
                classical_delay_model = FibreDelayModel(c=float(self._networkmanager.get_config('links',link,'photon_speed_fibre')))
            elif fibre_delay_model == 'GaussianDelayModel':
                classical_delay_model = GaussianDelayModel(delay_mean=float(self._networkmanager.get_config('links',link,'gaussian_delay_mean')),
                    delay_std = float(self._networkmanager.get_config('links',link,'gaussian_delay_std')))
            else: # In case other, we assume FibreDelayModel
                classical_delay_model = FibreDelayModel(c=float(self._networkmanager.get_config('links',link,'photon_speed_fibre')))

            #Create classical connection for each link
            cconn = ClassicalConnection(name=f"cconn_{self._path['nodes'][nodepos]}_{self._path['nodes'][nodepos+1]}_{self._path['request']}_teleport", 
                length=self._networkmanager.get_config('links',link,'distance'))
            cconn.subcomponents['Channel_A2B'].models['delay_model'] = classical_delay_model

            port_name, port_r_name = self._networkmanager.network.add_connection(
                self._networkmanager.network.get_node(self._path['nodes'][nodepos]), 
                self._networkmanager.network.get_node(self._path['nodes'][nodepos+1]), 
                connection=cconn, label=f"cconn_{self._path['nodes'][nodepos]}_{self._path['nodes'][nodepos+1]}_{self._path['request']}_teleport",
                port_name_node1=f"ccon_R_{self._path['nodes'][nodepos]}_{self._path['request']}_teleport", 
                port_name_node2=f"ccon_L_{self._path['nodes'][nodepos+1]}_{self._path['request']}_teleport")

            #Forward cconn to right most node
            if f"ccon_L_{self._path['nodes'][nodepos]}_{self._path['request']}_teleport" in self._networkmanager.network.get_node(self._path['nodes'][nodepos]).ports:
                self._networkmanager.network.get_node(self._path['nodes'][nodepos]).ports[f"ccon_L_{self._path['nodes'][nodepos]}_{self._path['request']}_teleport"].bind_input_handler(self._handle_message,tag_meta=True)    

    def _handle_message(self,msg):
        input_port = msg.meta['rx_port_name']
        forward_port = input_port.replace('ccon_L_','ccon_R_')
        port_elements = input_port.split('_')
        node = self._networkmanager.network.get_node(port_elements[2])
        node.ports[forward_port].tx_output(msg)
        return
    
    def run(self):
        self.start_subprotocols()

        #Though in this simulations positions in nodes are always 0, we query in case this is changed in the future
        first_link = self._path['comms'][0]['links'][0]
        last_link = self._path['comms'][-1]['links'][0]
        mem_posA_1 = self._networkmanager.get_mem_position(self._path['nodes'][0],first_link.split('-')[0],first_link.split('-')[1])
        mem_posB_1 = self._networkmanager.get_mem_position(self._path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])
        mem_posTeleport = 2

        #Init measurement program
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

        set_qstate_formalism(QFormalism.KET)
        qubit = create_qubits(1)

        first_node = self._networkmanager.network.get_node(self._path['nodes'][0])
        last_node = self._networkmanager.network.get_node(self._path['nodes'][-1])

        num_qubits = len(self._qubits) #Number of qubits to teleport
        tx_qubit = 0 #Position of qubit to transmit

        while True:
            #TODO: Implementar el demand rate. Ahora mismo lo hace de forma continua, sin seguir distribución dada

            #Get state to teleport. First normalize it
            alpha = complex(self._qubits[tx_qubit][0])
            beta = complex(self._qubits[tx_qubit][1])
            norm = cmath.sqrt(np.conjugate(alpha)*alpha + np.conjugate(beta)*beta)
            state = np.array([[alpha], [beta]], dtype=complex)/norm

            #Set position of next qubit to transmit
            tx_qubit = tx_qubit + 1 if tx_qubit < num_qubits-1 else 0
            
            assign_qstate(qubit, state)

            #If position is not being used, we can store the qubit
            if 2 in self._networkmanager.network.get_node(self._path['nodes'][0]).qmemory.unused_positions:
                #store qubit un memory position
                first_node.qmemory.put(qubit, 2, replace = False)

                start_time = sim_time()
                #Request entanglement to RouteProtocol
                self.send_signal(self._ent_request)

                #Wait for  entanglement to be generated on both ends
                yield self.await_signal(self.subprotocols[f"RouteProtocol_{self._path['request']}"],Signals.SUCCESS)

                #Measure in Bell basis positions 0 and 2
                yield first_node.qmemory.execute_program(self._program, qubit_mapping=[mem_posTeleport,mem_posA_1])
                m, = self._program.output["m"]

                # Send result to right node on end
                first_node.ports[f"ccon_R_{self._path['nodes'][0]}_{self._path['request']}_teleport"].tx_output(Message(m))

                #Wait for Teleportation to complete
                yield self.await_signal(self.subprotocols[f"TeleportCorrectProtocol_{self._path['request']}"],Signals.SUCCESS)

                result_qubit, = last_node.qmemory.pop(0)
                fid = qapi.fidelity(result_qubit, state, squared = True)
                qapi.discard(result_qubit)
                result = {
                    'posA': mem_posA_1,
                    'posB': mem_posB_1,
                    'fid': fid,
                    'time': sim_time() - start_time
                }
 
                #send result to datacollector
                self.send_signal(Signals.SUCCESS, result)
            else:
                self.await_timer(1000)

class RateTeleportationApplication(GeneralApplication):

    def __init__(self, path, networkmanager, name=None):
        name = name if name else f"RateTeleportationApplication_Unidentified"
        super().__init__(path, networkmanager,)
    
    def run(self):
        pass


class TeleportCorrectProtocol(NodeProtocol):
    """Perform corrections for a swap on an end-node.
    Adapted from NetSquid web examples

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    num_nodes : int
        Number of nodes in the repeater chain network.

    """
    def __init__(self, node, mempos, name, request,epr_state):
        super().__init__(node, name)
        self._mempos = mempos
        self._request = request
        self._epr_state = epr_state

        self._x_corr = 0
        self._z_corr = 0

        self._program = SwapCorrectProgram()
   

    def run(self):
        while True:
            #Wait for a classical signal to arrive or a request from main protocol to restart
            yield self.await_port_input(self.node.ports[f"ccon_L_{self.node.name}_{self._request}_teleport"]) 
            
            message = self.node.ports[f"ccon_L_{self.node.name}_{self._request}_teleport"].rx_input()

            if message is not None:
                m = message.items[0]
                if self._epr_state == 'PHI_PLUS':
                    if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                        self._x_corr += 1
                    if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                        self._z_corr += 1
                else:
                    if m == ks.BellIndex.B10 or m == ks.BellIndex.B00:
                        self._x_corr += 1
                    if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                        self._z_corr += 1
                        
                if self._x_corr or self._z_corr:
                    self._program.set_corrections(self._x_corr, self._z_corr)
                    if self.node.qmemory.busy:
                        yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[self._mempos])
                
                self.send_signal(Signals.SUCCESS)
                self._x_corr = 0
                self._z_corr = 0
                