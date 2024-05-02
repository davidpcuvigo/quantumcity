from utils import dc_setup
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from icecream import ic
from netsquid.util.simtools import sim_time
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from protocols import RouteProtocol


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

    def __init__(self, path, networkmanager, name=None):
        name = name if name else f"TeleportApplication_Unidentified"
        super().__init__(path, networkmanager)
    
    def run(self):
        pass

class CircTeleportationApplication(GeneralApplication):

    def __init__(self, path, networkmanager, name=None):
        name = name if name else f"CircTeleportationApplication_Unidentified"
        super().__init__(path, networkmanager,)
    
    def run(self):
        pass