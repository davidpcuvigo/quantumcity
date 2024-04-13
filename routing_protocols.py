
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
import netsquid as ns
from netsquid.qubits import ketstates as ks
from icecream import ic
from netsquid.components import Message, QuantumProgram
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z
from netsquid.util.simtools import sim_time
from netsquid.qubits import qubitapi as qapi

class LinkFidelityProtocol(LocalProtocol):
            
    def __init__(self, networkmanager, origin, dest, link, qsource_index, name=None):
        self._origin = origin
        self._dest = dest
        self._link = link
        self._networkmanager = networkmanager
        self._qsource_index = qsource_index
        name = name if name else f"LinkFidelityEstimator_{origin.name}_{dest.name}"
        self._memory_left = networkmanager.get_mem_position(self._origin.name, self._link, self._qsource_index)
        self._memory_right = networkmanager.get_mem_position(self._dest.name, self._link, self._qsource_index)
        self._portleft = self._origin.qmemory.ports[f"qin{self._memory_left}"]
        self._portright = self._dest.qmemory.ports[f"qin{self._memory_right}"]
        self.fidelities = []
        super().__init__(nodes={"A": origin, "B": dest}, name=name)

    def run(self):
        #Signal Qsource to start. Must trigger correct source
        trig_origin = self._origin if self._networkmanager.get_config('nodes',self._origin.name,'type') == 'switch' else self._dest
        trig_origin.subcomponents[f"qsource_{trig_origin.name}_{self._link}_0"].trigger()

        while True:
            yield (self.await_port_input(self._portleft) & self.await_port_input(self._portright))
            qubit_a, = self._origin.qmemory.peek([self._memory_left])
            qubit_b, = self._dest.qmemory.peek([self._memory_right])
            self.fidelities.append(ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True))
            #self.send_signal(Signals.SUCCESS, 0)
            trig_origin.subcomponents[f"qsource_{trig_origin.name}_{self._link}_0"].trigger()

class PathFidelityProtocol(LocalProtocol):

    def __init__(self, networkmanager, path, num_runs, name=None):
        self._path = path
        self._num_runs = num_runs
        self._networkmanager = networkmanager
        name = name if name else f"PathFidelityEstimator_{path['request']}"
        super().__init__(nodes=networkmanager.network.nodes, name=name)
        first_link = self._path['comms'][0]['links'][0]
        last_link = self._path['comms'][-1]['links'][0]
        self._mem_posA = self._networkmanager.get_mem_position(self._path['nodes'][0],first_link.split('-')[0],first_link.split('-')[1])
        self._mem_posB = self._networkmanager.get_mem_position(self._path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])
        self._portleft = networkmanager.network.get_node(path['nodes'][0]).qmemory.ports[f"qin{self._mem_posA}"]

        for nodepos in range(1,len(path['nodes'])-1):
            node = path['nodes'][nodepos]
            link_left = path['comms'][nodepos-1]['links'][0]
            link_right = path['comms'][nodepos]['links'][0]
            mem_pos_left = networkmanager.get_mem_position(node,link_left.split('-')[0],link_left.split('-')[1])
            mem_pos_right = networkmanager.get_mem_position(node,link_right.split('-')[0],link_right.split('-')[1])
            subprotocol = SwapProtocol(node=networkmanager.network.get_node(node), mem_left=mem_pos_left, mem_right=mem_pos_right, name=f"Swap_{node}_{path['request']}", request = path['request'])
            self.add_subprotocol(subprotocol)
        last_link = path['comms'][-1]['links'][0]
        mempos= networkmanager.get_mem_position(path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])
        subprotocol = CorrectProtocol(networkmanager.network.get_node(path['nodes'][-1]), mempos, len(path['nodes']), path['request'])
        self.add_subprotocol(subprotocol)

    def signal_sources(self):
        '''
        Signals all sources in the path in order to generate EPR
        '''
        for link in self._path['comms']:
            trigger_node = self._networkmanager.network.get_node(link['source'])
            trigger_link = link['links'][0].split('-')[0]
            #JUAN CUANDO HAYA 2 POR PURIFICACIÓN HAY QUE HACERLO EN AMBOS
            trigger_link_index = link['links'][0].split('-')[1]
            trigger_node.subcomponents[f"qsource_{trigger_node.name}_{trigger_link}_{trigger_link_index}"].trigger()
            #ic(f"Señalizo qsource_{trigger_node.name}_{trigger_link}_{trigger_link_index}")

    def run(self):
        self.start_subprotocols()

        for i in range(self._num_runs):
            start_time = sim_time()
            #trigger all sources in the path
            self.signal_sources()
            #ic(f'{self.name}: Finalizo señalización fuentes, ronda {i}')
            yield (self.await_port_input(self._portleft)) & \
                (self.await_signal(self.subprotocols[f"CorrectProtocol_{self._path['request']}"], Signals.SUCCESS))
            #signal = self.subprotocols[f"CorrectProtocol_{self._path['request']}"].get_signa_result(
            #    label=Signals.SUCESS, receiver=self)
            #ic(f'{self.name}: Termino la espera de condiciones, ronda {i}. Cogeré de nodoA la posición {self._mem_posA} y del B la {self._mem_posB}')
            qa, = self._networkmanager.network.get_node(self._path['nodes'][0]).qmemory.pop(positions=[self._mem_posA])
            qb, = self._networkmanager.network.get_node(self._path['nodes'][-1]).qmemory.pop(positions=[self._mem_posB])
            fid = qapi.fidelity([qa, qb], ks.b00, squared=True)
            result = {
                'posA': self._mem_posA,
                'posB': self._mem_posB,
                'pairsA': 0,
                'pairsB': 0,
                'fid': fid,
                'time': sim_time() - start_time
            }
            #ic(f'{self.name}: result')
            self.send_signal(Signals.SUCCESS, result)


class SwapProtocol(NodeProtocol):
    """Perform Swap on a repeater node.
    Adapted from NetSquid web examples

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """

    def __init__(self, node, mem_left, mem_right, name, request):
        super().__init__(node, name)
        self._request = request
        self._mem_left = mem_left
        self._mem_right = mem_right
        self._qmem_input_port_l = self.node.qmemory.ports[f"qin{mem_left}"]
        self._qmem_input_port_r = self.node.qmemory.ports[f"qin{mem_right}"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):
        while True:
            #ic(f'{self.name}: Antes')
            yield (self.await_port_input(self._qmem_input_port_l) &
                   self.await_port_input(self._qmem_input_port_r))
            # Perform Bell measurement
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[self._mem_right, self._mem_left])
            m, = self._program.output["m"]
            #ic(f'{self.name}: Mido {m}')
            # Send result to right node on end
            self.node.ports[f"ccon_R_{self.node.name}_{self._request}_1"].tx_output(Message(m))
            #ic(f'{self.name}: Envío mensage {m}')

class SwapCorrectProgram(QuantumProgram):
    """Quantum processor program that applies all swap corrections."""
    default_num_qubits = 1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(INSTR_Z, q1)
        yield self.run()


class CorrectProtocol(NodeProtocol):
    """Perform corrections for a swap on an end-node.
    Adapted from NetSquid web examples

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    num_nodes : int
        Number of nodes in the repeater chain network.

    """
    def __init__(self, node, mempos, num_nodes, request):
        super().__init__(node, f"CorrectProtocol_{request}")
        self._mempos = mempos
        self.num_nodes = num_nodes
        self._request = request
        self._x_corr = 0
        self._z_corr = 0
        self._program = SwapCorrectProgram()
        self._counter = 0

    def run(self):
        while True:
            #ic(f'{self.name}:Antes')
            yield self.await_port_input(self.node.ports[f"ccon_L_{self.node.name}_{self._request}_{1}"])
            #ic(f'{self.name}:Después')
            message = self.node.ports[f"ccon_L_{self.node.name}_{self._request}_{1}"].rx_input()
            #ic(f'{self.name}: He recibido {message}')
            if message is None: #or len(message.items) != 1:
                continue
            else: #Port can receive more than one classical message at the same time
                for m in message.items:
                    #m = message.items[0]
                    if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                        self._x_corr += 1
                    if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                        self._z_corr += 1
                    self._counter += 1

            if self._counter == self.num_nodes - 2:
                if self._x_corr or self._z_corr:
                    self._program.set_corrections(self._x_corr, self._z_corr)
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[self._mempos])
                self.send_signal(Signals.SUCCESS)
                self._x_corr = 0
                self._z_corr = 0
                self._counter = 0
