import pandas
import pydynaa
import functools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from random import uniform, gauss
import re
from configparser import ConfigParser
import ast

import netsquid as ns
from netsquid.qubits import ketstates as ks
from netsquid.components import Message, QuantumProcessor, QuantumProgram, PhysicalInstruction, ClassicalChannel
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel, QuantumErrorModel
from netsquid.components.instructions import INSTR_MEASURE_BELL, INSTR_X, INSTR_Z, INSTR_I, INSTR_SWAP
from netsquid.nodes import Node, Network
from netsquid.nodes.connections import DirectConnection
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from netsquid.util.datacollector import DataCollector
from netsquid.examples.teleportation import EntanglingConnection, ClassicalConnection
#from netsquid.examples.purify import Distil, Filter
from netsquid.util import simlog
from netsquid.util.constrainedmap import ConstrainedMapView, ValueConstraint, ConstrainedMap
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
#from netsquid.examples.entanglenodes import EntangleNodes
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.component import Message, Port
import netsquid.qubits.operators as ops
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate
from pydynaa import EventExpression
from network import NetworkManager

__all__ = [
    "FidelityObtainer",
    "SwapProtocol",
    "SwapProtocol2",
    "SwapCorrectProgram",
    "CorrectProtocol"
]

SWITCH_NODE_BASENAME = "switch_node_"
END_NODE_BASENAME = "end_node_"      
        
class EntangleNode(NodeProtocol):
    def __init__(self, node=None, name=None, start_expression=None):
        super().__init__(node, name)
        self.start_expression=start_expression #This start_expression is actually defined with the start_on_signal function, created for convenience
        self.node=node
    def start(self):
        super().start()
        
    def stop(self):
        super().stop()
        
    def run(self):
        while True:
            yield self.start_expression #We wait for any of the signals we defined for the protocol to start
            qsource, = [item for item in self.node.subcomponents.items if 'qsource' in item.name] #We search for a qsource in the node
            qsource.trigger()
            self.send_signal(Signals.FINISHED)

class RouteProtocol(LocalProtocol):
    def __init__(self, nodes=None, name=None, max_nodes=-1, start_expression=None):
        super().__init__(nodes, name, max_nodes)
        self.start_expression=start_expression #This start_expression is actually defined with the start_on_signal function, created for convenience
        self.nodes=nodes
    def start(self):
        super().start()
        
    def stop(self):
        super().stop()
        
    def run(self):
        while True:
            yield self.start_expression #We wait for any of the signals we defined for the protocol to start
            qsource, = [item for item in self.node.subcomponents.items if 'qsource' in item.name] #We search for a qsource in the node
            qsource.trigger()
            self.send_signal(Signals.FINISHED)
 
class DistilProtocol(NodeProtocol):
    """Protocol that does local DEJMPS distillation on a node.

    This is done in combination with another node.

    Parameters
    ----------
    node : :py:class:`~netsquid.nodes.node.Node`
        Node with a quantum memory to run protocol on.
    port : :py:class:`~netsquid.components.component.Port`
        Port to use for classical IO communication to the other node.
    role : "A" or "B"
        Distillation requires that one of the nodes ("B") conjugate its rotation,
        while the other doesn't ("A").
    start_expression : :class:`~pydynaa.EventExpression`
        EventExpression node should wait for before starting distillation.
        The EventExpression should have a protocol as source, this protocol should signal the quantum memory position
        of the qubit.
    msg_header : str, optional
        Value of header meta field used for classical communication.
    name : str or None, optional
        Name of protocol. If None a default name is set.

    """
    # set basis change operators for local DEJMPS step
    _INSTR_Rx = IGate("Rx_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0)))
    _INSTR_RxC = IGate("RxC_gate", ops.create_rotation_op(np.pi / 2, (1, 0, 0), conjugate=True))

    def __init__(self, node, port, role, start_expression=None, msg_header="distil", name=None, rounds= 5):
        if role.upper() not in ["A", "B"]:
            raise ValueError
        conj_rotation = role.upper() == "B"
        if not isinstance(port, Port):
            raise ValueError("{} is not a Port".format(port))
        name = name if name else "DistilNode({}, {})".format(node.name, port.name)
        super().__init__(node, name=name)
        self.port = port
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression
        self._program = self._setup_dejmp_program(conj_rotation)
        # self.INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self.header = msg_header
        self._qmem_positions = [None, None]
        self._waiting_on_second_qubit = False
        self.total_rounds= rounds
        self.round=0
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def _setup_dejmp_program(self, conj_rotation):
        INSTR_ROT = self._INSTR_Rx if not conj_rotation else self._INSTR_RxC
        prog = QuantumProgram(num_qubits=2)
        q1, q2 = prog.get_qubit_indices(2)
        prog.apply(INSTR_ROT, [q1])
        prog.apply(INSTR_ROT, [q2])
        prog.apply(INSTR_CNOT, [q1, q2])
        prog.apply(INSTR_MEASURE, q2, output_key="m", inplace=False)
        return prog

    def run(self):
        cchannel_ready = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            while True:
                expr = yield cchannel_ready | qmemory_ready
                if expr.first_term.value:
                    classical_message = self.port.rx_input(header=self.header)
                    if classical_message:
                        self.remote_qcount, self.remote_meas_result = classical_message.items
                elif expr.second_term.value:
                    source_protocol = expr.second_term.atomic_source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=expr.second_term.triggered_events[0], receiver=self)
                    print(ready_signal.result, self.node.name, self.round)
                    yield from self._handle_new_qubit(ready_signal.result)
                if self.round >= 3 and self.local_meas_result != self.remote_meas_result:
                    break
            print('hola')
            self._check_success()

    def start(self):
        # Clear any held qubits
        self._clear_qmem_positions()
        self.local_qcount = 0
        self.local_meas_result = None
        self.remote_qcount = 0
        self.remote_meas_result = None
        self._waiting_on_second_qubit = False
        self.rounds= 0
        return super().start()

    def _clear_qmem_positions(self):
        positions = [pos for pos in self._qmem_positions if pos is not None]
        if len(positions) > 0:
            self.node.qmemory.pop(positions=positions)
        self._qmem_positions = [None, None]

    def _handle_new_qubit(self, memory_position):
        # Process signalling of new entangled qubit
        assert not self.node.qmemory.mem_positions[memory_position].is_empty
        if self._waiting_on_second_qubit:
            # Second qubit arrived: perform distil
            #print(self._qmem_positions[0])
            assert not self.node.qmemory.mem_positions[self._qmem_positions[0]].is_empty
            assert memory_position != self._qmem_positions[0]
            self._qmem_positions[1] = memory_position
            self._waiting_on_second_qubit = True 
            yield from self._node_do_DEJMPS()
        else:
            # New candidate for first qubit arrived
            # Pop previous qubit if present:
            pop_positions = [p for p in self._qmem_positions if p is not None and p != memory_position]
            if len(pop_positions) > 0:
                self.node.qmemory.pop(positions=pop_positions)
            # Set new position:
            self._qmem_positions[0] = memory_position
            self._qmem_positions[1] = None
            self.local_qcount += 1
            self.local_meas_result = None
            self._waiting_on_second_qubit = True

    def _node_do_DEJMPS(self):
        # Perform DEJMPS distillation protocol locally on one node
        pos1, pos2 = self._qmem_positions
        if self.node.qmemory.busy:
            yield self.await_program(self.node.qmemory)
        # We perform local DEJMPS
        yield self.node.qmemory.execute_program(self._program, [pos1, pos2])  # If instruction not instant
        self.node.qmemory.pop(positions=pos2)
        self.local_meas_result = self._program.output["m"][0]
        #self._qmem_positions[1] = None
        # Send local results to the remote node to allow it to check for success.
        self.port.tx_output(Message([self.local_qcount, self.local_meas_result],
                                    header=self.header))
        self.send_signal(Signals.FAIL, self.local_qcount)
        self.round +=1
        
    def _check_success(self):
        # Check if distillation succeeded by comparing local and remote results
        if (self.local_qcount == self.remote_qcount and
                self.local_meas_result is not None and
                self.remote_meas_result is not None):
            if self.local_meas_result == self.remote_meas_result:
                # SUCCESS
                self.send_signal(Signals.SUCCESS, self._qmem_positions[0])
            else:
                # FAILURE
                self._clear_qmem_positions()
                self.send_signal(Signals.FAIL, self.local_qcount)
            self.local_meas_result = None
            self.remote_meas_result = None
            self._qmem_positions = [None, None]

    @property
    def is_connected(self):
        if self.start_expression is None:
            return False
        if not self.check_assigned(self.port, Port):
            return False
        if not self.check_assigned(self.node, Node):
            return False
        if self.node.qmemory.num_positions < 2:
            return False
        return True
                    
class DistilProtocol(LocalProtocol):
    def __init__(self, nodes=None, name=None, max_nodes=-1, start_expression=None):
        super().__init__(nodes, name, max_nodes)
        self.start_expression=start_expression #This start_expression is actually defined with the start_on_signal function, created for convenience
        self.nodes=nodes
    def start(self):
        super().start()
        
    def stop(self):
        super().stop()
        
    def run(self):
        while True:
            yield self.start_expression #We wait for any of the signals we defined for the protocol to start
            qsource, = [item for item in self.node.subcomponents.items if 'qsource' in item.name] #We search for a qsource in the node
            qsource.trigger()
            self.send_signal(Signals.FINISHED)
                     
class SwapProtocol(NodeProtocol):
    def __init__(self, node=None, name=None, start_expression=None, input_ports=None):
        super().__init__(node, name)
        self.start_expression=start_expression #This start_expression is actually defined with the start_on_signal function, created for convenience
        self.node=node        
        port_events= [self.await_port_input(port) for port in input_ports] #await for all input ports
        self._program=GHZMeasureProgram()
        self.combined_events = port_events[0] if len(port_events) == 1 else functools.reduce(lambda x, y: x & y, port_events) #combine the events with AND
    def start(self):
        super().start()
        
    def stop(self):
        super().stop()
        
    def run(self):
        while True:
            yield self.combined_events #We wait for any of the signals we defined for the protocol to start
            positions_to_measure = []
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=positions_to_measure)
            m = self._program.output["m"]
            # Send result to right node on end
            self.node.ports["ccon_R"].tx_output(Message(m))
                     
class CorrectProtocol(NodeProtocol):
    def __init__(self, node=None, name=None, start_expression=None):
        super().__init__(node, name)
        self.start_expression=start_expression #This start_expression is actually defined with the start_on_signal function, created for convenience
        self.node=node
    def start(self):
        super().start()
        
    def stop(self):
        super().stop()
        
    def run(self):
        while True:
            yield self.start_expression #We wait for any of the signals we defined for the protocol to start
            qsource, = [item for item in self.node.subcomponents.items if 'qsource' in item.name] #We search for a qsource in the node
            qsource.trigger()
            self.send_signal(Signals.FINISHED)
                     
class TeleportProtocol(NodeProtocol):
    def __init__(self, node=None, name=None, start_expression=None, role=None):
        if role.lower() not in ['sender','receiver']:
            raise ValueError
        super().__init__(node, name)
        self.start_expression=start_expression
        self.node=node
        self.role=role
    def start(self):
        super().start()
        
    def stop(self):
        super().stop()
        
    def run(self):
        while True:
            start_expressions= yield self.start_expression | self.await_port_input(self.node.ports["ccon_L"])            
            if start_expressions.first_term.value:
                #yield self.await_signal(RouteProtocol, signal_label=Signals.WAITING)
                state=ns.b00 #se puede cambiar en el futuro
            elif start_expressions.second_term.value:
                message=self.node.ports["ccon_L"].rx_input().items
                m, = message[0][0]
                if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                    self._x_corr += 1
                if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                    self._z_corr += 1
                self._counter += 1
                if self._counter == self.num_nodes - 2:
                    if self._x_corr or self._z_corr:
                        self._program.set_corrections(self._x_corr, self._z_corr)
                        position_a=3
                        yield self.node.qmemory.execute_program(self._program, qubit_mapping=[position_a])
                    self.send_signal(Signals.SUCCESS)               
                    self._x_corr = 0
                    self._z_corr = 0
                    self._counter = 0
                
class GHZMeasureProgram(QuantumProgram):
    """Performs a measurement in the GHZ basis.
    """
    connect_size=2
    default_num_qubits = connect_size

    @classmethod
    def get_correction_operators(cls, outcomes):
        """
        Parameters
        ----------
        outcomes : list of int

        Returns
        -------
        list of :obj:`netsquid.qubits.operators.Operator`
        """
        if len(outcomes) != cls.default_num_qubits:
            raise ValueError("Number of outcomes should equal" +
                                "number of measured qubits")
        control_qubit_correction = ops.Z if outcomes[0] == 1 else ops.I
        target_qubit_corrections = []
        for outcome in outcomes[1:]:
            target_qubit_correction = ops.X if outcome == 1 else ops.I
            target_qubit_corrections.append(target_qubit_correction)
        return [control_qubit_correction] + target_qubit_corrections

    def program(self):
        qubits = self.get_qubit_indices(connect_size)
        control_qubit = qubits[0]
        target_qubits = qubits[1:]
        for target_qubit in target_qubits:
            self.apply(INSTR_CNOT, [control_qubit, target_qubit])
        self.apply(INSTR_H, control_qubit)
        for qubit_index, qubit in enumerate(qubits):
            self.apply(INSTR_MEASURE,
                        qubit,
                        output_key='m{}'.format(qubit_index),
                        inplace=False)
        yield self.run()                
                
def start_on_signal(protocol, start_subprotocol, signaling_subprotocol, signal):
    # Function to set subprotocols' start expression to be controlled by another protocol's signal
    # signal:  netsquid.protocols.protocol.Signals object. Recommended: Signals.SUCCESS, Signals.WAITING, Signals.FAIL, Signals.FINISHED
    protocol.subprotocols[start_subprotocol].start_expression = (
        protocol.subprotocols[start_subprotocol].await_signal(
            protocol.subprotocols[signaling_subprotocol], signal))

def setup_protocols(network):
    paths=NetworkManager.paths #TODO instanciar la clase
    number_of_paths= 2 #TODO llamarla bien, algo como len(paths)
    for i,path in zip(range(number_of_paths),paths): #Each path gets a Local controller
        
        protocol=LocalProtocol(nodes=path) ###Este ha de ser el route?
        
        for node in path[:-1]:
            protocol.add_subprotocol(EntangleNode(node=node)) #Every node except the last one generates entanglement
            
        subprotocol_teleport1=TeleportProtocol(node=path[0],role='sender')
        subprotocol_teleport2=TeleportProtocol(node=path[-1],role='receiver')
        protocol.add_subprotocol(subprotocol_teleport1)
        protocol.add_subprotocol(subprotocol_teleport2)
        
        subprotocol_correct=CorrectProtocol(node=path[-1])
        protocol.add_subprotocol(subprotocol_correct)
        
        subprotocol_distil1=DistilProtocol(node=path[0],role='sender') #End nodes collaborate to achieve distilation of the quantum state
        subprotocol_distil2=DistilProtocol(node=path[-1],role='receiver')
        protocol.add_subprotocol(subprotocol_distil1)
        protocol.add_subprotocol(subprotocol_distil2)
        
        for node in path[1:-1]:
            protocol.add_subprotocol(SwapProtocol(node=node)) #Every switch is given swapping capabilities