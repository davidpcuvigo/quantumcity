
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
import netsquid as ns
from netsquid.qubits import ketstates as ks

class FidelityProtocol(LocalProtocol):
            
    def __init__(self, networkmanager, origin, dest, link, qsource_index, name=None):
        self._origin = origin
        self._dest = dest
        self._link = link
        self._networkmanager = networkmanager
        self._qsource_index = qsource_index
        name = name if name else f"FidelityEstimator_{origin.name}_{dest.name}"
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
        