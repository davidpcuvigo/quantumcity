import numpy as np
import netsquid as ns
from netsquid.components.models import QuantumErrorModel
from random import gauss

class FibreDepolarizeModel(QuantumErrorModel):
    """
    Custom depolarization model, empirically obtained from https://arxiv.org/abs/0801.3620.
    It uses polarization mode dispersion time to evaluate the probability of depolarization.


    """
    def __init__(self):
        super().__init__()
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Uses the length property to calculate a depolarization probability,
        and applies it to the qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Not used.

        """
        for qubit in qubits:
            dgd=0.6*np.sqrt(float(kwargs['length'])/50)
            tau=gauss(dgd,dgd)
            tdec=1.6
            if tau >= tdec:
                prob=1
            elif tau < tdec:
                prob=0
            ns.qubits.depolarize(qubit, prob=prob)