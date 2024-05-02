from utils import dc_setup
from netsquid.protocols import LocalProtocol, NodeProtocol, Signals
from icecream import ic
from netsquid.util.simtools import sim_time
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits import ketstates as ks
from routing_protocols import SwapProtocol, CorrectProtocol, DistilProtocol
from pydynaa import EventExpression, EventType

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
        name = name if name else f"TeleportApplication_Unidentified"
        super().__init__(path, networkmanager,)
    
    def run(self):
        pass


class RouteProtocol(LocalProtocol):
    '''
    TODO: REVISAR ESTE PROTOCOLO. ES CASI IGUAL AL PATHFIDELITYESTIMATOR
    Diffs:
        Does not execute for a fixed number of times. Instead, waits for a signal to be received
        Does not measure qubits in memory for fidelity calculation
        Checks for purif_rounds and creates all channels if it is greater than 0
    '''

    def __init__(self, networkmanager, path, start_expression, purif_rounds= 0, name=None):
        self._path = path
        self._networkmanager = networkmanager
        self.start_expression = start_expression
        self._purif_rounds = purif_rounds
        name = name if name else f"RouteProtocol_{path['request']}"
        super().__init__(nodes=networkmanager.network.nodes, name=name)
        first_link = self._path['comms'][0]['links'][0]
        last_link = self._path['comms'][-1]['links'][0]
        self._mem_posA_1 = self._networkmanager.get_mem_position(self._path['nodes'][0],first_link.split('-')[0],first_link.split('-')[1])
        self._mem_posB_1 = self._networkmanager.get_mem_position(self._path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])
        self._portleft_1 = self._networkmanager.network.get_node(self._path['nodes'][0]).qmemory.ports[f"qin{self._mem_posA_1}"]

        # add purification signals
        #start purification signal
        self._start_purif_signal = 'START_PURIFICATION'
        self.add_signal(self._start_purif_signal)
        #end purification signal: Distil protocol will send 0 if purification successful or 1 if failed
        self._purif_result_signal = 'PURIF_DONE'
        self.add_signal(self._purif_result_signal)
        #Qubit lost when qchannel model has losses
        self._evtypetimer = EventType("Timer","Qubit is lost")
        #add correct protocol restart signal. Needed when purification is used and one quit is lost
        self._restart_signal = 'RESTART_CORRECT_PROTOCOL'
        self.add_signal(self._restart_signal)

        # preparation of entanglement swaping from second to the last-1
        for nodepos in range(1,len(path['nodes'])-1):
            node = path['nodes'][nodepos]
            link_left = path['comms'][nodepos-1]['links'][0]
            link_right = path['comms'][nodepos]['links'][0]
            mem_pos_left = networkmanager.get_mem_position(node,link_left.split('-')[0],link_left.split('-')[1])
            mem_pos_right = networkmanager.get_mem_position(node,link_right.split('-')[0],link_right.split('-')[1])
            subprotocol = SwapProtocol(node=networkmanager.network.get_node(node), mem_left=mem_pos_left, mem_right=mem_pos_right, name=f"SwapProtocol_{node}_{path['request']}_1", request = path['request'])
            self.add_subprotocol(subprotocol)

        # preparation of correct protocol in final node
        epr_state =  self._networkmanager.get_config('epr_pair','epr_pair')
        mempos= networkmanager.get_mem_position(path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])
        restart_expr = self.await_signal(self,self._restart_signal)
        subprotocol = CorrectProtocol(networkmanager.network.get_node(path['nodes'][-1]), mempos, len(path['nodes']), f"CorrectProtocol_{path['request']}_1", path['request'],restart_expr, epr_state)
        self.add_subprotocol(subprotocol)

        if purif_rounds > 0:
            first_link = self._path['comms'][0]['links'][1]
            last_link = self._path['comms'][-1]['links'][1]
            self._mem_posA_2 = self._networkmanager.get_mem_position(self._path['nodes'][0],first_link.split('-')[0],first_link.split('-')[1])
            self._mem_posB_2 = self._networkmanager.get_mem_position(self._path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])
            self._portleft_2 = self._networkmanager.network.get_node(self._path['nodes'][0]).qmemory.ports[f"qin{self._mem_posA_2}"]

            #add SwapProtocol in second instance of link
            for nodepos in range(1,len(self._path['nodes'])-1):
                node = self._path['nodes'][nodepos]
                link_left = self._path['comms'][nodepos-1]['links'][1]
                link_right = self._path['comms'][nodepos]['links'][1]
                mem_pos_left = self._networkmanager.get_mem_position(node,link_left.split('-')[0],link_left.split('-')[1])
                mem_pos_right = self._networkmanager.get_mem_position(node,link_right.split('-')[0],link_right.split('-')[1])
                subprotocol = SwapProtocol(node=self._networkmanager.network.get_node(node), mem_left=mem_pos_left, mem_right=mem_pos_right, name=f"SwapProtocol_{node}_{self._path['request']}_2", request = self._path['request'])
                self.add_subprotocol(subprotocol)

            #add Classical protocols for second instance of link
            epr_state = epr_state =  self._networkmanager.get_config('epr_pair','epr_pair')
            mempos= self._networkmanager.get_mem_position(self._path['nodes'][-1],last_link.split('-')[0],last_link.split('-')[1])
            restart_expr = self.await_signal(self,self._restart_signal)
            subprotocol = CorrectProtocol(self._networkmanager.network.get_node(self._path['nodes'][-1]), mempos, len(self._path['nodes']), f"CorrectProtocol_{self._path['request']}_2", self._path['request'],restart_expr, epr_state)
            self.add_subprotocol(subprotocol)

            nodeA = self._networkmanager.network.get_node(self._path['nodes'][0])
            nodeB = self._networkmanager.network.get_node(self._path['nodes'][-1])
    
            #Distil will wait for START_PURIFICATION signal
            start_expression = self.await_signal(self, self._start_purif_signal)
            self.add_subprotocol(DistilProtocol(nodeA, nodeA.ports[f"ccon_distil_{nodeA.name}_{self._path['request']}"],
                'A',self._mem_posA_1,self._mem_posA_2,start_expression, msg_header='distil', name=f"DistilProtocol_{nodeA.name}_{self._path['request']}"))
            self.add_subprotocol(DistilProtocol(nodeB, nodeB.ports[f"ccon_distil_{nodeB.name}_{self._path['request']}"],
                'B',self._mem_posB_1,self._mem_posB_2,start_expression, msg_header='distil',name=f"DistilProtocol_{nodeB.name}_{self._path['request']}"))


        # calculate total distance and delay, in order to set timer
        self._total_delay = 0
        for comm in path['comms']:
            link_name = comm['links'][0].split('-')[0]
            distance = float(networkmanager.get_config('links',link_name,'distance'))
            photon_speed = float(networkmanager.get_config('links',link_name,'photon_speed_fibre'))
            self._total_delay += 1e9 * distance / photon_speed
        #We need to add some nanoseconds to timer. If distances are short we 
        # can receive a lost qubit signal when it is not correct
        self._total_delay += 50000    

    def signal_sources(self,index=[1]):
        '''
        Signals all sources in the path in order to generate EPR
        Receives the index to trigger the generation. If none, only first instance will be triggered
        If index=[1,2] then both instances are signaled (purification)
        '''
        if index not in [[1],[2],[1,2]]:
            raise ValueError('Unsupported trigger generation')
        for link in self._path['comms']:
            trigger_node = self._networkmanager.network.get_node(link['source'])
            for i in index:
                trigger_link = link['links'][i-1].split('-')[0]
                trigger_link_index = link['links'][i-1].split('-')[1]
                trigger_node.subcomponents[f"qsource_{trigger_node.name}_{trigger_link}_{trigger_link_index}"].trigger()


    def run(self):
        self.start_subprotocols()
        #set event type in order to detect qubit losses
        evexpr_timer = EventExpression(source=self, event_type=self._evtypetimer)

        #Get type of EPR to use
        epr_state = ks.b00 if self._networkmanager.get_config('epr_pair','epr_pair') == 'PHI_PLUS' else ks.b01

        #for i in range(self._num_runs):
        while True:
            #ic(f"{self.name} en ejecución")
            yield self.start_expression
            round_done = False
            start_time = sim_time()
            while not round_done: #need to repeat in case qubit is lost
                if self._purif_rounds == 0:
                    #trigger all sources in the path
                    self.signal_sources(index=[1])
                    timer_event = self._schedule_after(self._total_delay, self._evtypetimer)

                    evexpr_protocol = (self.await_port_input(self._portleft_1)) & \
                        (self.await_signal(self.subprotocols[f"CorrectProtocol_{self._path['request']}_1"], Signals.SUCCESS))
                    #if timer is triggered, qubit has been lost in a link. Else entanglement
                    # swapping has succeeded
                    evexpr = yield evexpr_timer | evexpr_protocol
                    if evexpr.second_term.value: #swapping ok
                        timer_event.unschedule()
                        round_done = True
                    else:
                        #qubit is lost, must restart
                        ic(f"{self.name} Lost qubit")
                        #restart correction protocol
                        self.send_signal(self._restart_signal)
                        #repeat round
                        continue

                else: #we have to perform purification
                    purification_done = False
                    while not purification_done:
                        pur_round = 0
                        while (pur_round <= self._purif_rounds):# and (qubit_lost == False):
                            if pur_round == 0: #First round
                                #trigger all sources in the path
                                self.signal_sources(index=[1,2])

                                evexpr_protocol = (self.await_port_input(self._portleft_1) & \
                                    self.await_signal(self.subprotocols[f"CorrectProtocol_{self._path['request']}_1"], Signals.SUCCESS) &\
                                    self.await_port_input(self._portleft_2) & \
                                    self.await_signal(self.subprotocols[f"CorrectProtocol_{self._path['request']}_2"], Signals.SUCCESS))

                                timer_event = self._schedule_after(self._total_delay, self._evtypetimer)

                            else: #we keep the qubit in the first link and trigger EPRs in the second
                                #trigger all sources in the path
                                self.signal_sources(index=[2])

                                #Wait for qubits in both links and corrections in both
                                evexpr_protocol = (self.await_port_input(self._portleft_2) & \
                                    self.await_signal(self.subprotocols[f"CorrectProtocol_{self._path['request']}_2"], Signals.SUCCESS))

                                timer_event = self._schedule_after(self._total_delay, self._evtypetimer)

                            #Wait for qubits in both links and corrections in both or timer is over
                            evexpr_proto = yield evexpr_timer | evexpr_protocol

                            if evexpr_proto.second_term.value: #swapping ok
                                #unchedule timer
                                timer_event.unschedule()
                                
                                #trigger purification
                                self.send_signal(self._start_purif_signal, 0)
    
                                #wait for both ends to finish purification
                                expr_distil = yield (self.await_signal(self.subprotocols[f"DistilProtocol_{self._path['nodes'][0]}_{self._path['request']}"], self._purif_result_signal) &
                                    self.await_signal(self.subprotocols[f"DistilProtocol_{self._path['nodes'][-1]}_{self._path['request']}"], self._purif_result_signal))

                                source_protocol1 = expr_distil.second_term.atomic_source
                                ready_signal1 = source_protocol1.get_signal_by_event(
                                    event=expr_distil.second_term.triggered_events[0], receiver=self)
                                source_protocol2 = expr_distil.second_term.atomic_source
                                ready_signal2 = source_protocol2.get_signal_by_event(
                                    event=expr_distil.second_term.triggered_events[0], receiver=self)
                                
                                #if both SUCCESS signals have result 0, purification has succeeded
                                #if any has value not equal to cero, purification must be restarted
                                if ready_signal1.result == 0 and ready_signal2.result ==0:
                                    purification_done = True
                                else:
                                    #self.start_subprotocols()
                                    #restart purification from beggining
                                    purification_done = False
                                    break 
                            else: 
                                #qubit is lost, must restart round
                                ic(f"{self.name} Lost qubit")
                                #restart correction protocol
                                self.send_signal(self._restart_signal)

                                #restart purification from beggining
                                purification_done = False
                                break
                            
                            #so far purification protocol is ok, next purification round
                            pur_round += 1

                        #if we get to this point, we have ended the fidelity estimation round
                        round_done = True

            #round is done we measure fidelity
            if round_done:# and purification_done:
                self.send_signal(Signals.SUCCESS)