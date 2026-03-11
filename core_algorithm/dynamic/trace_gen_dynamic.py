import configparser
import os
import random
import subprocess
from typing import List, Set, Dict

# import skg_main.skg_mgrs.connector_mgr as conn
# from skg_main.skg_mgrs.skg_reader import Skg_Reader
# from skg_main.skg_model.schema import Entity
# from skg_main.skg_model.schema import Timestamp as skg_Timestamp
# from skg_main.skg_model.semantics import EntityForest, EntityTree

from core_algorithm.lsha.sha_learning.domain.lshafeatures import Trace, Event
from core_algorithm.lsha.sha_learning.learning_setup.logger import Logger

CS = 'Thermostat'
CS_VERSION = '1'
UPP_EXE_PATH = ''
UPP_OUT_PATH = ''
SCRIPT_PATH = ''
SIM_LOGS_PATH = ''
UPP_MODEL_PATH = ''
UPP_QUERY_PATH = ''
RESAMPLE_STRATEGY = 'UPPAAL'
MAX_E = 15

LOGGER = Logger('TRACE GENERATOR')


class TraceGenerator:
    def __init__(self, word: Trace = Trace([]), pov: str = None,
                 start_dt: str = None, end_dt: str = None, start_ts: str = None, end_ts: str = None,
                 config_data: Dict = None):
        self.word = word
        self.events: List[Event] = word.events
        self.evt_int: List[int] = []

        self.ONCE = False
        self.processed_traces: Set[str] = set()
        
        # Configuration Injection
        self.config_data = config_data if config_data else {}
        
        self.CS = self.config_data.get('CS', CS)
        self.CS_VERSION = str(self.config_data.get('CS_VERSION', CS_VERSION))
        self.UPP_EXE_PATH = self.config_data.get('UPP_EXE_PATH', UPP_EXE_PATH)
        self.UPP_OUT_PATH = self.config_data.get('UPP_OUT_PATH', UPP_OUT_PATH)
        self.SCRIPT_PATH = self.config_data.get('SCRIPT_PATH', SCRIPT_PATH)
        self.UPP_MODEL_PATH = self.config_data.get('UPP_MODEL_PATH', UPP_MODEL_PATH)
        self.UPP_QUERY_PATH = self.config_data.get('UPP_QUERY_PATH', UPP_QUERY_PATH)
        self.RESAMPLE_STRATEGY = self.config_data.get('RESAMPLE_STRATEGY', RESAMPLE_STRATEGY)
        self.SIM_LOGS_PATH = self.config_data.get('SIM_LOGS_PATH', SIM_LOGS_PATH)
        self.MAX_E = self.config_data.get('MAX_E', MAX_E)

        # Initialize lines based on CS
        if self.CS == 'HRI':
            self.LINE_1 = ['bool force_exe = true;\n', 'bool force_exe']
            self.LINE_2 = ['int force_act[MAX_E] = ', 'int force_act']
            self.LINE_3 = ['const int TAU = {};\n', 'const int TAU']
            self.LINE_4 = ['amy = HFoll_{}(1, 48, 2, 3, -1);\n', 'amy = HFoll_']
            self.LINE_5 = ['const int VERSION = {};\n', 'const int VERSION']
            self.LINES_TO_CHANGE = [self.LINE_1, self.LINE_2, self.LINE_3, self.LINE_4, self.LINE_5]
        else:
            self.LINE_1 = ['bool force_exe = true;\n', 'bool force_exe']
            self.LINE_2 = ['int force_open[MAX_E] = ', 'int force_open']
            self.LINE_3 = ['const int TAU = {};\n', 'const int TAU']
            self.LINE_4 = ['r = Room_{}(15.2);\n', 'r = Room']
            self.LINES_TO_CHANGE = [self.LINE_1, self.LINE_2, self.LINE_3, self.LINE_4]

        if self.RESAMPLE_STRATEGY == 'SKG':
            self.labels_hierarchy: List[List[str]] = []
            self.processed_entities: Dict[Entity, EntityTree] = {}
            self.pov = pov
            self.start_dt = start_dt
            self.end_dt = end_dt
            self.start_ts = start_ts
            self.end_ts = end_ts

    def set_word(self, w: Trace):
        self.events = w.events
        self.evt_int = []
        self.word = w

    def evts_to_ints(self):
        for e in self.events:
            if self.CS == 'HRI':
                if e.symbol in ['u_2', 'u_4']:
                    self.evt_int.append(1)
                elif e.symbol in ['u_3']:
                    self.evt_int.append(3)
                elif e.symbol in ['d_3', 'd_4']:
                    self.evt_int.append(0)
                elif e.symbol in ['d_2']:
                    self.evt_int.append(2)
                else:
                    self.evt_int.append(-1)
            else:
                # for thermo example: associates a specific value
                # to variable open for each event in the requested trace
                if int(self.CS_VERSION) < 8:
                    if e.symbol in ['h_0', 'c_0']:
                        self.evt_int.append(0)
                    elif e.symbol in ['h_1', 'c_1']:
                        self.evt_int.append(1)
                    elif e.symbol in ['h_2', 'c_2']:
                        self.evt_int.append(2)
                else:
                    if e.symbol in ['h_0', 'c_0']:
                        self.evt_int.append(0)
                    elif e.symbol in ['h_1', 'c_1']:
                        self.evt_int.append(1)
                    elif e.symbol in ['h_2', 'c_2']:
                        self.evt_int.append(2)
                    elif e.symbol in ['h_3', 'c_3']:
                        self.evt_int.append(0)

    def get_evt_str(self):
        self.evts_to_ints()

        res = '{'
        i = 0
        for evt in self.evt_int:
            res += str(evt) + ', '
            i += 1
        while i < self.MAX_E - 1:
            res += '-1, '
            i += 1
        res += '-1};\n'
        return res

    def fix_model(self):
        # customized uppaal model based on requested trace
        m_r = open(self.UPP_MODEL_PATH, 'r')

        new_line_1 = self.LINE_1[0]
        values = self.get_evt_str()
        new_line_2 = self.LINE_2[0] + values
        tau = max(len(self.evt_int) * 50, 200)
        new_line_3 = self.LINE_3[0].format(tau)
        new_line_4 = self.LINE_4[0].format(self.CS_VERSION)
        new_line_5 = self.LINE_5[0].format(int(self.CS_VERSION) - 1) if self.CS == 'HRI' else None
        new_lines = [new_line_1, new_line_2, new_line_3, new_line_4, new_line_5]

        lines = m_r.readlines()
        found = [False] * len(new_lines)
        for line in lines:
            for (i, l) in enumerate(self.LINES_TO_CHANGE):
                if line.startswith(self.LINES_TO_CHANGE[i][1]) and not found[i]:
                    lines[lines.index(line)] = new_lines[i]
                    found[i] = True
                    break

        m_r.close()
        m_w = open(self.UPP_MODEL_PATH, 'w')
        m_w.writelines(lines)
        m_w.close()

    def get_traces(self, n: int = 1):
        if self.RESAMPLE_STRATEGY == 'UPPAAL':
            return self.get_traces_uppaal(n)


    def get_traces_uppaal(self, n: int):
        # sample new traces through uppaal command line tool
        self.fix_model()
        LOGGER.debug('!! GENERATING NEW TRACES FOR: {} !!'.format(self.word))
        new_traces: List[str] = []

        for i in range(n):
            random.seed()
            n = random.randint(0, 2 ** 32)
            s = '{}_{}_{}'.format(self.CS, self.CS_VERSION, n)
            FNULL = open(os.devnull, 'w')
            p = subprocess.Popen([self.SCRIPT_PATH, self.UPP_EXE_PATH, self.UPP_MODEL_PATH,
                                  self.UPP_QUERY_PATH, str(n), self.UPP_OUT_PATH.format(s)], stdout=FNULL)
            p.wait()
            if p.returncode == 0:
                LOGGER.info('TRACES SAVED TO ' + s)
                # returns out file where new traces are stored
                new_traces.append(UPP_OUT_PATH.format(s))

        return new_traces
