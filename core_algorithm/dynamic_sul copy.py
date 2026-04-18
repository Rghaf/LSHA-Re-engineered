# Energy case study

import configparser
import csv
import os
from typing import List, Tuple

from sha_learning.domain.lshafeatures import Event, FlowCondition
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from sha_learning.learning_setup.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

try:
    CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
except ValueError:
    CS_VERSION = None
SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])
MIN_SPEED = int(config['ENERGY CS']['MIN_SPEED'])
MAX_SPEED = int(config['ENERGY CS']['MAX_SPEED'])

LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return abs(curr[0] - prev[0]) > SPEED_RANGE or curr[1] != prev[1]


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    speed_sig = signals[1]
    pressure_sig = signals[2]
    speed = {pt.timestamp: (i, pt.value) for i, pt in enumerate(speed_sig.points)}
    pressure = {pt.timestamp: (i, pt.value) for i, pt in enumerate(pressure_sig.points)}

    SPEED_INTERVALS: List[Tuple[int, int]] = []
    for i in range(MIN_SPEED, MAX_SPEED, SPEED_RANGE):
        if i < MAX_SPEED - SPEED_RANGE:
            SPEED_INTERVALS.append((i, i + SPEED_RANGE))
        else:
            SPEED_INTERVALS.append((i, None))

    curr_speed_index, curr_speed = speed[t]
    if curr_speed_index > 0:
        try:
            prev_index = [tup[0] for tup in speed.values() if tup[0] < curr_speed_index][-1]
            prev_speed = speed_sig.points[prev_index].value
        except IndexError:
            prev_speed = None
    else:
        prev_speed = curr_speed

    curr_press_index, curr_press = pressure[t]
    if curr_press_index > 0:
        try:
            prev_index = [tup[0] for tup in pressure.values() if tup[0] < curr_press_index][-1]
            prev_press = pressure_sig.points[prev_index].value
        except IndexError:
            prev_press = None
    else:
        prev_press = curr_press

    identified_event = None

    if curr_press != prev_press:
        if curr_press == 1.0 and prev_press == 0.0:
            identified_event = events[-2]
        else:
            identified_event = events[-1]
    # if spindle was moving previously and now it is idle, return "stop" event
    elif curr_speed < MIN_SPEED and (prev_speed is not None and prev_speed >= MIN_SPEED):
        identified_event = events[-3]
    else:
        # if spindle is now moving at a different speed than before,
        # return 'new speed' event, which varies depending on current speed range
        if prev_speed is None or abs(curr_speed - prev_speed) >= SPEED_RANGE:
            for i, interval in enumerate(SPEED_INTERVALS):
                if (i < len(SPEED_INTERVALS) - 1 and interval[0] <= curr_speed < interval[1]) or \
                        (i == len(SPEED_INTERVALS) - 1 and curr_speed >= interval[0]):
                    identified_event = events[i]

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: str):
    fields = ts.split(':')
    return Timestamp(0, 0, 0, int(fields[0]), int(fields[1]), int(fields[2]))


def parse_data(path: str):
    # support method to parse traces sampled by ref query
    power: SampledSignal = SampledSignal([], label='P')
    speed: SampledSignal = SampledSignal([], label='w')
    pressure: SampledSignal = SampledSignal([], label='pr')

    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        counter = 0

        for i, row in enumerate(reader):
            if i == 0:
                continue

            ts = parse_ts(row[2])

            if i > 1 and ts == speed.points[-1].timestamp:
                # parse power value
                power.points[-1].value = (power.points[-1].value * counter + float(row[4])) / (counter + 1)

                # parse speed value: round to closest [100]
                speed_v = round(float(row[3]) / 100) * 100
                speed.points[-1].value = min(speed_v, speed.points[-1].value)

                # parse pallet pressure value
                pressure_v = float(row[1] != 'UNLOAD')
                pressure.points[-1].value = min(pressure_v, pressure.points[-1].value)

                counter += 1
            else:
                counter = 0

                # parse power value
                power.points.append(SignalPoint(ts, float(row[4])))

                # parse speed value: round to closest [100]
                speed_v = round(float(row[3]) / 100) * 100
                speed.points.append(SignalPoint(ts, speed_v))

                # parse pallet pressure value
                pressure_v = float(not (row[1] == 'UNLOAD' or (row[1] == 'LOAD' and i == 1)))
                pressure.points.append(SignalPoint(ts, pressure_v))

        return [power, speed, pressure]


def get_power_param(segment: List[SignalPoint], flow: FlowCondition):
    sum_power = sum([pt.value for pt in segment])
    avg_power = sum_power / len(segment)
    return avg_power


# Green case study

import configparser
import os
from datetime import datetime
from typing import List

import pandas as pd

from sha_learning.domain.lshafeatures import Event, FlowCondition
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from sha_learning.learning_setup.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

try:
    CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', ''))
except ValueError:
    CS_VERSION = None

LOGGER = Logger('SUL DATA HANDLER')
PUMP_SPEED_RANGE = int(config['GR3N']['PUMP_SPEED_RANGE'])
MIN_PUMP_SPEED = int(config['GR3N']['MIN_PUMP_SPEED'])
MAX_PUMP_SPEED = int(config['GR3N']['MAX_PUMP_SPEED'])

TMPRT_RANGE = int(config['GR3N']['TMPRT_RANGE'])
MIN_TMPRT = int(config['GR3N']['MIN_TMPRT'])
MAX_TMPRT = int(config['GR3N']['MAX_TMPRT'])


def is_chg_pt(curr, prev):
    for THRESHOLD in range(MIN_PUMP_SPEED, MAX_PUMP_SPEED, PUMP_SPEED_RANGE):
        if curr[0] < THRESHOLD <= prev[0] or prev[0] < THRESHOLD <= curr[0]:
            return True

    for THRESHOLD in range(MIN_TMPRT, MAX_TMPRT, TMPRT_RANGE):
        if curr[1] < THRESHOLD <= prev[1] or prev[1] < THRESHOLD <= curr[1]:
            return True

    return False


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    pump_speed_sig = signals[1]
    pump_speed = {pt.timestamp: (i, pt.value) for i, pt in enumerate(pump_speed_sig.points)}

    tmprt_sig = signals[2]
    tmprt = {pt.timestamp: (i, pt.value) for i, pt in enumerate(tmprt_sig.points)}

    curr_pump_speed_index, curr_pump_speed = pump_speed[t]
    if curr_pump_speed_index > 0:
        try:
            prev_index = [tup[0] for tup in pump_speed.values() if tup[0] < curr_pump_speed_index][-1]
            prev_pump_speed = pump_speed_sig.points[prev_index].value
        except IndexError:
            prev_pump_speed = None
    else:
        prev_pump_speed = curr_pump_speed

    curr_tmprt_index, curr_tmprt = tmprt[t]
    if curr_tmprt_index > 0:
        try:
            prev_index = [tup[0] for tup in tmprt.values() if tup[0] < curr_tmprt_index][-1]
            prev_tmprt = tmprt_sig.points[prev_index].value
        except IndexError:
            prev_tmprt = None
    else:
        prev_tmprt = curr_tmprt

    identified_event = None
    if prev_tmprt is not None:  # for now we just ignore prev_tmprt None, but in case this function have to be revised
        # Identify event as in is_chg_pts
        for i, THRESHOLD in enumerate(range(MIN_TMPRT, MAX_TMPRT, TMPRT_RANGE)):
            if curr_tmprt < THRESHOLD <= prev_tmprt or prev_tmprt < THRESHOLD <= curr_tmprt:
                identified_event = events[i + int((MAX_PUMP_SPEED - MIN_PUMP_SPEED) / PUMP_SPEED_RANGE)]
    else:
        identified_event = events[int((MAX_PUMP_SPEED - MIN_PUMP_SPEED) / PUMP_SPEED_RANGE)]

    if prev_pump_speed is not None:  # for now we just ignore prev_tmprt None, but in case this function have to be revised
        for i, THRESHOLD in enumerate(range(MIN_PUMP_SPEED, MAX_PUMP_SPEED, PUMP_SPEED_RANGE)):
            if curr_pump_speed < THRESHOLD <= prev_pump_speed or prev_pump_speed < THRESHOLD <= curr_pump_speed:
                identified_event = events[i]  # I already know that there is an event for the pump speed
    else:
        identified_event = events[0]

    if identified_event is None:
        LOGGER.error("No event was identified at time {}.".format(t))

    return identified_event


def parse_ts(ts: datetime):
    return Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def parse_data(path: str):
    pump_speed: SampledSignal = SampledSignal([], label='sp')
    Talim: SampledSignal = SampledSignal([], label='Ta')
    tmprt: SampledSignal = SampledSignal([], label='tmp')

    dd_real = pd.read_csv(path)

    dd_pump_speed = dd_real[dd_real['DataObjectField'] == 'SpeedSP']
    dd_pump_speed.loc[:, 'TimeStamp'] = pd.to_datetime(dd_pump_speed['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_pump_speed.sort_values(by='TimeStamp')

    dd_Talim = dd_real[dd_real['DataObjectField'] == 'TCuscinettiAlimentazione']
    dd_Talim.loc[:, 'TimeStamp'] = pd.to_datetime(dd_Talim['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_Talim.sort_values(by='TimeStamp')

    dd_tmprt = dd_real[dd_real['DataObjectField'] == 'Value']
    dd_tmprt.loc[:, 'TimeStamp'] = pd.to_datetime(dd_tmprt['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    dd_tmprt.sort_values(by='TimeStamp')

    pump_speed.points.extend(
        [SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_pump_speed.iterrows()])
    Talim.points.extend(
        [SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_Talim.iterrows()])
    tmprt.points.extend(
        [SignalPoint(parse_ts(record['TimeStamp']), record['Value']) for index, record in dd_tmprt.iterrows()])

    return [Talim, pump_speed, tmprt]


def get_absorption_param(segment: List[SignalPoint], flow: FlowCondition):
    if len(segment) != 0:
        sum_abs = sum([pt.value for pt in segment])
        avg_abs = sum_abs / (len(segment))
        return avg_abs

    return 0


# HRI case study

import configparser
import math
import os
from typing import List

from sha_learning.domain.lshafeatures import FlowCondition
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, Event, SignalPoint
from sha_learning.learning_setup.logger import Logger

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS_VERSION = int(config['SUL CONFIGURATION']['CS_VERSION'].replace('\n', '')[0])
SAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY']
LOGGER = Logger('SUL DATA HANDLER')


def is_chg_pt(curr, prev):
    return curr != prev


def label_event(events: List[Event], signals: List[SampledSignal], t: Timestamp):
    posX = signals[1]
    moving = signals[2]

    '''
    Repeat for every channel in the system
    '''
    curr_mov = list(filter(lambda x: x.timestamp == t, moving.points))[0]
    identified_channel = 'start' if curr_mov.value == 1 else 'stop'

    identified_guard = ''
    if (SAMPLE_STRATEGY == 'SIM' and CS_VERSION in [1, 2, 3]) or \
            (SAMPLE_STRATEGY == 'UPPAAL' and CS_VERSION in [2, 3, 4, 5]):
        posY = signals[3]
        curr_posx = list(filter(lambda x: x.timestamp <= t, posX.points))[-1]
        curr_posy = list(filter(lambda x: x.timestamp <= t, posY.points))[-1]
        if SAMPLE_STRATEGY == 'UPPAAL':
            in_waiting = curr_posx.value >= 2000.0 and curr_posy.value <= 3000.0
        else:
            in_waiting = 16 <= curr_posx.value <= 23.0 and 1.0 <= curr_posy.value <= 10.0
        identified_guard += 'sit' if in_waiting else '!sit'

    if (SAMPLE_STRATEGY == 'SIM' and CS_VERSION in [1, 2, 3]) or \
            (SAMPLE_STRATEGY == 'UPPAAL' and CS_VERSION in [3, 4, 5]):
        posY = signals[3]
        curr_posx = list(filter(lambda x: x.timestamp <= t, posX.points))[-1]
        curr_posy = list(filter(lambda x: x.timestamp <= t, posY.points))[-1]
        if SAMPLE_STRATEGY == 'UPPAAL':
            in_office = curr_posx.value >= 2000.0 and 1000.0 <= curr_posy.value <= 3000.0
        else:
            in_office = 1.0 <= curr_posx.value <= 11.0 and 1.0 <= curr_posy.value <= 10.0
        identified_guard += 'run' if in_office else '!run'

    if SAMPLE_STRATEGY == 'SIM' and CS_VERSION in [4]:
        posY = signals[3]
        curr_posx = list(filter(lambda x: x.timestamp <= t, posX.points))[-1]
        curr_posy = list(filter(lambda x: x.timestamp <= t, posY.points))[-1]

        close_to_chair = 16 <= curr_posx.value <= 20.0 and 3.0 <= curr_posy.value <= 6.0
        identified_guard += 's' if close_to_chair and identified_channel == 'stop' else '!s'

        next_sig_x = list(filter(lambda x: x.timestamp > t, posX.points))
        next_sig_y = list(filter(lambda x: x.timestamp > t, posY.points))
        nextx = next_sig_x[0] if len(next_sig_x) > 0 else curr_posx
        nexty = next_sig_y[0] if len(next_sig_y) > 0 else curr_posy
        dist = math.sqrt((nextx.value - curr_posx.value) ** 2 + (nexty.value - curr_posy.value) ** 2)
        vel = dist / 2.0

        room = signals[4]
        curr_room_status = list(filter(lambda x: x.timestamp <= t.to_secs(), room.points))[-1]
        identified_guard += 'r' if vel > 0.8 and ((close_to_chair and identified_channel == 'stop')
                                                  or curr_room_status.value) else '!r'
        identified_guard += 'h' if curr_room_status.value and not (close_to_chair and vel > 1.0) else '!h'

        identified_guard += 'l' if 0.2 <= vel < 0.4 and not close_to_chair \
                                   and (identified_channel == 'start' or curr_room_status.value) else '!l'
        identified_guard += 'a' if vel < 0.2 and not curr_room_status else '!a'

    '''
    Find symbol associated with guard-channel combination
    '''
    identified_event = [e for e in events if e.guard == identified_guard and e.chan == identified_channel][0]
    return identified_event


def parse_data(path: str):
    if SAMPLE_STRATEGY == 'SIM':
        return parse_traces_sim(path)
    else:
        return parse_traces_uppaal(path)


def parse_traces_sim(path: str):
    if CS_VERSION == 4:
        logs = ['humanFatigue.log', 'humanPosition.log', 'environmentData.log']
    else:
        logs = ['humanFatigue.log', 'humanPosition.log']

    new_traces: List[SampledSignal] = []
    for i, log in enumerate(logs):
        f = open(path + log)
        lines = f.readlines()[1:]
        lines = [line.replace('\n', '') for line in lines]
        t = [float(line.split(':')[0]) for line in lines]
        if i == 0:
            v = [float(line.split(':')[2]) for line in lines]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), v[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, str(len(new_traces))))
        elif i == 1:
            pos = [line.split(':')[2] for line in lines]
            pos_x = [float(line.split('#')[0]) for line in pos]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), pos_x[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, 'humanPositionX'))
            busy = [float(v != pos[j - 1]) for j, v in enumerate(pos) if j > 0]
            busy = [0.0] + busy
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), busy[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, str(len(new_traces))))
            pos_y = [float(line.split('#')[1]) for line in pos]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, x), pos_y[j]) for j, x in enumerate(t)]
            new_traces.append(SampledSignal(signal, 'humanPositionY'))
        else:
            data = [line.split(':')[1] for line in lines]
            data = list(map(lambda x: (float(x.split('#')[0]), float(x.split('#')[1])), data))
            harsh = []
            for pt in data:
                temp = pt[0]
                hum = pt[1]
                harsh.append(temp <= 12.0 or temp >= 32.0 or hum <= 30.0 or hum >= 60.0)
            signal = [SignalPoint(x, harsh[j]) for (j, x) in enumerate(t)]
            new_traces.append(SampledSignal(signal, log))

    return new_traces


def parse_traces_uppaal(path: str):
    f = open(path, 'r')
    print("PARSE TRACES UPPAAL - SUL FUNCTIONS")
    print(path)
    print(f)
    if CS_VERSION in [1, 2]:
        variables = ['humanFatigue[currH - 1]', 'humanPositionX[currH - 1]',
                     'amy.busy || amy.p_2', 'humanPositionY[currH - 1]']
        print(variables)
    else:
        variables = ['humanFatigue[currH - 1]', 'humanPositionX[currH - 1]',
                     'amy.busy || amy.p_2 || amy.run || amy.p_4', 'humanPositionY[currH - 1]']
    lines = f.readlines()
    print(lines)
    split_indexes = [lines.index(k + ':\n') for k in variables]
    print(split_indexes)
    split_lines = [lines[i + 1:split_indexes[ind + 1]] for (ind, i) in enumerate(split_indexes) if
                   i != split_indexes[-1]]
    split_lines.append(lines[split_indexes[-1] + 1:len(lines)])
    traces = len(split_lines[0])
    new_traces: List[SampledSignal] = []
    for trace in range(traces):
        for i, v in enumerate(variables):
            entries = split_lines[i][trace].split(' ')
            entries = entries[1:]
            for e in entries:
                new = e.replace('(', '')
                new = new.replace(')', '')
                entries[entries.index(e)] = new
            t = [float(x.split(',')[0]) for x in entries]
            v = [float(x.split(',')[1]) for x in entries]
            signal = [SignalPoint(Timestamp(0, 0, 0, 0, 0, t[i]), v[i]) for i in range(len(t))]
            new_traces.append(SampledSignal(signal, str(i)))
    return new_traces


def get_ftg_param(segment: List[SignalPoint], flow: FlowCondition):
    try:
        val = [pt.value for pt in segment]
        # metric for walking
        if flow.f_id == 1:
            lambdas = []
            for (i, v) in enumerate(val):
                if i > 0 and v != val[i - 1]:
                    lambdas.append(math.log((1 - v) / (1 - val[i - 1])))
            est_rate = sum(lambdas) / len(lambdas) if len(lambdas) > 0 else None
        # metric for standing/sitting
        else:
            mus = []
            for (i, v) in enumerate(val):
                if i > 0 and v != val[i - 1] and val[i - 1] != 0:
                    mus.append(math.log(v / val[i - 1]))
            est_rate = sum(mus) / len(mus) if len(mus) > 0 else None

        return abs(est_rate) if est_rate is not None else None
    except ValueError:
        return None

