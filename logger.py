from __future__ import annotations
import numpy as np
import copy
import json

class Log(object):
    def __init__(self, times: np.ndarray[np.float64] = None, states: np.ndarray[dict] = None):
        if times is None and states is None:
            self.times = None
            self.states = None
            self.start_time = None
            self.end_time = None
            self.keys = None
            return
        
        if len(times) != len(states):
            raise Exception('must have same number of times and states; %d != %d'\
                % (len(times), len(states)))

        if len(times) > 0:
            self.keys = states[0].keys()
            for (i,s) in enumerate(states):
                if s.keys() != self.keys:
                    raise Exception('Key mismatch between state %d and %d' % (0, i))
        else:
            self.keys = None

        self.times: np.ndarray[np.float64] = times.copy().astype(np.float64)
        self.states: np.ndarray[dict] = copy.deepcopy(states)
        self.start_time: np.float64 = times[0]
        self.end_time: np.float64 = times[-1]
    
    def insert(self, time: np.float64, state: dict):
        state = copy.deepcopy(state)
        if self.times is None:
            self.times = np.array([time], dtype=np.float64)
            self.states = np.array([state])
            self.start_time = time
            self.end_time = time
            self.keys = state.keys()
            return
        
        if self.keys is None:
            self.keys = state.keys()
        
        if self.keys != state.keys():
            raise Exception('Key mismatch')
        
        next_idx = np.searchsorted(self.times, time) # this gives the index s.t. time <= self.times[i]

        if next_idx < len(self.times) and self.times[next_idx] == time:
            # time already in log; overwrite state
            self.states[next_idx] = state
        else:
            # add to log
            self.times = np.insert(self.times, next_idx, time)
            self.states = np.insert(self.states, next_idx, state)

            if next_idx == 0:
                self.start_time = time
            elif next_idx == len(self.times) - 1:
                self.end_time = time
        
    def clip_time(self, time: np.float64) -> np.float64:
        return np.clip(time, self.start_time, self.end_time)
    
    def sample(self, time: np.float64) -> dict:
        """ Sample the trajectory for the given time.
            Linearly interpolates between trajectory samples.
            If time is outside of trajectory, gives the start/end state.
        
        Arguments:
            time: time to sample
        """
        if self.times is None or len(self.times) == 0:
            return None
        

        time = self.clip_time(time)
        next_idx = np.searchsorted(self.times, time)
        prev_idx = 0 if next_idx == 0 else next_idx-1
            
        if prev_idx == next_idx:
            return self.states[prev_idx]
        
        prev_val = self.states[prev_idx]
        next_val = self.states[next_idx]
        prev_time = self.times[prev_idx]
        next_time = self.times[next_idx]

        alpha = (time - prev_time) / (next_time - prev_time)

        interpolated_val = {k: __alpha_blend__(prev_val[k], next_val[k], alpha) for k in self.keys}

        return interpolated_val
    
    def append(self, other: Log) -> Log:
        appended = Log()
        for (t, s) in zip(self.times, self.states):
            appended.insert(t, s)

        for (t, s) in zip(other.times, other.states):
            appended.insert(t + self.end_time, s)
        
        return appended

def __alpha_blend__(v0, vf, alpha):
    return vf*alpha + (1-alpha)*v0

def from_json(file, index_var) -> Log:
    f = open(file)
    data = json.load(f)
    log = Log()

    for state in data:
        t = state[index_var]
        del state[index_var]
        log.insert(t, state)
    
    return log
