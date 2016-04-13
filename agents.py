import numpy as np

class FullEffortTrueAgent(object):
    def __init__(self, types):
        self.reset(types)
            
    def reset(self, tasks):
        self.tasks = tasks
        self.reports = {}
        self.reward = 0
        for idx in self.tasks:
            self.report(idx)
        
    def proficiency(self):
        return np.random.uniform(.5, 1)
    
    def has_effort(self):
        return 1.0
    
    def give_signal(self):
        return 1.0
    
    def report(self, idx):
        # agent's signal given proficiency
        signal = self.tasks[idx] if np.random.uniform() <= self.proficiency() else not self.tasks[idx]
        # see if agent gives effort this time
        if self.has_effort() >= np.random.uniform():
            # see if agent reports truthfully
            self.reports[idx] = signal if self.give_signal() >= np.random.uniform() else not signal
        else:
            # randomly give true/false
            self.reports[idx] = np.random.uniform() > 0.5
            
    def task_score(self, idx, reference, d, reward_factor=1):
        my_nonoverlap = []
        ref_nonoverlap = []
        for task in self.tasks:
            if task not in reference.tasks and task != idx:
                my_nonoverlap.append(self.reports[task])
                if len(my_nonoverlap) == d:
                    break
        for task in reference.tasks:
            if task not in self.tasks and task != idx:
                ref_nonoverlap.append(reference.reports[task])
                if len(ref_nonoverlap) == d:
                    break
        if len(ref_nonoverlap) != d or len(my_nonoverlap) != d:
            raise ValueError("Raters don't have {} non-overlapping tasks".format(d))
            
        my_sum = np.sum(my_nonoverlap, dtype=np.float)
        ref_sum = np.sum(ref_nonoverlap, dtype=np.float)
        reward = (self.reports[idx] == reference.reports[idx]) - (my_sum*ref_sum/d**2 + (1-my_sum/d)*(1-ref_sum/d))
        self.reward += reward_factor*reward
            
class FullEffortFalseAgent(FullEffortTrueAgent):
    def __init__(self, types):
        super(FullEffortFalseAgent, self).__init__(types)
        
    def give_signal(self):
        return 0.0
    
class NoEffortAgent(FullEffortTrueAgent):
    def __init__(self, types):
        super(NoEffortAgent, self).__init__(types)
        
    def has_effort(self):
        return 0.0
    
class RandomEffortTrueAgent(FullEffortTrueAgent):
    def __init__(self, types, max_effort=1.):
        self.max_effort = max_effort
        super(RandomEffortTrueAgent, self).__init__(types)
        
    def has_effort(self):
        return np.random.uniform(0, self.max_effort)
