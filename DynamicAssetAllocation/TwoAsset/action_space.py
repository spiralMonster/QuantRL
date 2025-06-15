import random

class Action_Space:
    
    def seed(self,seed):
        random.seed(seed)

    def sample(self):
        return random.random()