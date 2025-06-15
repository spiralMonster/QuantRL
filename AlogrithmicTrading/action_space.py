import random

class Action_Space:
    def __init__(self,action_type,n=1):
        self.action_type=action_type
        self.n=n

    def seed(self,seed):
        random.seed(seed)

    def sample(self):
        if self.action_type=='discrete':
            return random.randint(0,self.n-1)

        else:
            return random.uniform(0,self.n)    