import random

class Action_Space:
    def __init__(self,num_action):
        self.num_action=num_action

    def seed(self,seed):
        random.seed(seed)

    def sample(self):
        action=random.randint(0,num_action-1)
        return action