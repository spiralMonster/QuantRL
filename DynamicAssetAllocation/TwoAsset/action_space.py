import random

class Action_Space:
    def __init__(self,n):
        self.n=n

    def seed(self,seed):
        random.seed(seed)

    def sample(self):
        return random.random()