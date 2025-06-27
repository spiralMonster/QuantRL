import random

class Action_Space:
    def __init__(self,total_num_stock,num_stock_to_picked):
        self.total_num_stock=total_num_stock
        self.num_stock_to_picked=num_stock_to_picked

    def seed(self,seed):
        random.seed(seed)

    def sample(self):
        action=random.sample(range(self.total_num_stock),self.num_stock_to_picked)
        return action