import random
import math
import numpy as np
import pandas as pd
from pylab import plt,mpl

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"


class Environment:
    def __init__(
        self,
        number_shares,
        trading_time,
        trading_steps,
        stock_volatility,
        permanent_impact_factor,
        temporary_impact_factor,
        risk_aversion_factor
    ):
        
        self.number_shares=float(number_shares)
        self.trading_time=trading_time
        self.trading_steps=trading_steps
        self.dt=self.trading_time/self.trading_steps
        self.stock_volatility=stock_volatility
        self.permanent_impact_factor=permanent_impact_factor
        self.temporary_impact_factor=temporary_impact_factor
        self.risk_aversion_factor=risk_aversion_factor

        self.permanent_impact=0.0
        self.temporary_impact=0.0
        self.execution_risk=0.0
        self.total_execution_cost=0.0
        
        self.index=0
        self.training_episode=0
        self.remaining_share=self.number_shares
        self.xt=(self.trading_steps+1)*[0.0]

        self.optimal_execution_strategy()


    def random_xt_generator(self):
        rng=np.random.default_rng()
        alpha=np.ones(self.trading_steps)

        random_xt=rng.dirichlet(alpha)
        random_xt=np.insert(random_xt,0,0)

        self.random_xt=random_xt
    
    def cal_permanent_impact(self):
        gamma=self.permanent_impact_factor

        risk=np.sum(np.array(self.xt)*gamma*np.cumsum(np.array(self.xt)))
        return risk
        

    def cal_temporary_impact(self):
        eta=self.temporary_impact_factor

        risk=np.sum(eta*np.square(np.array(self.xt)/self.dt)*self.dt)
        return risk
        

    def cal_execution_risk(self):
        lambd=self.risk_aversion_factor
        sigma=self.stock_volatility

        risk=np.sum(((np.array(self.xt)[::-1].cumsum()[::-1]/self.dt)**2)*self.dt*lambd*(sigma**2))

        return risk

        
    
    def optimal_execution_strategy(self):
        X=self.number_shares
        T=self.trading_time
        N=self.trading_steps
        alpha=self.risk_aversion_factor
        sigma=self.stock_volatility
        eta=self.temporary_impact_factor

        kappa=np.sqrt(alpha*(sigma**2)/eta)

        t=np.linspace(0,T,N+1)
        xt_sum=X*np.sinh(kappa*(T-t))/np.sinh(kappa*T)

        xt_opt=-np.diff(xt_sum,prepend=0)
        xt_opt[0]=0

        self.xt_optimal=xt_opt
    

    def get_state(self):
        state=[]

        elem1=[[xt] for xt in self.xt]
        state.append(elem1)

        elem2=[
            self.remaining_share,
            self.index/self.trading_steps,
            self.permanent_impact,
            self.temporary_impact,
            self.execution_risk,
            
        ]
        
        state.append(elem2)

        return state
        
        
    def reset(self):
        self.index=0
        self.training_episode+=1
        self.random_xt_generator()
        self.remaining_share=self.number_shares
        self.permanent_impact=0.0
        self.temporary_impact=0.0
        self.execution_risk=0.0
        self.total_execution_cost=0.0

        self.xt=(self.trading_steps+1)*[0.0]

        self.reward_per_step=list()
        self.permanent_impact_per_step=list()
        self.temporary_impact_per_step=list()
        self.execution_risk_per_step=list()
        self.total_execution_cost_per_step=list()
        self.xt_learned_strategy_per_step=list()
        self.real_state_value=list()
        self.predicted_state_value=list()
        

        state=self.get_state()
        return state,False

    

    def plot(self):
        
        plt.plot(list(range(0,self.trading_steps+1)),1-(np.array(self.xt_optimal).cumsum()),lw=1.0,c="r")
        plt.xlabel("Trading Step")
        plt.ylabel("Shares")
        plt.title("Share at Particular Step")
        plt.show()
        
        plt.plot(list(range(1,self.trading_steps+1)),self.xt_optimal[1:],lw=1.0,c="b")
        plt.xlabel("Trading Step")
        plt.ylabel("Shares Traded")
        plt.ylim(0,1)
        plt.title("Optimal Execution Strategy")
        plt.show()
        

    