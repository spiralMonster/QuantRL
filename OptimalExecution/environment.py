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
        
        self.number_shares=number_shares
        self.trading_time=trading_time
        self.trading_steps=trading_steps
        self.stock_volatility=stock_volatility
        self.permanent_impact_factor=permanent_impact_factor
        self.temporary_impact_factor=temporary_impact_factor
        self.risk_aversion_factor=risk_aversion_factor

        self.index=0
        self.remaining_share=self.number_shares
        self.xt=(self.trading_steps+1)*[0.0]

        self.optimal_execution_strategy()
        

    def get_state(self):
        state=[]

        elem1=[[xt] for xt in self.xt]
        state.append(elem1)

        elem2=self.xt
        elem2.append(self.remaining_share)
        elem2.append(self.index/self.trading_steps)
        state.append(elem2)

        return state
        
    
    def reset(self):
        self.index=0
        self.remaining_share=self.number_shares

        self.xt=(self.trading_steps+1)*[0.0]

        state=self.get_state()
        return state,False

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
        

    def plot(self):
        t=np.linspace(0,self.trading_time,self.trading_steps+1)

        plt.plot(t,1-np.cumsum(self.xt_optimal),lw=1.0,c="r")
        plt.xlabel("Trading Step")
        plt.ylabel("Shares")
        plt.title("Share at Particular Step")
        plt.show()
        
        plt.plot(t,self.xt_optimal,lw=1.0,c="b")
        plt.xlabel("Trading Step")
        plt.ylabel("Shares Traded")
        plt.ylim(0,1)
        plt.title("Optimal Execution Strategy")
        plt.show()
        

    