import os
import numpy as np
import pandas as pd
from pprint import pprint
from pylab import plt,mpl
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from observation_space import Observation_Space
from action_space import Action_Space

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"

load_dotenv()

class TradingEnvironment:
    def __init__(
        self,
        symbol,
        start_date,
        end_date,
        number_lags,
        sma_period,
        window,
        action_type,
        num_action,
        include_additional_features=True
    ):
        self.symbol=symbol
        self.start_date=pd.to_datetime(start_date)
        self.end_date=pd.to_datetime(end_date)
        self.number_lags=number_lags
        self.sma_period=sma_period
        self.window=window
        self.action_type=action_type
        self.num_action=num_action
        self.include_additional_features=include_additional_features
        self.ts=TimeSeries(key=os.environ["ALPHA_VANTAGE_KEY"],output_format="pandas",indexing_type="date")
     
        self.get_raw_data()
        self.prepare_data_lag()

        if self.include_additional_features:
            self.add_extra_features()
            
        else:
            self.final_data=self.data

        self.observation_space=Observation_Space(self.n_features)
        self.action_space=Action_Space(action_type=self.action_type,n=self.num_action)
        self.action_space.seed(100)
            

    def get_raw_data(self):
        raw_data,_=self.ts.get_daily(symbol=self.symbol,outputsize="full")
        raw_data=raw_data.iloc[::-1]
        raw_data=pd.DataFrame(raw_data["4. close"]).dropna()
        raw_data=raw_data[(raw_data.index>=self.start_date)&(raw_data.index<=self.end_date)]
        raw_data.rename(columns={"4. close":self.symbol},inplace=True)
        self.raw_data=raw_data
        
    def prepare_data_lag(self):
        self.features=[]
        data=self.raw_data
        data['returns']=np.log(data[self.symbol]/data[self.symbol].shift(1))
        data['position']=np.where(data['returns']>0,1,0)

        for lag in range(1,self.number_lags+1):
            data[f'returns_lag{lag}']=data['returns'].shift(lag)
            self.features.append(f'returns_lag{lag}')
  
        self.n_features=self.number_lags
        self.data=data.dropna()
        

    def add_extra_features(self):
        data=self.data
        data['momentum']=data['returns'].rolling(self.window).mean().shift(1)
        data['volatilty']=data['returns'].rolling(self.window).std().shift(1)
        data['SMA']=data[self.symbol].rolling(self.sma_period).mean().shift(1)
        data['max']=data[self.symbol].rolling(self.sma_period).max().shift(1)
        data['min']=data[self.symbol].rolling(self.sma_period).min().shift(1)
        data=data.dropna()
        
        self.features.extend(['momentum','volatilty','SMA','max','min'])
        self.n_features+=5
        
        self.final_data=data

    def get_state(self):
        state=np.array(self.final_data[self.features].iloc[self.index])
        return state
        
    def reset(self):
        self.index=0
        self.performance=1
        self.pred_action=[]
        self.real_action=[]
        state=self.get_state()
        return state,False

    def inspect_data(self):
        print("Top 5 entries in Dataset:")
        pprint(self.final_data.head())

        print("Last 5 entries in Dataset:")
        pprint(self.final_data.tail())

        print("Info of Dataset:")
        self.final_data.info()

        print("Some Statistics about Dataset:")
        pprint(self.final_data.describe())

        print("Correlation among features: ")
        pprint(self.final_data.corr())

    def plots_from_dataset(self):
        if self.include_additional_features:
            self.final_data[[self.symbol,"SMA","min","max"]].plot(figsize=(10,6),style=["b","g","r","c"])
            title=f"{self.symbol}| Stock Price"
            
        else:
            self.final_data[[self.symbol]].plot(figsize,style=["b"])
            title=f"{self.symbol}| Stock Price| SMA| Min Price| Max Price"
            
        plt.ylabel("Stock Price")
        plt.title(title)
        plt.legend()
        plt.show()

        if self.include_additional_features:
            self.final_data[["returns"]].cumsum().apply(np.exp).plot(figsize=(10,6),style=["b"])
            title=f"{self.symbol}| Returns"
            plt.ylabel("Return")
            plt.title(title)
            plt.show()

            self.final_data[["momentum"]].plot(figsize=(10,6),style=["g"])
            title=f"{self.symbol}| Momentum"
            plt.ylabel("Return's Momentum")
            plt.title(title)
            plt.show()
        
            self.final_data[["volatilty"]].plot(figsize=(10,6),style=["r","g"])
            title=f"{self.symbol}| Volatility"
            plt.ylabel("Return's Volatility")
            plt.title(title)
            plt.show()

        else:
            self.final_data[["returns"]].cumsum().apply(np.exp).plot(figsize=(10,6),style=["b"])
            plt.ylabel("Return")
            title=f"{self.symbol}| Returns"
            plt.show()

        
            
        
        
        