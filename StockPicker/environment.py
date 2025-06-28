import os
import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from pylab import plt,mpl
from alpha_vantage.timeseries import TimeSeries
from action_space import Action_Space

load_dotenv()

plt.style.use("seaborn-v0_8")
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.family"]="serif"

data_dir_path=r"/home/spiralmonster/Projects/ReinforcementLearningForFinance/StockPicker"


class Environment:
    def __init__(
        self,
        stock_symbol_list,
        num_stock_to_picked,
        execution_gap,
        normalize_price_features=True,
        data_retrieved=False
    ):
        
        self.stock_symbol_list=stock_symbol_list
        self.num_stock_to_picked=num_stock_to_picked
        self.execution_gap=execution_gap
        self.normalize_price_features=normalize_price_features
        self.data_retrieved=data_retrieved
        self.data_path=os.path.join(data_dir_path,"data.csv")

        self.scaler=StandardScaler()

        self.ts=TimeSeries(key=os.environ["ALPHA_VANTAGE_KEY"],output_format="pandas",indexing_type="date")

        self.total_stock=len(self.stock_symbol_list)
       

        self.action_space=Action_Space(total_num_stock=self.total_stock,num_stock_to_picked=self.num_stock_to_picked)

        self.index=0
        self.training_episode=0
        self.current_stocks=self.action_space.sample()
        self.index_to_stock=dict(ind:sym for ind,sym in enumerate(self.stock_symbol_list))

        if self.data_retrieved:
            self.final_data=pd.read_csv(self.data_path)
            self.steps=len(self.final_data)

        else:
            self.get_raw_data()
            self.prepare_data()
            self.final_data.to_csv(self.data_path)
        
                                 

    def get_raw_data(self):
        data=[]
        for symbol in self.stock_symbol_list:
            data_symbol,_=self.ts.get_daily(symbol=symbol,outputsize="full")
            data_symbol=data_symbol.iloc[::-1]
            data_symbol=pd.DataFrame(data_symbol["4. close"])
            data_symbol.rename(columns={"4. close":symbol},inplace=True)
            data.append(data_symbol)

        data_lengths=[len(d) for d in data]
        min_len=min(data_lengths)

        raw_data=[]
        for data_sym in data:
            data_sym=data_sym.iloc[:min_len]
            raw_data.append(data_sym)

        raw_data=pd.concat(raw_data,axis=1)
        raw_data.dropna(inplace=True)

        self.raw_data=raw_data


    
    def prepare_data(self):
        data=self.raw_data
        price_features=[]

        for symb in self.stock_symbol_list:
            data[f"{symb}_returns"]=np.log(data[symb]/data[symb].shift(1))
            data[f"{symb}_mom"]=data[f"{symb}_returns"].rolling(self.execution_gap).mean()
            data[f"{symb}_vol"]=data[f"{symb}_returns"].rolling(self.execution_gap).std()
            data[f"{symb}_SMA"]=data[symb].rolling(self.execution_gap).mean()
            data[f"{symb}_min"]=data[symb].rolling(self.execution_gap).min()
            data[f"{symb}_max"]=data[symb].rolling(self.execution_gap).max()

            price_features.extend(
                [
                    symb,
                    f"{symb}_SMA",
                    f"{symb}_min",
                    f"{symb}_max"
                ]
            )

        data.dropna(inplace=True)

        for lag in range(1,self.execution_gap+1):
            for sym in self.stock_symbol_list:
                data[f"{sym}_returns_lag_{lag}"]=data[f"{sym}_returns"].shift(lag)
                data[f"{sym}_lag_{lag}"]=data[sym].shift(lag)
                data[f"{sym}_mom_lag_{lag}"]=data[f"{sym}_mom"].shift(lag)
                data[f"{sym}_vol_lag_{lag}"]=data[f"{sym}_vol"].shift(lag)
                data[f"{sym}_SMA_lag_{lag}"]=data[f"{sym}_SMA"].shift(lag)
                data[f"{sym}_min_lag_{lag}"]=data[f"{sym}_min"].shift(lag)
                data[f"{sym}_max_lag_{lag}"]=data[f"{sym}_max"].shift(lag)

                price_features.extend([
                    f"{sym}_lag_{lag}",
                    f"{sym}_SMA_lag_{lag}",
                    f"{sym}_min_lag_{lag}"
                    f"{sym}_max_lag_{lag}" 
                ])

        
        data.dropna(inplace=True)
        data=data.iloc[::self.execution_gap+1]
        self.steps=len(data)
        
        if self.normalize_price_features:
            data[price_features]=self.scaler.fit_transform(data[price_features])
        
        self.final_data=data

    
        
    def cal_total_return_for_stocks(stocks):
        total_return=0.0
        data=self.env.final_data.iloc[self.env.index]

        for sym in stocks:
            data_sym=np.array(data[[f"{sym}_lag_{lag}" for lag in range(1,self.env.execution_gap)]])
            ret=np.exp(data_sym.sum(axis=1))
            total_return+=ret

        total_return/=self.env.num_stock_to_picked
        return total_return

        
    
    def get_stock_from_index(self,index):
        stock_symb=[self.env.index_to_stock[ind] for ind in index]
        return stock_symb
        

        
    def get_state(self):
        data=self.final_data.iloc[self.index]

        returns_data=[[data[f"{sym}_returns_lag_{lag}"] for sym in self.stock_symbol_list] for lag in list(range(self.execution_gap,0,-1))]
        price_data=[[data[f"{sym}_lag_{lag}"] for sym in self.stock_symbol_list] for lag in range(list(self.execution_gap,0,-1))]
        momentum_data=[[data[f"{sym}_mom_lag_{lag}"] for sym in self.stock_symbol_list] for lag in list(range(self.execution_gap,0,-1))]
        vol_data=[[data[f"{sym}_vol_lag_{lag}"] for sym in self.stock_symbol_list] for lag in list(range(self.execution_gap,0,-1))]
        sma_data=[[data[f"{sym}_SMA_lag_{lag}"] for sym in self.stock_symbol_list] for lag in list(range(self.execution_gap,0,-1))]
        min_data=[[data[f"{sym}_min_lag_{lag}"] for sym in self.stock_symbol_list] for lag in list(range(self.execution_gap,0,-1))]
        max_data=[[data[f"{sym}_max_lag_{lag}"] for sym in self.stock_symbol_list] for lag in list(range(self.execution_gap,0,-1))]

        curr_stock_state=self.total_stock*[0.0]
        for stock in self.current_stocks:
            curr_stock_state[stock]=1.0

        state=[
            returns_data,
            momentum_data,
            vol_data,
            price_data,
            sma_data,
            min_data,
            max_data,
            curr_stock_state
        ]

        return state
        

        
    def reset(self):
        self.index=0
        self.training_episode+=1
        self.current_stocks=self.action_space.sample()

        self.stock_per_step=[]
        
        self.reward_per_step=[]
        self.return_per_step=[]
        self.real_state_value_per_step=[]
        self.pred_state_value_per_step=[]

        self.set1_stocks=self.get_stock_from_index(self.action_space.sample())
        self.set2_stocks=self.get_stock_from_index(self.action_space.sample())
        self.set3_stocks=self.get_stock_from_index(self.action_space.sample())

        self.set1_stocks_returns_per_step=[]
        self.set2_stocks_returns_per_step=[]
        self.set3_stocks_returns_per_step=[]
        

        state=self.get_state()

        return state,False

        

    def plots(self):
        data[[f"{sym}_returns" for sym in self.stock_symbol_list]].plot(
            figsize=(10,6),
            style=['b','g','c','r','m','y','k']
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Returns")
        plt.title("Time VS Returns for different stocks")
        plt.show()
        

        data[[f"{sym}_mom" for sym in self.stock_symbol_list]].plot(
            figsize=(10,6),
            style=['b','g','c','r','m','y','k']
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Momentum")
        plt.title("Time VS Stock Momentum for different stocks")
        plt.show()


        data[[f"{sym}_vol" for sym in self.stock_symbol_list]].plot(
            figsize=(10,6),
            style=['b','g','c','r','m','y','k']
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Volatility")
        plt.title("Time VS Stock Volatility for different stocks")
        plt.show()


        data[[f"{sym}_SMA" for sym in self.stock_symbol_list]].plot(
            figsize=(10,6),
            style=['b','g','c','r','m','y','k']
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.title("Time VS Stock SMA(Simple Moving Average) for different stocks")
        plt.show()
        

        data[[f"{sym}_min" for sym in self.stock_symbol_list]].plot(
            figsize=(10,6),
            style=['b','g','c','r','m','y','k']
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.title("Time VS Minimum Stock Price for different stocks")
        plt.show()

        
        data[[f"{sym}_max" for sym in self.stock_symbol_list]].plot(
            figsize=(10,6),
            style=['b','g','c','r','m','y','k']
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.title("Time VS Maximum Stock Price for different stocks")
        plt.show()

        
