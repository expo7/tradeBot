print('forwardTest')
import alpaca_trade_api as tradeapi
import pandas as pd
# import data as data
import datetime
import os
import datetime
import backtest as bt
import yfinance as yf
import numpy as np
import ta
import statsmodels.api as sm

# bot.py
from .alerts import send_email_alert, send_sms_alert
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

def trigger_alerts(stock, prediction):
    subject = f"Trading Alert for {stock}"
    message = f"The bot predicts a {prediction} for {stock}."
    recipient_list = ['user@example.com']
    
    # Send email alert
    send_email_alert(subject, message, recipient_list)
    
    # Send SMS alert
    send_sms_alert(message, '+1234567890')
    
    # Send browser notification
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        'alerts',
        {
            'type': 'send_alert',
            'message': message
        }
    )


# Get the current time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Print the current time
print("Current Time:", current_time)
risk_free_rate = 0.02
interval='1d'
period='2y'
benchmark = 'spy.csv'
tickers = ['msft.csv', 'gs.csv','wmt.csv','nvda.csv','coin.csv','spy.csv']
API_KEY = 'PK6UGQC2A8K4MRPCW8Z1'
API_SECRET = 'BfZxdNbb9KmKjoYCn5zFCnL1izcnLpMmG31MuMEz'
BASE_URL = 'https://paper-api.alpaca.markets/'  # Use the paper trading URL
liveTrading = False
def get_account():
    positions_dict={}
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    cash=account.cash
    positions=api.list_positions()
    for stock in positions:
        symbol = stock.symbol
        shares = stock.qty
        positions_dict[symbol] = int(shares)
    positions_dict['cash']=float(account.cash)
    pending_orders = api.list_orders(status='open')
    return positions_dict, pending_orders

def execute_trade(symbol, signal, qty,api):
    print(f'Executing trade for {symbol} signal:{signal} qty:{qty}')
    
    if signal == 1:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
    elif signal == -1:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )

def apply_market_conditions(df, df2):
    l=len(df)
    # print(l)
    slow=20
    fast=5
    # print(fast)
    # print(slow)
    df2['smaFast'] = ta.trend.sma_indicator(df2['Close'], window=fast)
    df2['smaSlow'] = ta.trend.sma_indicator(df2['Close'], window=slow)
    # df['smaFast'] = ta.trend.sma_indicator(df['Close'], window=fast)
    # df['smaSlow'] = ta.trend.sma_indicator(df['Close'], window=slow)  
    df['marketUp'] = df2['smaFast'] > df2['smaSlow']
    df['marketDown'] = df2['smaFast'] < df2['smaSlow']
    
    df['marketUp'] = df['Close'].loc[df['marketUp']] * .95
    df['marketDown'] = df['Close'].loc[df['marketDown']] * 1.05
    df['sma_bull'] = df['Close'].loc[df['smaFast'] > df['smaSlow']] * .8
    df['sma_bear'] = df['Close'].loc[df['smaFast'] < df['smaSlow']] * 1.2
    
    df['RSI_bull'] = (df['rsi'] <= 30) 
    df['RSI_bear'] = (df['rsi'] >= 70) 
    
    return df


def execute_trades_multi(df_dict, starting_cash=10000.0, daily_risk_free_rate=0,api=None,myAccount=None):
    cash = starting_cash
    equity_list = []
    equity_list2 = [cash]

    position_dict = {stock: 0.0 for stock in df_dict.keys()}
    # print(position_dict)
    print(myAccount)
    last = len(df_dict[list(df_dict.keys())[0]])
    for i in range(1, len(df_dict[list(df_dict.keys())[0]])):  # assuming all DataFrames have the same length
        cash = myAccount['cash']
        for stock, df in df_dict.items():
            if i == last-1:
                score = df.iloc[i]['score']
                print(f'{stock} {score}')
                cash = myAccount['cash']
                if stock.upper() not in myAccount.keys():   
                    position_dict[stock] = 0
                    print(f'cash:{cash} position:{position_dict[stock]} {stock} not in myAccount')
                else:
                    position_dict[stock] = myAccount[stock.upper()]
                    print(f'cash:{cash} position:{position_dict[stock]} {stock} in myAccount')
            if position_dict[stock] == 0 and df.loc[df.index[i], 'score'] >= 3:
            # if cash>0 and df.loc[df.index[i], 'score'] >= 3:
                price = df.loc[df.index[i], 'Close']
                available_cash = cash / len(df_dict)  # distribute cash equally among stocks
                position_dict[stock] = available_cash / price
                df.loc[df.index[i], 'BUY'] = price
                cash -= available_cash
                df.loc[df.index[i], 'position'] = round(position_dict[stock])
                if i==last-1:
                    execute_trade(stock.upper(), signal=1, qty=round(position_dict[stock]),api=api)
                    print(f'{i} Buying {round(position_dict[stock])} shares of {stock} at {round(price)} cash:{round(cash)}  Date:{df.index[i]}')
                    

            elif position_dict[stock] > 0 and df.loc[df.index[i], 'score'] <= -3:
                cash += position_dict[stock] * df.loc[df.index[i], 'Close']
                x=position_dict[stock]
                position_dict[stock] = 0
                df.loc[df.index[i], 'SELL'] = df.loc[df.index[i], 'Close']
                df.loc[df.index[i], 'position'] = round(position_dict[stock])
                if i==last-1:
                    cash=myAccount['cash']
                    if stock in myAccount.keys().upper():
                            
                        position_dict[stock]=myAccount[stock.upper()]
                        print(f'cash:{cash} position:{position_dict[stock]}')
                        execute_trade(stock.upper(), signal=-1, qty=round(position_dict[stock]),api=api)
                        print(f'{i} Selling {x} shares of {stock} at {round(df.loc[df.index[i], "Close"])} cash:{round(cash)} equity:{equity_list[-1]} Date:{df.index[i]}')
                # print(f'{i} Selling {x} shares of {stock} at {round(df.loc[df.index[i], "Close"])} cash:{round(cash)} equity:{equity_list[-1]}')

            else:
                equity = sum([position_dict[s] * df_dict[s].loc[df.index[i], 'Close'] for s in df_dict]) + cash
                # print(f'{i} Holding cash:{round(cash)} equity:{round(equity)}')
                equity_list.append(equity)
                df.loc[df.index[i], 'equity'] = equity
                df.loc[df.index[i], 'position'] = round(position_dict[stock])
                if i==last-1:
                    cash=myAccount['cash']
                    if stock in myAccount.keys():
                        position_dict[stock]=myAccount[stock]
                    else:
                        position_dict[stock]=0
                    cash=myAccount['cash']
                    # print(f'cash:{cash} position:{position_dict[stock]}')
                    # print(f'{i} Holding cash:{round(cash)} equity:{round(equity)} Date:{df.index[i]}')
        equity_list2.append(equity_list[-1])
        

        cash *= (1 + daily_risk_free_rate)
        
    return df_dict, equity_list2

def get_forward_data(tickers,period=period,interval=interval,path='path'):
    for ticker_file in tickers:
        ticker_symbol = ticker_file.split('.')[0]
        if not os.path.exists(f'{path}/{ticker_file}'):
            ticker_data = yf.Ticker(ticker_symbol)
            historical_data = ticker_data.history(period=period, interval=interval)
            historical_data.to_csv(f'{path}/{ticker_file}')
            print(f'getting data for {ticker_symbol}')
        else:
            print(f'{path}/{ticker_file} already exists')           

def load_data(file,fullPath):
    stock = file.split('.')[0]
    df = pd.read_csv(f'{fullPath}/{stock}.csv', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return stock, df

def makePath(invterval,period):    
    current_datetime = datetime.datetime.now()
    basepath = 'forwardData'
    folder_name = current_datetime.strftime('%Y-%m-%d')
    fullPath=f'{basepath}/{folder_name}/{interval}{period}'
    if not os.path.exists(f'{basepath}/{folder_name}'):
        os.makedirs(f'{basepath}/{folder_name}')
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    if not os.path.exists(fullPath):
        os.makedirs(fullPath)
        print(f"{fullPath}' created successfully.")
    else:
        print(f"'{fullPath}' already exists.")
    return fullPath




def regression_analysis(bench, eq_list, starting_cash, risk_free_rate=0.02, trading_days=252, printResults=False):
    # Convert DataFrames to UTC
    # bench.index = bench.index.tz_convert('UTC') if bench.index.tz is not None else bench.index
    eq_list = pd.DataFrame(eq_list, index=bench.index)
    eq_list.index = pd.to_datetime(eq_list.index, utc=True)
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    # Calculate percentage changes
    eqd = eq_list.pct_change()
    df1['strategy_returns'] = eqd
    df2['benchmark_returns'] = bench['Close'].pct_change()

    df1.fillna(0, inplace=True)
    df2.fillna(0, inplace=True)
    
    # Ensure consistent length
    trading_days = 252
    df1['excess_returns'] = df1['strategy_returns'] - (risk_free_rate / trading_days)
    df2['excess_benchmark_returns'] = df2['benchmark_returns'] - (risk_free_rate / trading_days)

    aligned_df = df1.join(df2, how='inner')
    X = sm.add_constant(aligned_df['benchmark_returns'])
    model = sm.OLS(aligned_df['strategy_returns'], X).fit()
    alpha, beta = model.params

    sharpe_ratio = (df1['strategy_returns'].mean() - (risk_free_rate / trading_days)) / df1['strategy_returns'].std() * np.sqrt(trading_days)
    downside_returns = df1['strategy_returns'][df1['strategy_returns'] < 0]
    sortino_ratio = (df1['strategy_returns'].mean() - (risk_free_rate / trading_days)) / downside_returns.std() * np.sqrt(trading_days)
    
    if printResults:
        print('********************************')
        print('Regression Analysis')
        print(f'Alpha: {alpha:.4f}')
        print(f'Beta: {beta:.4f}')
        print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
        print(f'Sortino Ratio: {sortino_ratio:.4f}')
        print('********************************')
    
    return alpha, beta, sharpe_ratio, sortino_ratio

def calculate_indicators(df):
    df['smaFast'] = ta.trend.sma_indicator(df['Close'], window=5)
    df['smaSlow'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    df['MACD_bull'] = (df['MACD'] > df['MACD_signal'])
    df['MACD_bear'] = (df['MACD'] < df['MACD_signal'])
    
    return df

def assign_scores(df):
    df['MACD_bull_score'] = df['MACD_bull'].apply(lambda x: 1 if x else 0)
    df['MACD_bear_score'] = df['MACD_bear'].apply(lambda x: -1 if x else 0)
    df['RSI_bull_score'] = df['RSI_bull'].apply(lambda x: 1 if x else 0)
    df['RSI_bear_score'] = df['RSI_bear'].apply(lambda x: -1 if x else 0)
    df['sma_bull_score'] = df['sma_bull'].apply(lambda x: 1 if x > 0 else 0)
    df['sma_bear_score'] = df['sma_bear'].apply(lambda x: -1 if x > 0 else 0)
    df['marketUp_score'] = df['marketUp'].apply(lambda x: 1 if x > 0 else 0)
    df['marketDown_score'] = df['marketDown'].apply(lambda x: -1 if x > 0 else 0)

    # Calculate the final score
    df['score'] = df[['MACD_bull_score', 'MACD_bear_score', 'RSI_bull_score', 'RSI_bear_score', 
                      'sma_bull_score', 'sma_bear_score', 'marketUp_score', 'marketDown_score']].sum(axis=1)
    return df

def main(files=tickers,benchmark=benchmark,interval=interval,period=period,API_KEY=API_KEY,API_SECRET=API_SECRET,BASE_URL=BASE_URL,liveTrading=liveTrading):
    print('main')
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    myAccount,pending_orders=get_account()
    # starting_cash, positions,pending_orders = get_account(API_KEY=API_KEY,API_SECRET=API_SECRET,BASE_URL=BASE_URL)
    # starting_cash = float(starting_cash)
    starting_cash = 100000.0
    fullPath = makePath(interval,period)
    get_forward_data(tickers=files,period=period,interval=interval,path=fullPath)
    _, df2 = load_data(file=benchmark, fullPath=fullPath)
    df_dict = {}
    for file in files:
        if file != benchmark:
            stock, df = load_data(file, fullPath=fullPath)
            df = calculate_indicators(df)
            df_dict[stock] = df
            
        
    for stock, df in df_dict.items():
        df = apply_market_conditions(df, df2)
        df = assign_scores(df)
        df_dict[stock] = df
        
        
    trading_days = 252
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / trading_days) - 1
    if len(pending_orders)==0:
        df_dict, equity_list = execute_trades_multi(df_dict=df_dict, starting_cash=starting_cash, daily_risk_free_rate=daily_risk_free_rate,api=api,myAccount=myAccount)
        # print(equity_list)
        e=pd.DataFrame(equity_list) 
        # print(len(e))   
        # e.plot()
        # alpha, beta, sharpe_ratio, sortino_ratio=regression_analysis(bench=df2,eq_list=equity_list, starting_cash=starting_cash, risk_free_rate=0.02, trading_days=252,printResults=True)
        # print(df_dict.keys())
    else:
        print('orders pending')    
    

if __name__ == "__main__":
    main()
    from datetime import datetime

# Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Print the current time
    print("Current Time:", current_time)








        

