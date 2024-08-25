import pandas as pd
import numpy as np
import mplfinance as mpf
import statsmodels.api as sm
import ta
import matplotlib.pyplot as plt
# import os
import pickle
import warnings
import yfinance as yf
# import time
import os
import datetime
import pdb  
import random
from PIL import Image
def get_image_dimensions(image_path):
    """
    Get the dimensions of an image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tuple: Width and height of the image.
    """
    with Image.open(image_path) as img:
        return img.size


def getRandiTickers(n):
    """
    Get a list of random ticker symbols.

    Parameters:
    n (int): Number of random tickers to select.

    Returns:
    list: List of random ticker symbols.
    """
    with open('bot/tickers.txt', 'r', encoding='utf-8') as file:
        content = file.readlines()

    tickers = [line.split('\t')[0] + '.csv' for line in content]
    random_tickers = random.sample(tickers, n)
    return random_tickers
  

# Ignore all warnings
warnings.filterwarnings('ignore')
import os
import datetime

def makePath(i, p, useOldData=False):
    """
    Create a directory path for backtest data based on the current date and input parameters.
    If useOldData is True, return the most recent folder with data.

    Parameters:
    i (str): Identifier for the folder.
    p (str): Additional parameter for the folder name.
    useOldData (bool): Flag to use the most recent folder with data if True (default is False).

    Returns:
    str: The full path to the created or found folder.
    """
    current_datetime = datetime.datetime.now()
    basepath = 'backtestData'
    folder_name = current_datetime.strftime('%Y-%m-%d')
    fullPath = f'{basepath}/{folder_name}/{i}{p}'

    if useOldData:
        # Find the most recent folder with data
        if os.path.exists(basepath):
            folders = sorted([f for f in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, f))], reverse=True)
            for folder in folders:
                potential_path = f'{basepath}/{folder}/{i}{p}'
                if os.path.exists(potential_path):
                    print(f"Using existing folder: '{potential_path}'")
                    return potential_path
        print("No existing folder found, creating a new one.")
    
    # Create the new folder if useOldData is False or no existing folder is found
    if not os.path.exists(f'{basepath}/{folder_name}'):
        os.makedirs(f'{basepath}/{folder_name}')
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    if not os.path.exists(fullPath):
        os.makedirs(fullPath)
        print(f"'{fullPath}' created successfully.")
    else:
        print(f"'{fullPath}' already exists.")
    
    return fullPath
# def makePath(i,p,useOldData=False):    
#     current_datetime = datetime.datetime.now()
#     basepath = 'backtestData'
#     folder_name = current_datetime.strftime('%Y-%m-%d')
#     fullPath=f'{basepath}/{folder_name}/{i}{p}'
#     if not os.path.exists(f'{basepath}/{folder_name}'):
#         os.makedirs(f'{basepath}/{folder_name}')
#         print(f"Folder '{folder_name}' created successfully.")
#     else:
#         print(f"Folder '{folder_name}' already exists.")
#     if not os.path.exists(fullPath):
#         os.makedirs(fullPath)
#         print(f"{fullPath}' created successfully.")
#     else:
#         print(f"'{fullPath}' already exists.")
#     return fullPath


def get_backtest_data(tickers, period='1d', interval='2y', path='path'):
    """
    Fetch historical data for a list of tickers and save it to CSV files.

    Parameters:
    tickers (list): List of ticker file names (e.g., ['AAPL.csv', 'GOOG.csv']).
    period (str): The period for which to fetch the data (default is '1d').
    interval (str): The interval for the data (default is '2y').
    path (str): The directory path where the CSV files will be saved (default is 'path').

    Returns:
    None
    """
    for ticker_file in tickers:
        ticker_symbol = ticker_file.split('.')[0]
        if not os.path.exists(f'{path}/{ticker_file}'):
            ticker_data = yf.Ticker(ticker_symbol)
            historical_data = ticker_data.history(period=period, interval=interval)
            historical_data.to_csv(f'{path}/{ticker_file}')
            print(f'getting data for {ticker_symbol}')
        else:
            print(f'{path}/{ticker_file} already exists')   
# Load and prepare data
def check():
    spy=pd.read_csv('2y1d/spy.csv', index_col=0, parse_dates=True)
    d='2y1d'
    all_csv_files = [f for f in os.listdir(d) if f.endswith('.csv')]
    for file in all_csv_files:


        appl=pd.read_csv(f'{d}/{file}', index_col=0, parse_dates=True)
        l=len(spy)==len(appl)
        s=spy.index[0]==spy.index[0]
        e=appl.index[-1]==spy.index[-1]
        if l and s and e:
            # print(f'{file} matches spy')
            print('ok')
        else:
            print(f'{file} does not match spy')
def check_data(L=502, directory='2y1d'): 
    lengthProblem=[]
    startProblme=[]
    endProblem=[]
    all_csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    # print(all_csv_files)
    spy=pd.read_csv(f'{directory}/spy.csv', index_col=0, parse_dates=True)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    spyStart=spy.index[0]
    spyEnd=spy.index[-1]    
    L=len(spy)
    for file in all_csv_files:
        # if file == 'spy.csv':
        #     df = pd.read_csv(f'{directory}/{file}', index_col=0, parse_dates=True)
        #     dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
            
        #     spyStart=df.index[0]
        #     spyEnd=df.index[-1]
        #     print(f'{file} has {len(df)} rows from {start} to {end}')
        
        stock = file.split('.')[0]
        df = pd.read_csv(f'done/{stock}.csv', index_col=0, parse_dates=True)
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
        
        start=df.index[0]
        end=df.index[-1]
        if len(df) != L:
            lengthProblem.append(file)
            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
            # start=df.index[0]
            # end=df.index[-1]
            # print(f'{file} has {len(df)} rows from {start} to {end}')
        if start!=spyStart:
            startProblme.append(file)
            # print(f'{file} has start date problem')
        if end!=spyEnd: 
            endProblem.append(file)
            # print(f'{file} has end date problem')
        else:
            print(f'{file} matches spy')
            
            
            
    print(f'L {lengthProblem}')
    print(f'S {startProblme}')
    print(f'E {endProblem}')
    
    stock = file.split('.')[0]
    df = pd.read_csv(f'done/{stock}.csv', index_col=0, parse_dates=True)
    if len(df) != 502:
        length = len(df)
    df.index = pd.to_datetime(df.index, utc=True)
    return stock, df


# Load and prepare data
def load_data(file, directory='2y1d'):
    if file.endswith('.csv'):
        stock = file.split('.')[0]
    else:
        stock = file    
    
    df = pd.read_csv(f'{directory}/{file}', index_col=0, parse_dates=True)
    if len(df) == 502:
        length = len(df)
    df.index = pd.to_datetime(df.index, utc=True)
    return stock, df

# Calculate technical indicators
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

# Apply market conditions
def apply_market_conditions(df, df2):
    # l=len(df)
    # print(l)
    # slow=int(l/25)
    # fast=int(l/(25*4))
    # print(fast)
    # print(slow)
    fast=5
    slow=20
    df2['smaFast'] = ta.trend.sma_indicator(df2['Close'], window=fast)
    df2['smaSlow'] = ta.trend.sma_indicator(df2['Close'], window=slow)
    df['marketUp'] = df2['smaFast'] > df2['smaSlow']
    df['marketDown'] = df2['smaFast'] < df2['smaSlow']
    
    df['marketUp'] = df['Close'].loc[df['marketUp']] * .95
    df['marketDown'] = df['Close'].loc[df['marketDown']] * 1.05
    df['sma_bull'] = df['Close'].loc[df['smaFast'] > df['smaSlow']] * .8
    df['sma_bear'] = df['Close'].loc[df['smaFast'] < df['smaSlow']] * 1.2
    
    df['RSI_bull'] = (df['rsi'] <= 30) 
    df['RSI_bear'] = (df['rsi'] >= 70) 
    
    return df

# Assign scores based on conditions
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

# Execute trades based on scores
def execute_trades(df, starting_cash=10000.0, daily_risk_free_rate=0):
    cash = starting_cash
    equity = cash
    equity_list = [equity]
    position = 0.0

    for i in range(1, len(df)):
        if position == 0 and df.loc[df.index[i], 'score'] >= 3:
            price = df.loc[df.index[i], 'Close']
            position = cash / price
            df.loc[df.index[i], 'BUY'] = price
            df.loc[df.index[i], 'position'] = round(position)
            cash = 0
            df.loc[df.index[i], 'BUY'] = price
        elif position > 0 and df.loc[df.index[i], 'score'] <= -3:
            cash = position * df.loc[df.index[i], 'Close']
            position = 0
            df.loc[df.index[i], 'position'] = round(position)  
            df.loc[df.index[i], 'SELL'] = df.loc[df.index[i], 'Close']
        else:
            equity = position * df.loc[df.index[i], 'Close'] + cash
            if position == 0:
                cash *= (1 + daily_risk_free_rate)
                equity = cash
        equity_list.append(equity)
        df.loc[df.index[i], 'equity'] = equity
        df.loc[df.index[i], 'position'] = round(position)

    return df, equity_list

# Calculate and print performance metrics
def calculate_metrics(df,df2, equity_list, starting_cash, daily_risk_free_rate, total_days, inMarket,name,winList=[],loseList=[],printResults=False):
    inMarket = df['position'].ne(0).sum()
    total_days = len(df)
    profit = equity_list[-1] - equity_list[0]
    start_pos = starting_cash / df['Close'].iloc[0]
    end = df['Close'].iloc[-1] * start_pos
    buyhold_profit = end - starting_cash
    profit_per_day = profit / inMarket
    buyhold_per_day = buyhold_profit / total_days
    return_p = profit / starting_cash
    return_bh = buyhold_profit / starting_cash
    return_daily = return_p / inMarket
    return_bh_daily = return_bh / total_days
    per_day_net_improvement=return_daily-return_bh_daily 
    # improvement = (return_daily - return_bh_daily) / abs(return_bh_daily)
    if per_day_net_improvement > 0 and profit > 0:
    # if True==1:
        if printResults:
            print('********************************')
            print  (f'     {name}')
            print(f'Owned for {inMarket} days of {total_days} days  {round((inMarket/total_days)*100)} %')
            print(f'Profit: ${profit:.2f} vs ${buyhold_profit:.2f}')
            print(f'Owned for {inMarket} days of {total_days} days  {round((inMarket/total_days)*100)} %')
            print(f'Profit per day: ${profit_per_day:.2f} vs ${buyhold_per_day:.2f} per day')
            print(f'Return %: {return_p*100:.2f}% vs {return_bh*100:.2f}%')
            print(f'Return per day: {return_daily*100:.2f}% vs {return_bh_daily*100:.2f}% vs {daily_risk_free_rate*100:.4f}% per day')
            # print(f'{improvement*100:.2f}% better than buy and hold')
        alpha, beta, sharpe_ratio, sortino_ratio=regression_analysis(df, df2, starting_cash,printResults=printResults)
        result={'stock':name,'profit':profit,'buyhold_profit':buyhold_profit,'profit_per_day':profit_per_day,'buyhold_per_day':buyhold_per_day,'return_p':return_p,'return_bh':return_bh,'return_daily':return_daily,'return_bh_daily':return_bh_daily,'alpha':alpha,'beta':beta,'sharpe_ratio':sharpe_ratio,'sortino_ratio':sortino_ratio}
        winList.append(result)
    else:
        loseList.append(name)
    return winList,loseList

# Perform regression analysis
def regression_analysis(df, df3, starting_cash, risk_free_rate=0.02, trading_days=252,printResults=False):
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df1['strategy_returns'] = df['equity'].pct_change()
    df2['benchmark_returns'] = df3['Close'].pct_change()

    df1.fillna(0, inplace=True)
    df2.fillna(0, inplace=True)
    trading_days=int(len(df/2))
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

# Plot the results
# import matplotlib.pyplot as plt
# import mplfinance as mpf

# import matplotlib.pyplot as plt
# import mplfinance as mpf

# import matplotlib.pyplot as plt
# import mplfinance as mpf

# import matplotlib.pyplot as plt



def plot_closing_prices(df, stock, show=True, save=False):
    """
    Plot the closing prices for the given stock with buy and sell signals.

    Parameters:
    df (DataFrame): DataFrame containing stock data with 'Close', 'BUY', and 'SELL' columns.
    stock (str): The stock symbol.

    Returns:
    None
    """
    
    # pdb.set_trace()
    # Get image dimensions
    width, height = get_image_dimensions('ICE.png')
    
    # Convert dimensions to inches for matplotlib (assuming 100 DPI)
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
        
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Adjust figsize as needed
    
    # Plot closing prices
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    
    # Plot buy signals
    ax.scatter(df.index, df['BUY']*.97 , label='Buy Signal', color='green', marker='^',s=50)
    
    # # Plot sell signals
    ax.scatter(df.index, df['SELL']*1.03 , label='Sell Signal', color='red', marker='v',s=50)
        

    
    # Set title and labels
    ax.set_title(f'{stock} Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Add legend
    ax.legend()
    
    # Display the plot
    plt.tight_layout()
    if save:
        plt.savefig(f'{stock}_signals.png')
    if show:
        plt.show()

def plot_results(df, stock):
    """
    Plot the candlestick chart for the given stock with buy and sell signals.

    Parameters:
    df (DataFrame): DataFrame containing stock data with 'BUY' and 'SELL' columns.
    stock (str): The stock symbol.

    Returns:
    None
    """
    df = df.tail(252)
    
    apds = [
        mpf.make_addplot(df['BUY'] * .9, type='scatter', markersize=50, marker='^', color='green', panel=0),
        mpf.make_addplot(df['SELL'] * 1.1, type='scatter', markersize=50, marker='v', color='red', panel=0),
    ]

    mpf.plot(df, type='candle', style='charles', title=f'{stock} Candlestick Chart', ylabel='Price',
             volume=False, figratio=(35, 7), figscale=1, addplot=apds, tight_layout=True)
    plt.show()

# Main function to run the entire process
def main():
    useOldData=True
    printResults=True  
    plot=True
    savePlot=True
    show=False
    save=True
    plot_stock = 'ANET'
    i='1d'
    p='2y'
    tickers=getRandiTickers(10)
    print(tickers)
    
    fullPath=makePath(i,p,useOldData=useOldData)
    print(fullPath)
    # get_backtest_data(tickers,period=p,interval=i,path=fullPath)     
    stock2 = 'SPY'
    _, df2 = load_data(f'{stock2}.csv',directory=fullPath)
 
    winList=[]
    loseList=[]

    # files = [f for f in os.listdir(fullPath) if f.endswith('.csv')]
    files=[f for f in os.listdir(fullPath)]
    for file in files:
        if file.endswith('.pkl'):
            files.remove(file)
    print(files)
            

    # files = ['msft.csv','gs.csv']
    starting_cash = 10000.0
    risk_free_rate = 0.02
    print(f'fullPath:{fullPath}')
    for file in files:
        print(f'file: {file}')
        if file.endswith('.csv'):
            name=file.split('.')[0]
            stock, df = load_data(file=file,directory=fullPath)
        else:
            name=file
            stock, df = load_data(file=file,directory=fullPath)
        df['position']=0
        trading_days = 252
        # trading_days=len(df/2)
        daily_risk_free_rate = (1 + risk_free_rate) ** (1 / trading_days) - 1
        df = calculate_indicators(df)
        df = apply_market_conditions(df, df2)
        df = assign_scores(df)
        total_days = len(df)
        inMarket = df['position'].ne(0).sum()
        df, equity_list = execute_trades(df, starting_cash, daily_risk_free_rate)
        winList,loseList=calculate_metrics(df,df2, equity_list, starting_cash, daily_risk_free_rate, total_days, inMarket,name,winList,loseList,printResults=printResults)
        if plot and stock == plot_stock:
            # print(len(df),len(equity_list))
            df['Equity']=equity_list
            # plot_results(df, stock)
            plot_closing_prices(df, stock, show=show, save=savePlot)
        # print(f'winList: {winList}')
        result_df=pd.DataFrame(winList)
        # result_df=pd.DataFrame(loseList)
        # print(result_df)
    try:
        print(f"^^^^^^^^Alphas^^^^^^^^")
        print(result_df.sort_values(by='alpha',ascending=False))
        print(f"^^^^^^^^Betas^^^^^^^^")
        print(result_df.sort_values(by='beta',ascending=True))
        print(f"^^^^^^^^Sharpe Ratios^^^^^^^^")
        print(result_df.sort_values(by='sharpe_ratio',ascending=False))
        print(f"^^^^^^^^Sortino Ratios^^^^^^^^")
        print(result_df.sort_values(by='sortino_ratio',ascending=False))
        # print(f"^^^^^^^^improvement^^^^^^^^")
        # print(result_df.sort_values(by='improvement',ascending=False))
        print(f"^^^^^^^^profit_per_day^^^^^^^^")
        print(result_df.sort_values(by='profit_per_day',ascending=False))
        result_df = pd.DataFrame(winList)
    except:
        print('err')
    print(f'result: {result_df}')
        

# Calculate ranks for each metric
    # try:
    #     result_df['alpha_rank'] = result_df['alpha'].rank(ascending=False)
    #     result_df['beta_rank'] = result_df['beta'].rank(ascending=False)
    #     result_df['sharpe_ratio_rank'] = result_df['sharpe_ratio'].rank(ascending=False)
    #     result_df['sortino_ratio_rank'] = result_df['sortino_ratio'].rank(ascending=False)
    #     result_df['improvement_rank'] = result_df['improvement'].rank(ascending=False)
    #     result_df['profit_per_day_rank'] = result_df['profit_per_day'].rank(ascending=False)

    #     rank_df = result_df[['stock', 'alpha_rank', 'beta_rank', 'sharpe_ratio_rank', 'sortino_ratio_rank', 'improvement_rank', 'profit_per_day_rank']].copy()

    #     # Calculate the average rank
    #     rank_df.loc[:, 'average_rank'] = rank_df[['alpha_rank', 'beta_rank', 'sharpe_ratio_rank', 'sortino_ratio_rank', 'improvement_rank', 'profit_per_day_rank']].mean(axis=1)

        # Display the new DataFrame
    #     print(rank_df.sort_values(by='average_rank'))
    #     # Pickle the DataFrames
    #     result_df_pickle_path = os.path.join(fullPath, 'result_df.pkl')
    #     rank_df_pickle_path = os.path.join(fullPath, 'rank_df.pkl')
    # except:
    #     print('err')
    # if save:
    #     with open(result_df_pickle_path, 'wb') as f:
    #         pickle.dump(result_df, f)

    #     with open(rank_df_pickle_path, 'wb') as f:
    #         pickle.dump(rank_df, f)
            
            
if __name__ == '__main__':
    main()
         
