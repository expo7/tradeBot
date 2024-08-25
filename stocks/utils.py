import matplotlib.pyplot as plt
import mplfinance as mpf
from .models import TradingResult
import os

def save_backtest_plot(df, stock, trading_result_id):
    """
    Save the backtest plot as an image and associate it with the TradingResult instance.

    Parameters:
    df (DataFrame): DataFrame containing stock data.
    stock (str): The stock symbol.
    trading_result_id (int): The ID of the TradingResult instance.

    Returns:
    None
    """
    df = df.tail(300)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    mpf.plot(df, type='candle', style='charles', title=f'{stock} Candlestick Chart', ylabel='Price',
             volume=False, figratio=(35, 7), figscale=1, tight_layout=True)
    
    # Save the plot as an image file
    image_dir = os.path.join('media', 'backtest_plots')
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f'{stock}_backtest_plot.png')
    plt.savefig(image_path)
    plt.close(fig)
    
    # Associate the image with the TradingResult instance
    trading_result = TradingResult.objects.get(id=trading_result_id)
    trading_result.backtest_plot = f'backtest_plots/{stock}_backtest_plot.png'
    trading_result.save()