import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
from config import REQUIRED_COLUMNS
 
def choose_file(window_title="Choose data file", filetypes="*.xlsx *.xls *.csv *.parquet"):
    """Opens a window so the user can select a file"""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=window_title, filetypes=[("files", filetypes)],)
    root.destroy()
 
    if not path:
        raise FileNotFoundError("No file chosen")
    return path

def load_file(path, required_columns=REQUIRED_COLUMNS):
    """Gets a path, loads the entire file into a dataframe that it returns"""
    file_type = Path(path).suffix.lower()
 
    if file_type in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif file_type == ".parquet":
        df = pd.read_parquet(path)
    elif file_type == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
 
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing} \nColumns found: {list(df.columns)}")
 
    return df

def get_ticker_data(df, ticker):
    """Gets a dataframe with all the data, a string with the ticker name, 

       Returns a dataframe with only the ticker's rows and the required columns.
       
       Also prints what it got."""
    ticker = ticker.strip().upper()
    stock = df.loc[df["ticker"] == ticker, list(REQUIRED_COLUMNS)].copy()
 
    if stock.empty:
        raise KeyError(f"{ticker} not found")

    print(f"\n{ticker.strip().upper()}: {len(stock)} rows | {stock['date'].min()} → {stock['date'].max()}\n")

    stock["date"] = pd.to_datetime(stock["date"])
    return stock.sort_values("date", ascending=False).reset_index(drop=True)
 
 