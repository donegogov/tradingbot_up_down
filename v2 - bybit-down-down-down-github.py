from pybit.unified_trading import HTTP
from time import sleep
from decimal import Decimal, ROUND_DOWN
import requests
import socket
import os
import json
from dotenv import load_dotenv
import json
import time
import requests
from web3 import Web3
import os
import logging
from dotenv import load_dotenv
from web3.middleware import ExtraDataToPOAMiddleware
from eth_utils import event_abi_to_log_topic
import random
from eth_account import Account
from datetime import datetime
import csv
# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = False  # True means your API keys were generated on testnet.bybit.com
START_PRICE_QTY = 130
SYMBOL = 'ETHUSDT'
TOKEN_START_PRICE = 0.0
TOKEN_PROFIT_PRICE = 0.0
COIN = 'ETH'
HELP_COIN = 'USDT'
CONNECTION_ERRORS = (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                     requests.exceptions.Timeout, socket.timeout)
CONNECTION_ERRORS += (ConnectionResetError,)
# ✅ File to store held tokens
HELD_TOKENS_FILE = "held_tokensETH.json"
PRICE_HISTORY_FILE = "price_historyETH.json"
PRICE_HISTORY_FULL_FILE = "price_history_fullETH.json"
TOKEN_PRICES_FILENAME = "token_pricesETH.txt"
ORDER_FILENAME = "ordersETH.txt"



# Connect to the blockchain
RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")


if not RPC_URL or not PRIVATE_KEY:
    raise ValueError("Missing required environment variables. Check your .env file.")

print(f"RPC_URL: {RPC_URL}")
print(f"PRIVATE_KEY: {PRIVATE_KEY[:5]}...")  # Print first 5 characters for verification


w3 = Web3(Web3.HTTPProvider(RPC_URL))
# Add middleware for Proof-of-Authority chains (e.g., Polygon, BSC)                
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
if w3.is_connected():
    print("Connected to the network")
else:
    print("Failed to connect to the network")
AERODROME_ROUTER = w3.to_checksum_address("0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43")  # Confirm actual address

owner_address = w3.eth.account.from_key(PRIVATE_KEY).address
WALLET_ADDRESS = owner_address
print(owner_address)





BYBIT_USER_ID = XXX #sekoj bybit askant ima id kopiraj go tuka pastoral go zameni XXX so tcoeto id
KOLKU_DA_SE_SIMNE_PA_DA_PRODADAM = 0
KOLKU_DA_SE_DIGNE_PA_DA_KUPAM = 0



# Load ERC-20 ABI (Minimal)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "remaining", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_from", "type": "address"},
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "transferFrom",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "owner", "type": "address"},
            {"indexed": True, "name": "spender", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Approval",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    },
]



AERODROME_ROUTER_ABI = json.loads('[{"inputs": [{ "internalType": "uint256", "name": "amountIn", "type": "uint256" },{ "internalType": "uint256", "name": "amountOutMin", "type": "uint256" },{ "internalType": "tuple[]","name": "routes","type": "tuple[]","components": [{ "internalType": "address", "name": "from", "type": "address" },{ "internalType": "address", "name": "to", "type": "address" },{ "internalType": "bool", "name": "stable", "type": "bool" },{ "internalType": "address", "name": "factory", "type": "address" }]},{ "internalType": "address", "name": "to", "type": "address" },{ "internalType": "uint256", "name": "deadline", "type": "uint256" }],"name": "swapExactTokensForTokens","outputs": [],"stateMutability": "nonpayable","type": "function"},{"inputs": [{ "internalType": "uint256", "name": "amountIn", "type": "uint256" },{ "internalType": "tuple[]","name": "routes","type": "tuple[]","components": [{ "internalType": "address", "name": "from", "type": "address" },{ "internalType": "address", "name": "to", "type": "address" },{ "internalType": "bool", "name": "stable", "type": "bool" },{ "internalType": "address", "name": "factory", "type": "address" }]}],"name": "getAmountsOut","outputs": [{"internalType": "uint256[]","name": "amounts","type": "uint256[]"}],"stateMutability": "view","type": "function"}]')


# Token Addresses (Example: WMATIC -> USDC)
USDC_ADDRESS = w3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")  # usdc
WETH_ADDRESS = w3.to_checksum_address("0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619")  # with


SWAP_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "sender", "type": "address"},
        {"indexed": False, "name": "amount0In", "type": "uint256"},
        {"indexed": False, "name": "amount1In", "type": "uint256"},
        {"indexed": False, "name": "amount0Out", "type": "uint256"},
        {"indexed": False, "name": "amount1Out", "type": "uint256"},
        {"indexed": True, "name": "to", "type": "address"},
    ],
    "name": "Swap",
    "type": "event"
}


def telegram(message):
    TOKEN = "8081213042:AAGeIe7hAmrh7EMlPis3YwmgfZzby04jCO4"
    chat_id = "7396126252"
    #message = "hello from your telegram bot"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json()) # this sends the message



def to_checksum(address):
    """Converts an Ethereum address to a checksum address."""
    try:
        return w3.to_checksum_address(address)
    except Exception:
        logging.error(f"⚠️ Invalid address: {address}")
        return address  # Return original if conversion fails

def get_wallet_token_balance(token_id):
    # Standard ERC-20 balanceOf function ABI
    balance_of_abi = [{"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}]
    
    # Create contract instance
    token_contract = w3.eth.contract(address=Web3.to_checksum_address(token_id), abi=balance_of_abi)
    
    # Get token balance
    balance = token_contract.functions.balanceOf(WALLET_ADDRESS).call()
    
    return balance  # Returns balance in token's smallest unit (e.g., Wei)


# Function to get ERC-20 balance
def get_token_balance_aerodrome(token_address, owner):
    token_contract = w3.eth.contract(address=to_checksum(token_address), abi=ERC20_ABI)
    balance = token_contract.functions.balanceOf(to_checksum(owner)).call()
    return balance

# Function to approve Uniswap V2 Router to spend tokens
def approve_token(token_address, spender, amount):
    token_contract = w3.eth.contract(address=token_address, abi=ERC20_ABI)
    
    tx = token_contract.functions.approve(spender, amount).build_transaction({
        "from": WALLET_ADDRESS,
        "gas": 10000,
        "gasPrice": w3.eth.gas_price,
        "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS),
    })
    try:
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"✅ Approve Transaction Sent: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        return tx_hash
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    return 


def aerodrome_swap(token_in, token_out, amount_in):

    router_contract = w3.eth.contract(address=AERODROME_ROUTER, abi=AERODROME_ROUTER_ABI)
    print('aerodrome_swap')
    print(token_in)
    print(token_out)
    # Build Swap Transaction
    AERODROME_FACTORY_ADDRESS = w3.to_checksum_address('0x420DD381b31aEf6683db6B902084cB0FFECe40Da')
    deadline = w3.eth.get_block("latest")["timestamp"] + 300  # 5-minute deadline
    stable = False
    routes = [{
        "from": w3.to_checksum_address(token_in),
        "to": w3.to_checksum_address(token_out),
        "stable": False,  # If the pool is stable, set this to True
        "factory": AERODROME_FACTORY_ADDRESS
    }]
    amountOut = router_contract.functions.getAmountsOut(amount_in, routes).call()
    amountOutMin = amountOut[1] / (10**get_token_decimals(token_out))
    amountOutMin = amountOutMin * 0.998
    amountOutMin = amountOutMin * (10**get_token_decimals(token_out))
    amountOutMin = int(amountOutMin)
    swap_tx = router_contract.functions.swapExactTokensForTokens(
        amount_in,
        amountOutMin,
        routes,  
        w3.to_checksum_address(WALLET_ADDRESS),
        deadline
    ).build_transaction({
        "from": w3.to_checksum_address(WALLET_ADDRESS),
        "gas": 3000,
        "gasPrice": int(w3.eth.gas_price * 1.02),
        "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS),
    })

    # Sign and Send Transaction
    signed_swap_tx = w3.eth.account.sign_transaction(swap_tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_swap_tx.raw_transaction)
    print(f"✅ Swap Transaction Sent: {tx_hash.hex()}")
    try:
    # Wait for Confirmation
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Swap Successful! Tx Hash: {tx_hash.hex()}")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    time.sleep(1)
  
START_TRADING_APPROVE = True

def autoTrade(token_in, token_out):
    global START_TRADING_APPROVE
    try:
        token_in = w3.to_checksum_address(token_in)
        token_out = w3.to_checksum_address(token_out)
        if START_TRADING_APPROVE == True:
            approve_token('0x4200000000000000000000000000000000000006', AERODROME_ROUTER, 10)
            sleep(1)
            approve_token(w3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"), AERODROME_ROUTER, 10000)
            sleep(1)
            START_TRADING_APPROVE = False

        tiken_in_balance = get_token_balance_aerodrome(token_in, WALLET_ADDRESS)
        
        if token_in == w3.to_checksum_address("0x4200000000000000000000000000000000000006"):
            tiken_in_balance -= 0.0005
            save_orders('ETH', 'sell', tiken_in_balance, 'down')
        else:
            save_orders('ETH', 'buy', tiken_in_balance, 'up')
        

        aerodrome_swap(token_in, token_out, tiken_in_balance)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    return 



def make_order(symbol, side, qty, direct):
    save_orders(symbol, side, qty, direct) 
    telegram(f'order created down down down {symbol}, {side}, {qty}, {direct}')   
    print(session.place_order(
        category="spot",
        symbol=symbol,
        side=side,
        orderType="Market",
        qty=qty,
        orderFilter="Order",
    ))
    

def make_tp_order(symbol, side, qty):
    print(session.place_order(
        category="spot",
        symbol=symbol,
        side=side,
        orderType="Market",
        qty=qty,
        orderFilter="Order",
    ))

def get_token_price(symbol):
    response_price_dict = session.get_tickers(
        category="spot",
        symbol=symbol,
    )
    token_price = response_price_dict['result']['list'][0]['ask1Price']
    print('price')
    print(token_price)

    return token_price

def get_token_balance(coin):
    response_dict = session.get_coin_balance(
        accountType="UNIFIED",
        coin=coin,
        memberId="BYBIT_USER_ID"
    )
    token_wallet_balance = float(response_dict['result']['balance']['walletBalance'])

    return token_wallet_balance

# ✅ Save Held Tokens to File (every time we buy/sell)
def save_held_tokens():
    global held_tokens, held_token_prices
    
    data = {
        "held_tokens": list(held_tokens),
        "held_token_prices": held_token_prices
    }
    
    with open(HELD_TOKENS_FILE, "w") as file:
        json.dump(data, file, indent=4)


# ✅ Load Held Tokens from File (at startup)
def load_held_tokens():
    global held_tokens, held_token_prices

    if os.path.exists(HELD_TOKENS_FILE):
        with open(HELD_TOKENS_FILE, "r") as file:
            data = json.load(file)
            held_tokens = set(data.get("held_tokens", []))
            held_token_prices = data.get("held_token_prices", {})
        print(f"🔄 Loaded {len(held_tokens)} held tokens from file.")


def find_price_jump(token_price_history, min_x, percentage_threshold=0.03):
    prices = token_price_history[-min_x:]  # Get the last min_x prices
    min_price = float('inf')  # Start with a very high number
    last_price = None  # Store the last price when threshold is crossed
    price_change = 0.0

    for price in prices:
        if price < min_price:
            min_price = price  # Update min_price if a new lower value is found
        
    price_change_temp = (token_price_history[-1] - min_price) / min_price
    print(f"cenata se promenila od minimalista cena za {price_change_temp} procenti")
    if price_change_temp >= percentage_threshold:
        price_change = price_change_temp

    #ako e pod threshold promenata na cenata vrati 0
    if price_change == 0:
        return 0

    return price_change  # ako e nad threshold vrati ja promenata na cenata


# **Load price history from file**
def load_price_history():
    if os.path.exists(PRICE_HISTORY_FILE):
        with open(PRICE_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

# **Load price history from file**
def load_price_history_full():
    if os.path.exists(PRICE_HISTORY_FULL_FILE):
        with open(PRICE_HISTORY_FULL_FILE, "r") as f:
            return json.load(f)
    return {}

# **Save price history to file**
def save_price_history(price_history):
    with open(PRICE_HISTORY_FILE, "w") as f:
        json.dump(price_history, f)

# **Save price history to file**
def save_price_history_full(price_history):
    with open(PRICE_HISTORY_FULL_FILE, "w") as f:
        json.dump(price_history, f)

def save_prices(start_price, profit_price):
    with open(TOKEN_PRICES_FILENAME, "w") as file:
        file.write(f"{start_price},{profit_price}")


def load_prices():
    if os.path.exists(TOKEN_PRICES_FILENAME):
        with open(TOKEN_PRICES_FILENAME, "r") as file:
            data = file.read().strip()
            if data:
                start_price, profit_price = map(float, data.split(","))
                return start_price, profit_price
    return None, None  # Default values if file doesn't exist



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import random
import pandas as pd

def detect_spikes_dynamic1(prices, min_spike=3, max_spike=300, recovery_threshold=0.053, max_time_range=20):
    """
    Detects significant price spikes by comparing each price with previous values across random time ranges.

    Parameters:
        prices (list): List of prices.
        min_spike (float): Minimum valid spike size in absolute price movement.
        max_spike (float): Maximum valid spike size.
        recovery_threshold (float): Minimum price increase after a down spike to count as recovery.
        max_time_range (int): Maximum lookback range (in number of data points). Time ranges will be randomly chosen up to this limit.

    Returns:
        list: List of tuples [(index, time_range, 'up'/'down'/'recovery', price change), ...]
    """
    df = pd.DataFrame({'price': prices})
    
    max_time_range = 7200  # Last 20 hours of stored prices
    # Generate random time ranges between 1 and max_time_range
    random_time_ranges = list(range(1, max_time_range + 1, 19))

    real_spikes = []
    last_down_index = {t: None for t in random_time_ranges}
    last_down_price = {t: None for t in random_time_ranges}

    for idx in range(max(random_time_ranges), len(df)):
        for time_range in random_time_ranges:
            past_price = df['price'].iloc[idx - time_range]
            current_price = df['price'].iloc[idx]
            change = round(float(current_price - past_price), 4)

            if min_spike <= abs(change) <= max_spike:
                direction = "up" if change > 0 else "down"
                real_spikes.append((idx, time_range, direction, round(change, 4)))

    return real_spikes



import pandas as pd

def detect_spikes_dynamic(prices, min_spike=3.0, max_spike=300.0, max_time_range=3000):
    """
    Detects price spikes (up or down) based on absolute change compared to past prices over multiple lookback periods.

    Parameters:
        prices (list of float): Price history.
        min_spike (float): Minimum price change to be considered a spike.
        max_spike (float): Maximum spike allowed to avoid outliers or bad data.
        max_time_range (int): How many data points to look back (default 7200 for ~20 hours with 1s resolution).

    Returns:
        list of tuples: [(index, time_range, 'up'/'down', price_change), ...]
    """
    df = pd.DataFrame({'price': prices})

    time_ranges = list(range(1, max_time_range + 1, 19))  # every ~19 sec until max_time_range
    spikes = []

    for idx in range(max(time_ranges), len(df)):
        current_price = df['price'].iloc[idx]
        for t in time_ranges:
            past_price = df['price'].iloc[idx - t]
            change = round(float(current_price - past_price), 4)

            if min_spike <= abs(change) <= max_spike:
                direction = "up" if change > 0 else "down"
                spikes.append((idx, t, direction, change))

    print('spikes')
    print(spikes)
    return spikes


def detect_spikes_full_range(prices, min_spike=3.0, max_spike=300.0):
    """
    Detects spikes of $min_spike or more between any two points in the full price history.

    Parameters:
        prices (list of float): ETH price history.
        min_spike (float): Minimum spike size to be considered valid.
        max_spike (float): Maximum spike size to filter out extreme outliers.

    Returns:
        List of tuples: (start_index, spike_index, 'up' or 'down', price_change)
    """
    spikes = []

    for i in range(len(prices)):
        start_price = prices[i]
        for j in range(i + 1, len(prices)):
            current_price = prices[j]
            change = round(current_price - start_price, 4)

            if min_spike <= abs(change) <= max_spike:
                direction = "up" if change > 0 else "down"
                spikes.append((i, j, direction, change))
                break  # only first valid spike after i

    return spikes


import numpy as np

def calculate_dynamic_noise(prices, window=600, multiplier=2.6):
    """
    Dynamically calculates the noise threshold based on recent volatility.
    
    Args:
        prices (list of float): Price history.
        window (int): How many previous steps to use for volatility estimation.
        multiplier (float): How aggressive the spike detection should be.

    Returns:
        noise_threshold (float): Minimum size of movement to NOT be considered noise.
    """
    if len(prices) < window + 1:
        return 0.0  # Not enough data

    diffs = np.diff(prices[-(window+1):])
    std_dev = np.std(diffs)

    return std_dev * multiplier



def detect_biggest_clean_down_to_last(prices, min_spike=7.0, max_spike=2300.0, max_noise=3.0):
    """
    Detects the biggest downward spike to the last price with all intermediate movements below max_noise.

    Parameters:
        prices (list of float): Price history
        min_spike (float): Minimum drop to be considered
        max_spike (float): Maximum drop to avoid outliers
        max_noise (float): Max allowed fluctuation between start and end

    Returns:
        Tuple: (start_index, last_index, 'down', price_change) or None
    """
    if len(prices) < 2:
        return None
    noise_threshold = calculate_dynamic_noise(prices)
    last_price = prices[-1]
    last_index = len(prices) - 1

    best_spike = None
    largest_drop = 0
    spikes = []
    for i in range(last_index - 1, -1, -1):
        change = round(last_price - prices[i], 4)

        #if change >= min_spike and change <= max_spike:
        if abs(change) > noise_threshold:
            # Check intermediate noise
            noisy = False
            for k in range(i + 1, last_index):
                if abs(prices[k] - last_price) > noise_threshold:
                    noisy = True
                    break

            if not noisy and change > largest_drop:
                largest_drop = change
                best_spike = (i, last_index, 'down', round(-change, 4))  # return as negative drop
                spikes.append(best_spike)
    

    # Find first upward spike FROM last price (going backward to earlier price)
    up_spike = None
    for j in range(last_index - 1, -1, -1):  # backwards
        change = round(prices[j] - last_price, 4)
        if change >= min_spike and change <= max_spike:
            up_spike = (last_index, j, 'up', change)
            spikes.append(up_spike)
            break

    return spikes



def save_spike_results(results, filename="spike_resultsETH.txt"):
    with open(filename, "w") as f:
        for entry in results:
            f.write(f"{entry}\n")






import numpy as np
from collections import defaultdict

def format_spikes_last_only(spike_results):
    last_spikes = {}  # Dictionary to store the last spike per time range

    for idx, time_range, spike_type, change in spike_results:
        change = round(float(change), 4)  # Ensure it's a standard float
        last_spikes[time_range] = (idx, time_range, spike_type, change)  # Store only the last occurrence

    # Convert to a list of tuples
    formatted_spikes = list(last_spikes.values())

    return formatted_spikes


def save_orders(symbol, side, qty, direct):
    with open(ORDER_FILENAME, "a") as file:
        file.write(f"{symbol},{side},{qty}, {direct} \n")


import requests

def get_price_cmc(symbol="ETH", convert="USDT", cmc_api_key="581dc005-ad4a-4b19-aa4b-f3c3bc9ed562"):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {
        "X-CMC_PRO_API_KEY": cmc_api_key
    }
    params = {
        "symbol": symbol,
        "convert": convert
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    try:
        price = data["data"][symbol]["quote"][convert]["price"]
        return round(price, 4)
    except KeyError:
        print("❌ Failed to fetch price:", data)
        return None



import numpy as np
import pandas as pd

def smooth_prices(prices, window=5):
    """
    Smooth prices using rolling median to reduce noise.
    """
    series = pd.Series(prices)
    smoothed = series.rolling(window=window, center=True, min_periods=1).median()
    return smoothed.tolist()

def calculate_dynamic_noise(prices, window=30, multiplier=2.5):
    """
    Dynamically calculate noise threshold based on recent volatility.
    """
    if len(prices) < window + 1:
        return 0.0
    diffs = np.diff(prices[-(window+1):])
    std_dev = np.std(diffs)
    return std_dev * multiplier

def detect_clean_spikes1(prices, smooth_window=5, noise_window=7200, multiplier=2.5):
    """
    Detects spikes after smoothing and filtering by dynamic noise level.
    """
    smoothed = smooth_prices(prices, window=smooth_window)
    noise_threshold = calculate_dynamic_noise(smoothed, window=noise_window, multiplier=multiplier)

    spikes = []
    for i in range(1, len(smoothed)):
        change = smoothed[i] - smoothed[i-1]
        if abs(change) > noise_threshold:
            spikes.append((i-1, i, round(change, 4)))

    return spikes

def detect_clean_spikes4(prices, smooth_window=5, noise_window=6000, multiplier=2.5, lookback=3000, min_spike_usd=7.0):
    """
    Detects biggest clean spike in the last `lookback` steps (e.g. 3 hours if data every 10s).
    Only detects spikes larger than `min_spike_usd`.
    """
    if len(prices) < lookback + 2:
        return []

    smoothed = smooth_prices(prices, window=smooth_window)
    noise_threshold = calculate_dynamic_noise(smoothed, window=noise_window, multiplier=multiplier)
    current_price = smoothed[-1]

    spikes = []
    max_spike = 0

    for i in range(-2, -lookback - 1, -1):
        past_price = smoothed[i]
        change = current_price - past_price

        if abs(change) >= max(noise_threshold, min_spike_usd):  # use whichever is higher
            direction = "up" if change > 0 else "down"
            spikes.append((len(prices) + i, len(prices) - 1, direction, round(change, 4), past_price, current_price))

    return spikes

def detect_clean_spikes5(prices, smooth_window=5, noise_window=6000, multiplier=2.5, lookback=3000, min_spike_usd=7.0):
    """
    Compares the last price against past smoothed prices to detect spike.
    """
    if len(prices) < lookback + 2:
        return []

    smoothed = smooth_prices(prices, window=smooth_window)
    noise_threshold = calculate_dynamic_noise(smoothed, window=noise_window, multiplier=multiplier)
    last_price = prices[-1]  # use raw price if you prefer exact
    spikes = []

    for i in range(-2, -lookback - 1, -1):
        past_price = smoothed[i]
        change = last_price - past_price
        if abs(change) >= max(noise_threshold, min_spike_usd):
            direction = "up" if change > 0 else "down"
            spikes.append((len(prices) + i, len(prices) - 1, direction, round(change, 4), past_price, last_price))

    return spikes


def detect_strongest_spike(prices, full_prices, smooth_window=5, noise_window=6000, multiplier=2.5, min_spike_usd=7.0):
    """
    Compares the last price to all previous smoothed prices and returns the strongest spike (up or down).
    """
    if len(prices) < 2:
        return None

    smoothed = smooth_prices(prices, window=smooth_window)
    #print('smoothed')
    #print(smoothed)
    noise_threshold = calculate_dynamic_noise(smoothed, window=noise_window, multiplier=multiplier)
    threshold = estimate_spike_threshold(full_prices)
    print('=========threshold============')
    print(threshold)
    last_price = prices[-1]  # or use smoothed[-1] if you prefer
    best_spike = None
    max_up_change = 0
    max_down_change = 0
    spikes = []
    best_up_spike = None
    best_down_spike = None
    for i in range(len(smoothed) - 1):  # compare all before last
        past_price = smoothed[i]
        change = last_price - past_price
        if abs(change) >= threshold:
            direction = 'up' if change > 0 else 'down'
            if direction == 'up' and abs(change) > max_up_change:
                best_up_spike = (i, len(prices)-1, direction, round(change, 4), past_price, last_price)
                
                max_up_change = change
            elif direction == 'down' and change < max_down_change:
                best_down_spike = (i, len(prices)-1, direction, round(change, 4), past_price, last_price)
                
                max_down_change = change
            #best_spike = (i, len(prices)-1, direction, round(change, 4), past_price, last_price)
            #max_change = change

    if best_up_spike is not None:
        spikes.append(best_up_spike)
    if best_down_spike is not None:
        spikes.append(best_down_spike)

    return spikes, threshold


def estimate_spike_threshold(prices, visual_noise_factor=0.75, min_spike=3.0, max_spike=25.0):
    """
    Estimate a good spike threshold based on total price range.
    - visual_noise_factor ~15% of full-day range is usually noise
    - Returns a threshold big enough to ignore noise but small enough to catch trades
    """
    # Example usage:
    prices = fetch_bybit_ohlcv()
    visual_noise_factor = calculate_real_noise_factor(prices)
    if len(prices) < 2:
        return min_spike
    
    price_range = max(prices) - min(prices)
    noise_cutoff = price_range * visual_noise_factor

    # Clamp to avoid too extreme thresholds
    threshold = max(min_spike, min(noise_cutoff, max_spike))
    return round(threshold, 2)


import requests
import numpy as np

def fetch_bybit_ohlcv(symbol="ETHUSDT", interval="1", limit=1440):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    if data["retCode"] != 0:
        raise Exception(f"Bybit API error: {data['retMsg']}")

    # Get closing prices
    prices = [float(entry[4]) for entry in data["result"]["list"]]
    prices.reverse()
    return prices



def calculate_real_noise_factor(prices, min_factor=0.05, max_factor=0.3):
    diffs = np.abs(np.diff(prices))
    median_diff = np.median(diffs)
    price_range = max(prices) - min(prices)
    if price_range == 0:
        return 0.10
    return round(max(min(median_diff / price_range, max_factor), min_factor), 4)



session = HTTP(
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
    testnet=TESTNET,
)

# ✅ Global variables
held_tokens = set()
held_token_prices = {}
load_held_tokens()
price_history = load_price_history()
full_prices = load_price_history_full()
X = 1900  # ✅ Check last 10 hours dynamically
min_X = 0  # ✅ Check last 5 minutes dynamically
smart_take_profit = False
can_buy = False
can_sell = False
sell_before = False
last_stop_loss_price = 0.0
smallest_price = 0
last_max_price = 0
direct = 'up' 

# Example Usage
TOKEN_START_PRICE, TOKEN_PROFIT_PRICE = load_prices()
if TOKEN_START_PRICE is None or TOKEN_PROFIT_PRICE is None:
    TOKEN_START_PRICE = 0.0  # Default values
    TOKEN_PROFIT_PRICE = 0.0
    

telegram(f'bot started X={X} good luck')
token_history_price = float(get_token_price(SYMBOL))
sleep(5)

while True:
    try:
        token_price = float(get_token_price(SYMBOL))
        grass = get_token_balance(COIN)
        usdt = get_token_balance(HELP_COIN)
        print('GRASS')
        print(grass)
        print('USDT')
        print(usdt)
        # **Price Change Calculation**
        historical_prices = price_history.get(COIN, {}).get("prices", [])
        historical_prices_full = full_prices.get(COIN, {}).get("prices", [])
        # Append latest token price
        if token_price:
            historical_prices.append(token_price)
            historical_prices_full.append(token_price)

        # Keep only last X prices
        price_history[COIN] = {"prices": historical_prices[-X:]}
        save_price_history(price_history)
        full_prices[COIN] = {"prices": historical_prices_full[-X:]}
        save_price_history_full(full_prices)

        # Ensure price change is calculated over 2.5 minutes to 4 hours
        if len(historical_prices) < min_X:
            print(f"🔍 DEBUG: prvo 2 minute da pominat if len(historical_prices) < min_X:")
            sleep(5)
            continue
        
        # Extract price history
        token_price_history = [entry for entry in historical_prices]
        # Extract price history
        token_price_history_full = [entry for entry in historical_prices_full]

        # Ensure we have enough history
        if len(token_price_history) < min_X:
            continue
        short_array = 3
        save_price_history(price_history)
        print(f"🔍 DEBUG: red if can_buy == False and can_sell == False:")
        if can_buy == False and can_sell == False:
            print(f"🔍 DEBUG: vo if can_buy == False and can_sell == False:")
            spike_fluct = 0.0
            # Run detection function
            #spike_results = detect_clean_spikes(token_price_history)
            spike_results, noisee = detect_strongest_spike(
                token_price_history,
                token_price_history_full
            )
            #nested_spikes = format_spikes_last_only(spike_results)  # Format them properly
            # Save results to a text file
            save_spike_results(spike_results)
            spike_fluct = 0
            short_array = 0
            print("Spikes saved to spike_results.txt")
            print(spike_results)
            print("Spikes saved to spike_results.txt")
            # Check if down exists
            for spike in spike_results:
                print('spike')
                print(spike)
                print('spike')
                if spike[2] == "down" and sell_before == False:
                    if smallest_price == 0:
                        smallest_price = spike[3]
                    spike_fluct = spike[3]
                    short_array = spike[0]
                    #print(f"🔍 DEBUG: spike_fluct={spike_fluct} short_array={short_array}")
                    if spike[3] < smallest_price:
                       smallest_price = spike[3]

                elif spike[2] == "up" and sell_before == True:
                    if last_max_price == 0:
                        last_max_price = spike[3]
                    #print("No down detected yet.")
                    spike_fluct = spike[3]
                    short_array = [0]
                    if spike[3] > last_max_price:
                        last_max_price = spike[3]

            #ako iame fluktacii od 6.5 do 16 centi
            print(f"🔍 DEBUG: pred if spike_fluct <= 0 and abs(spike_fluct) >= 0.053: spike_fluct={spike_fluct} short_array={short_array}")
            if smallest_price < 0 and sell_before == False and abs(smallest_price) >= KOLKU_DA_SE_SIMNE_PA_DA_PRODADAM:
                print(f"🔍 DEBUG: vo if spike_fluct <= 0 and abs(spike_fluct) >= 0.053:")
                TOKEN_PROFIT_PRICE = 0.0
                TOKEN_START_PRICE = 0.0
                can_sell = True
                can_buy = False
                price_history[COIN] = {"prices": historical_prices[-1:]}
                save_price_history(price_history)
                direct = 'down or up'
            elif last_max_price > 0 and sell_before == True and last_max_price >= KOLKU_DA_SE_DIGNE_PA_DA_KUPAM:
                print(f"🔍 DEBUG: vo elif spike_fluct >= 0 and spike_fluct >= 4:")
                #TOKEN_PROFIT_PRICE = token_price / 1.0005
                #TOKEN_START_PRICE = token_price * (1 + 0.0005)  # 0.005% greater
                can_buy = True
                can_sell = False
                price_history[COIN] = {"prices": historical_prices[-1:]}
                save_price_history(price_history)
                direct = 'up'

        print(f"🔍 DEBUG: pred kupuvame prodavame can_buy= {can_buy} can_sell= {can_sell} sell_before = {sell_before}")
        if can_sell == True and can_buy == False and sell_before == False:
            #tuka prodavame
            print(f"DEBUG tuka prodavame - token history > token start price and token_start > token_price")
            token_wallet_balance = get_token_balance(COIN)
            token_wallet_balance = Decimal(token_wallet_balance)          
            str_token_wallet_balance = token_wallet_balance.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)  # Keeps only 2 decimals
            side = 'Sell'
            make_order(SYMBOL, side, str_token_wallet_balance, direct)
            #autoTrade(w3.to_checksum_address("0x4200000000000000000000000000000000000006"), w3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"))
            sell_before = True
            can_buy = False
            can_sell = False
        elif can_sell == False and can_buy == True and sell_before == True:
            #tuka kupuvame
            print(f"DEBUG tuka kupuvame token_price > token start price and token_start_price > token history")
            token_wallet_balance_USDT = get_token_balance(HELP_COIN)
            token_wallet_balance_USDT = Decimal(token_wallet_balance_USDT)
            token_wallet_balance_USDT = token_wallet_balance_USDT.quantize(Decimal('0.01'), rounding=ROUND_DOWN)  # Keeps only 2 decimals
            side = 'Buy'
            print(token_wallet_balance_USDT)
            make_order(SYMBOL, side, token_wallet_balance_USDT, direct)
            #autoTrade(w3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"), w3.to_checksum_address("0x4200000000000000000000000000000000000006"))
            sell_before = False
            can_buy = False
            can_sell = False
            last_max_price = 0
            smallest_price = 0

        print('TOKEN_START_PRICE')
        print(TOKEN_START_PRICE)
        print('TOKEN_PROFIT_PRICE')
        print(TOKEN_PROFIT_PRICE)

        token_history_price = token_price
        sleep(5)
    except CONNECTION_ERRORS as e:
        print(f'Éxception  {e}')
    except Exception as e:
        print(f'Éxception  {e}') 

#print(get_token_balance(COIN))
