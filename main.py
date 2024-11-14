import logging
import json
import time
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from importlib.util import find_spec
from binance.client import Client
from binance.exceptions import BinanceAPIException

def setup_logger(config):
    """Configure logging based on config settings"""
    log_level = getattr(logging, config.get('LOG_LEVEL', 'DEBUG'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'

    @staticmethod
    def colorize(text, color):
        return f"{color}{text}{Colors.RESET}"

    @staticmethod
    def by_value(value, threshold=0):
        return Colors.GREEN if value > threshold else Colors.RED if value < threshold else Colors.RESET

class CacheManager:
    """Handles state persistence to JSON file"""
    def __init__(self, filename):
        self.filepath = Path(f"{filename}.json")

    def load(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return None

    def save(self, data):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

class BinanceClient:
    """Handles all Binance API interactions"""
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.symbol_info = {}
        self.last_symbol_info_update = 0

    def fetch_symbol_info(self):
        """Fetch and cache symbol information for futures market"""
        current_time = time.time()
        last_half_hour = current_time - (current_time % 1800)

        if not self.symbol_info or current_time - last_half_hour > 1800:
            try:
                exchange_info = self.client.futures_exchange_info()
                self.symbol_info = {symbol['symbol']: symbol for symbol in exchange_info['symbols']}
                self.last_symbol_info_update = current_time
                logging.debug("Symbol information updated")
            except BinanceAPIException as e:
                logging.error(f"Failed to fetch symbol information: {e}")
        return self.symbol_info

    def fetch_account_info(self):
        """Fetch futures account information"""
        try:
            return self.client.futures_account()
        except BinanceAPIException as e:
            logging.error(f"Failed to fetch account information: {e}")
            return None

    def fetch_positions(self):
        """Fetch current futures positions"""
        try:
            account = self.client.futures_account()
            return [pos for pos in account['positions'] if float(pos['positionAmt']) != 0]
        except BinanceAPIException as e:
            logging.error(f"Failed to fetch positions: {e}")
            return []

    def fetch_mark_price(self, symbol):
        """Fetch current mark price for symbol"""
        try:
            return self.client.futures_mark_price(symbol=symbol)
        except BinanceAPIException as e:
            logging.error(f"Failed to fetch mark price for {symbol}: {e}")
            return None

    def create_futures_order(self, symbol, side, order_type, quantity):
        """Create futures order"""
        try:
            return self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
        except BinanceAPIException as e:
            logging.error(f"Failed to create order: {e}")
            return None

class BalancerStrategy:
    """Main strategy class implementing the balancing logic"""
    def __init__(self):
        load_dotenv('config/.env')
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials not found in config/.env")
            
        with open('config/config.json', 'r') as f:
            self.config = json.load(f)
        setup_logger(self.config)
        
        self.binance = BinanceClient(self.api_key, self.api_secret)
        self.cache = CacheManager('state')
        self.state = self.initialize_state()

    def initialize_state(self):
        """Initialize or load existing state"""
        state = self.cache.load()
        if not state:
            state = {
                "positions": {},
                "symbol_info": {},
                "last_symbol_info_update": 0,
                "allocation": self.config["ALLOCATION"]
            }
            
            self.update_symbol_info(state)
            self.validate_allocation(state)
            self.cache.save(state)
            
        return state

    def update_symbol_info(self, state):
        """Update symbol information in state"""
        symbol_info = self.binance.fetch_symbol_info()
        if symbol_info:
            state["symbol_info"] = {
                symbol: info for symbol, info in symbol_info.items() 
                if symbol in self.config["ALLOCATION"]
            }
            state["last_symbol_info_update"] = time.time()

    def validate_allocation(self, state):
        """Validate allocation configuration and check minimum notional requirements"""
        total_allocation = sum(self.config["ALLOCATION"].values())
        if not 0.99 <= total_allocation <= 1.01:
            raise ValueError(f"Total allocation must be 100%, got {total_allocation*100}%")
            
        base_capital = self.config["BASE_CAPITAL"]
        warnings = []
        
        for symbol, allocation in self.config["ALLOCATION"].items():
            symbol_info = state["symbol_info"].get(symbol, {})
            if not symbol_info:
                warnings.append(f"No symbol info for {symbol}")
                continue
                
            filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
            min_notional = float(filters.get("MIN_NOTIONAL", {}).get("notional", 0))
            min_qty = float(filters.get("LOT_SIZE", {}).get("minQty", 0))
            
            allocated_capital = base_capital * allocation
            if allocated_capital < min_notional:
                warnings.append(
                    f"{symbol}: Allocated capital ${allocated_capital:.2f} is below minimum notional ${min_notional}"
                )
                
        if warnings:
            logging.warning("Allocation warnings:\n" + "\n".join(warnings))

    def calculate_position_sizes(self):
        """Calculate and adjust position sizes based on profit threshold"""
        account_info = self.binance.fetch_account_info()
        if not account_info:
            return
            
        equity = float(account_info["totalWalletBalance"])
        current_positions = {
            pos["symbol"]: {
                "size": float(pos["positionAmt"]),
                "notional": float(pos["notional"]),
                "unrealizedProfit": float(pos["unrealizedProfit"]),
                "realizedProfit": float(pos.get("realizedProfit", 0))
            }
            for pos in account_info["positions"]
            if float(pos["positionAmt"]) != 0
        }
        
        for symbol, allocation in self.config["ALLOCATION"].items():
            target_notional = equity * allocation
            current_pos = current_positions.get(symbol, {
                "notional": 0, 
                "unrealizedProfit": 0, 
                "size": 0,
                "realizedProfit": 0
            })
            
            if abs(current_pos["unrealizedProfit"]) / equity > self.config["PROFIT_THRESHOLD"]:
                self.place_hedge_position(symbol, current_pos, equity)

    def place_hedge_position(self, symbol, position, equity):
        """Place hedge order for a position"""
        hedge_size = abs(position["unrealizedProfit"])
        symbol_info = self.state["symbol_info"][symbol]
        
        min_qty = float(symbol_info["filters"][1]["minQty"])
        step_size = float(symbol_info["filters"][1]["stepSize"])
        
        precision = len(str(step_size).split(".")[-1])
        hedge_qty = round(hedge_size / position["notional"] * abs(position["size"]), precision)
        
        if hedge_qty >= min_qty:
            side = "SELL" if position["size"] > 0 else "BUY"
            order = self.binance.create_futures_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=hedge_qty
            )
            
            if order:
                logging.info(
                    f"{Colors.colorize('Opening hedge position', Colors.CYAN)} for {symbol}: "
                    f"{hedge_qty} contracts ({Colors.colorize(f'${hedge_size:.2f}', Colors.YELLOW)})"
                )
                self.rebalance_positions(equity)

    def rebalance_positions(self, equity):
        """Rebalance positions after hedge to maintain target allocation"""
        current_positions = self.binance.fetch_positions()
        
        realized_profits = sum(
            float(pos["realizedProfit"])
            for pos in current_positions
            if pos["symbol"] in self.config["ALLOCATION"]
        )
        
        rebalance_requirements = {}
        total_required = 0
        
        for symbol, allocation in self.config["ALLOCATION"].items():
            target_notional = equity * allocation
            main_position = next(
                (pos for pos in current_positions 
                 if pos["symbol"] == symbol and float(pos["positionAmt"]) != 0),
                None
            )
            
            if main_position:
                current_notional = float(main_position["notional"])
                notional_diff = abs(target_notional - current_notional)
                
                symbol_info = self.state["symbol_info"][symbol]
                min_notional = float(symbol_info["filters"][0]["notional"])
                
                if notional_diff > min_notional:
                    rebalance_requirements[symbol] = notional_diff
                    total_required += notional_diff
        
        if total_required > realized_profits:
            logging.warning(
                f"{Colors.colorize('Insufficient realized profit for rebalancing', Colors.YELLOW)}\n"
                f"Required: ${total_required:.2f}\n"
                f"Available: ${realized_profits:.2f}\n"
                f"Missing: ${Colors.colorize(f'{total_required - realized_profits:.2f}', Colors.RED)}\n"
                "Waiting for more profit before rebalancing..."
            )
            return
        
        self.execute_rebalancing(rebalance_requirements, equity)

    def execute_rebalancing(self, rebalance_requirements, equity):
        """Execute rebalancing orders"""
        current_positions = self.binance.fetch_positions()
        
        for symbol, required_notional in rebalance_requirements.items():
            main_position = next(
                (pos for pos in current_positions 
                 if pos["symbol"] == symbol and float(pos["positionAmt"]) != 0),
                None
            )
            
            if not main_position:
                continue
                
            current_notional = float(main_position["notional"])
            target_notional = equity * self.config["ALLOCATION"][symbol]
            price = float(self.binance.fetch_mark_price(symbol)["markPrice"])
            
            notional_diff = target_notional - current_notional
            qty_diff = abs(notional_diff) / price
            
            symbol_info = self.state["symbol_info"][symbol]
            step_size = float(symbol_info["filters"][1]["stepSize"])
            precision = len(str(step_size).split(".")[-1])
            qty = round(qty_diff, precision)
            
            if qty >= float(symbol_info["filters"][1]["minQty"]):
                side = "BUY" if notional_diff > 0 else "SELL"
                order = self.binance.create_futures_order(
                    symbol=symbol,
                    side=side,
                    order_type="MARKET",
                    quantity=qty
                )
                
                if order:
                    logging.info(
                        f"{Colors.colorize('Rebalancing', Colors.MAGENTA)} {symbol}: "
                        f"{side} {qty} contracts to match {self.config['ALLOCATION'][symbol]*100:.1f}% allocation"
                    )

    def display_status(self):
        """Display current strategy status"""
        account_info = self.binance.fetch_account_info()
        if not account_info:
            return
            
        equity = float(account_info["totalWalletBalance"])
        positions = {
            pos["symbol"]: pos 
            for pos in account_info["positions"]
            if float(pos["positionAmt"]) != 0
        }
        
        logging.info("\n" + "="*100)
        logging.info(f"Equity: {Colors.colorize(f'${equity:.2f}', Colors.CYAN)}")
        logging.info("\nPosition Details:")
        
        header = (
            f"{'Symbol':<8} | {'Size':>10} | {'Allocation':>9} | {'Target %':>8} | "
            f"{'Current %':>9} | {'Realized PnL':>12} | {'Unrealized PnL':>14} | {'Hedge Size':>10}"
        )
        logging.info("-"*100)
        logging.info(header)
        logging.info("-"*100)
        
        for symbol, allocation in self.config["ALLOCATION"].items():
            pos = positions.get(symbol, {})
            current_notional = float(pos.get("notional", 0))
            current_allocation = current_notional / equity if equity > 0 else 0
            upnl = float(pos.get("unrealizedProfit", 0))
            rpnl = float(pos.get("realizedProfit", 0))
            size = float(pos.get("positionAmt", 0))
            
            hedge_positions = [
                p for p in account_info["positions"]
                if p["symbol"] == symbol and float(p["positionAmt"]) * size < 0
            ]
            hedge_size = abs(float(hedge_positions[0]["positionAmt"])) if hedge_positions else 0
            
            logging.info(
                f"{symbol:<8} | {size:>10.4f} | "
                f"{Colors.colorize(f'{current_allocation*100:>8.1f}%', Colors.by_value(current_allocation-allocation))} | "
                f"{allocation*100:>7.1f}% | "
                f"{current_allocation*100:>8.1f}% | "
                f"{Colors.colorize(f'${rpnl:>11.2f}', Colors.by_value(rpnl))} | "
                f"{Colors.colorize(f'${upnl:>13.2f}', Colors.by_value(upnl))} | "
                f"{hedge_size:>10.4f}"
            )

    def run(self):
        """Main strategy loop"""
        while True:
            try:
                self.update_symbol_info(self.state)
                self.calculate_position_sizes()
                self.display_status()
                self.cache.save(self.state)
                time.sleep(self.config["UPDATE_INTERVAL"])
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(5)

def install_requirements():
    """Install required packages if not present"""
    required_packages = [
        'python-binance',
        'python-dotenv'
    ]

    for package in required_packages:
        if not find_spec(package.replace('-', '_')):
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logging.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to install {package}: {e}")
                sys.exit(1)

if __name__ == "__main__":
    install_requirements()
    strategy = BalancerStrategy()
    strategy.run() 