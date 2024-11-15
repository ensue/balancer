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
from dotenv import load_dotenv

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
        
        # Set position mode to one-way
        try:
            self.client.futures_change_position_mode(dualSidePosition=False)
        except BinanceAPIException as e:
            if e.code != -4059:  # Already in one-way mode
                raise
        
        self.sync_server_time()

    def sync_server_time(self):
        try:
            print("Synchronizing time with Binance server...")
            server_time = self.client.get_server_time()
            diff_time = server_time['serverTime'] - int(time.time() * 1000)
            self.client.timestamp_offset = diff_time
            print(f"Time synchronized. Offset: {diff_time} ms")
        except BinanceAPIException as e:
            print(f"Error synchronizing time: {str(e)}")
            logging.error(f"Error synchronizing time: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error during time sync: {str(e)}")
            logging.error(f"Unexpected error during time sync: {str(e)}")
            raise

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
            print("Fetching account information...")
            account = self.client.futures_account()
            print(f"Account information fetched. Balance: {account.get('totalWalletBalance', 'N/A')} USDT")
            return account
        except BinanceAPIException as e:
            print(f"Error fetching account information: {str(e)}")
            logging.error(f"Failed to fetch account information: {e}")
            return None

    def fetch_positions(self):
        """Fetch current futures positions"""
        try:
            return self.client.futures_account()['positions']
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

    def create_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET"):
        """Create a new order"""
        try:
            formatted_qty = self._format_quantity(symbol, quantity)
            if formatted_qty <= 0:
                raise ValueError(f"Invalid quantity: {formatted_qty}")
            
            print(f"Creating order: {symbol} {side} {formatted_qty} {order_type}")
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=formatted_qty
            )
            print("Order created successfully")
            return order
        except BinanceAPIException as e:
            print(f"Binance API error while creating order: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error while creating order: {str(e)}")
            raise

    def _format_quantity(self, symbol: str, quantity: float) -> float:
        """Format quantity according to symbol's precision"""
        symbol_info = self.symbol_info.get(symbol)
        if not symbol_info:
            self.fetch_symbol_info()
            symbol_info = self.symbol_info.get(symbol)
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found")
        
        filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
        lot_size = filters.get("LOT_SIZE", {})
        
        min_qty = float(lot_size.get("minQty", 0))
        step_size = float(lot_size.get("stepSize", 0))
        
        if quantity < min_qty:
            logging.error(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
            return 0
        
        # Calculate precision from stepSize
        precision = 0
        if step_size < 1:
            precision = len(str(step_size).split(".")[-1].rstrip('0'))
        
        # Round to the correct precision and ensure it's a multiple of stepSize
        formatted_qty = float(int(quantity / step_size) * step_size)
        formatted_qty = round(formatted_qty, precision)
        
        # Ensure quantity is not zero after rounding
        if formatted_qty <= 0:
            logging.error(f"Quantity {quantity} rounded to zero for {symbol}")
            return 0
        
        return formatted_qty

    def get_symbol_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error fetching price for {symbol}: {str(e)}")
            raise

class BalancerStrategy:
    """Main strategy class implementing the balancing logic"""
    def __init__(self):
        try:
            logging.info("Initializing BalancerStrategy...")
            self.project_root = Path(__file__).parent
            logging.info(f"Project path: {self.project_root}")
            
            env_path = self.project_root / 'config' / '.env'
            logging.info(f"Looking for .env file in: {env_path}")
            
            if not env_path.exists():
                logging.info("Creating new .env file...")
                print(f"\n{Colors.YELLOW}.env file not found in {env_path}{Colors.RESET}")
                print(f"{Colors.CYAN}Please enter your Binance API keys:{Colors.RESET}\n")
                
                api_key = input("API Key: ").strip()
                api_secret = input("API Secret: ").strip()
                
                env_path.parent.mkdir(parents=True, exist_ok=True)
                with open(env_path, 'w') as f:
                    f.write(f"API_KEY={api_key}\nAPI_SECRET={api_secret}")
                
                print(f"\n{Colors.GREEN}Keys have been saved to {env_path}{Colors.RESET}\n")
            
            load_dotenv(env_path)
            logging.info("Environment variables loaded")
            
            self.api_key = os.getenv('API_KEY')
            self.api_secret = os.getenv('API_SECRET')
            
            if not self.api_key or not self.api_secret:
                raise ValueError(f"API credentials not found in {env_path}")
            
            config_path = self.project_root / 'config' / 'config.json'
            logging.info(f"Loading configuration from: {config_path}")
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logging.info("Configuration loaded")
            
            setup_logger(self.config)
            logging.info("Logger configured")
            
            self.binance = BinanceClient(self.api_key, self.api_secret)
            logging.info("Binance client initialized")
            
            self.cache = CacheManager('state')
            logging.info("CacheManager initialized")
            
            self.state = self.initialize_state()
            logging.info("State initialized")
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

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
        """Calculate position sizes based on allocation"""
        account_info = self.binance.fetch_account_info()
        if not account_info:
            return None

        base_capital = max(self.config["BASE_CAPITAL"], 25)  # Minimum 25 USDT total to meet 5 USDT per position
        positions = {}
        
        for symbol, allocation in self.config["ALLOCATION"].items():
            position_value = base_capital * allocation
            # Upewnij się, że wartość pozycji jest co najmniej 5 USDT
            if position_value < 5:
                logging.warning(f"Skipping {symbol} - position value {position_value:.2f} USDT is below minimum 5 USDT")
                continue
            
            price = self.binance.get_symbol_price(symbol)
            if price:
                size = position_value / price
                positions[symbol] = self.binance._format_quantity(symbol, size)
        
        self.execute_positions(positions)
        return positions

    def execute_positions(self, target_positions):
        """Execute orders to achieve target positions"""
        current_positions = {
            pos["symbol"]: float(pos["positionAmt"])
            for pos in self.binance.fetch_positions()
            if float(pos["positionAmt"]) != 0
        }
        
        for symbol, target_size in target_positions.items():
            current_size = current_positions.get(symbol, 0)
            size_diff = target_size - current_size
            
            if abs(size_diff) > 0:
                # Sprawdź minimalną wartość zlecenia
                price = self.binance.get_symbol_price(symbol)
                order_value = abs(size_diff) * price
                
                if order_value < 5:
                    logging.warning(f"Skipping {symbol} order - value {order_value:.2f} USDT is below minimum 5 USDT")
                    continue
                
                side = "BUY" if size_diff > 0 else "SELL"
                try:
                    self.binance.create_order(
                        symbol=symbol,
                        side=side,
                        quantity=abs(size_diff)
                    )
                    logging.info(f"Executed {side} order for {symbol}: {abs(size_diff)} units")
                except Exception as e:
                    logging.error(f"Failed to execute order for {symbol}: {str(e)}")

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
        print("\nFetching current account status...")
        account_info = self.binance.fetch_account_info()
        if not account_info:
            print("Failed to fetch account information!")
            return
        
        equity = float(account_info["totalWalletBalance"])
        positions = {
            pos["symbol"]: pos 
            for pos in account_info["positions"]
            if float(pos["positionAmt"]) != 0
        }
        
        total_notional = sum(float(pos["notional"]) for pos in positions.values())
        total_hedge = sum(abs(float(pos["positionAmt"])) * float(pos["entryPrice"]) for pos in positions.values())
        
        print(f"\n{'='*85}")
        print(f"Account Balance: ${equity:.2f}    Total Positions: {len(positions)}    Total Notional: ${total_notional:.2f}")
        print(f"Total Hedge Value: ${total_hedge:.2f}")
        print(f"{'='*85}")
        
        if positions:
            headers = ["Symbol", "Size", "Notional $", "Hedge $", "Allocation %"]
            row_format = "│ {:<12} │ {:>10} │ {:>10} │ {:>8} │ {:>15} │"
            separator = "├" + "─"*14 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*10 + "┼" + "─"*17 + "┤"
            
            print("┌" + "─"*14 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*10 + "┬" + "─"*17 + "┐")
            print(row_format.format(*headers))
            print(separator)
            
            for symbol, pos in positions.items():
                size = float(pos["positionAmt"])
                notional = float(pos["notional"])
                hedge_value = abs(size) * float(pos["entryPrice"])
                allocation = (notional / total_notional * 100) if total_notional > 0 else 0
                
                print(row_format.format(
                    symbol,
                    f"{size:.4f}",
                    f"{notional:.2f}",
                    f"{hedge_value:.2f}",
                    f"{allocation:.1f}%"
                ))
            
            print("└" + "─"*14 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*10 + "┴" + "─"*17 + "┘")
        else:
            print("\nNo active positions")

    def run(self):
        """Main strategy loop"""
        print("Starting main strategy loop...")
        try:
            print("Updating symbol information...")
            self.update_symbol_info(self.state)
            
            print("Calculating position sizes...")
            self.calculate_position_sizes()
            
            print("Displaying status...")
            self.display_status()
            
            print("Saving state...")
            self.cache.save(self.state)
            
            print("First cycle completed, entering main loop...")
            
            while True:
                try:
                    self.update_symbol_info(self.state)
                    self.calculate_position_sizes()
                    self.display_status()
                    self.cache.save(self.state)
                    print(f"Waiting {self.config['UPDATE_INTERVAL']} seconds...")
                    time.sleep(self.config["UPDATE_INTERVAL"])
                except Exception as e:
                    print(f"Error in main loop: {str(e)}")
                    print(f"Critical error in main loop: {str(e)}")
                    logging.error(f"Critical error in main loop: {str(e)}", exc_info=True)
                    time.sleep(5)
        except Exception as e:
            print(f"Critical error in run(): {str(e)}")
            logging.error(f"Critical error in run(): {str(e)}", exc_info=True)
            raise

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
    print("Starting initialization...")
    install_requirements()
    print("Packages installed, creating BalancerStrategy instance...")
    try:
        strategy = BalancerStrategy()
        print("Instance created, starting strategy...")
        strategy.run()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        logging.error(f"Critical error: {str(e)}", exc_info=True)
        sys.exit(1) 