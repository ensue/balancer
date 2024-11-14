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
        self.sync_server_time()

    def sync_server_time(self):
        try:
            print("Synchronizuję czas z serwerem Binance...")
            server_time = self.client.get_server_time()
            diff_time = server_time['serverTime'] - int(time.time() * 1000)
            self.client.timestamp_offset = diff_time
            print(f"Czas zsynchronizowany. Offset: {diff_time} ms")
        except BinanceAPIException as e:
            print(f"Błąd podczas synchronizacji czasu: {str(e)}")
            logging.error(f"Błąd podczas synchronizacji czasu: {str(e)}")
            raise
        except Exception as e:
            print(f"Nieoczekiwany błąd podczas synchronizacji czasu: {str(e)}")
            logging.error(f"Nieoczekiwany błąd podczas synchronizacji czasu: {str(e)}")
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
            print("Pobieram informacje o koncie...")
            account = self.client.futures_account()
            print(f"Pobrano informacje o koncie. Saldo: {account.get('totalWalletBalance', 'N/A')} USDT")
            return account
        except BinanceAPIException as e:
            print(f"Błąd podczas pobierania informacji o koncie: {str(e)}")
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
            print(f"Tworzę zlecenie: {symbol} {side} {quantity} {order_type}")
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=self._format_quantity(symbol, quantity)
            )
            print(f"Zlecenie utworzone: {order}")
            return order
        except BinanceAPIException as e:
            print(f"Błąd Binance API podczas tworzenia zlecenia: {str(e)}")
            raise
        except Exception as e:
            print(f"Nieoczekiwany błąd podczas tworzenia zlecenia: {str(e)}")
            raise

    def _format_quantity(self, symbol: str, quantity: float) -> float:
        """Format quantity according to symbol's precision"""
        info = self.client.futures_exchange_info()
        symbol_info = next(
            (s for s in info['symbols'] if s['symbol'] == symbol),
            None
        )
        if not symbol_info:
            raise ValueError(f"Symbol {symbol} nie znaleziony")
        
        precision = int(symbol_info['quantityPrecision'])
        return round(quantity, precision)

    def get_symbol_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Błąd podczas pobierania ceny dla {symbol}: {str(e)}")
            raise

class BalancerStrategy:
    """Main strategy class implementing the balancing logic"""
    def __init__(self):
        try:
            logging.info("Inicjalizacja BalancerStrategy...")
            self.project_root = Path(__file__).parent
            logging.info(f"Ścieżka projektu: {self.project_root}")
            
            env_path = self.project_root / 'config' / '.env'
            logging.info(f"Szukam pliku .env w: {env_path}")
            
            if not env_path.exists():
                logging.info("Plik .env nie istnieje - tworzę nowy...")
                print(f"\n{Colors.YELLOW}Plik .env nie został znaleziony w {env_path}{Colors.RESET}")
                print(f"{Colors.CYAN}Proszę wprowadzić klucze API Binance:{Colors.RESET}\n")
                
                api_key = input("API Key: ").strip()
                api_secret = input("API Secret: ").strip()
                
                env_path.parent.mkdir(parents=True, exist_ok=True)
                with open(env_path, 'w') as f:
                    f.write(f"API_KEY={api_key}\nAPI_SECRET={api_secret}")
                
                print(f"\n{Colors.GREEN}Klucze zostały zapisane w {env_path}{Colors.RESET}\n")
            
            load_dotenv(env_path)
            logging.info("Załadowano zmienne środowiskowe")
            
            self.api_key = os.getenv('API_KEY')
            self.api_secret = os.getenv('API_SECRET')
            
            if not self.api_key or not self.api_secret:
                raise ValueError(f"Nie znaleziono poświadczeń API w {env_path}")
            
            config_path = self.project_root / 'config' / 'config.json'
            logging.info(f"Wczytuję konfigurację z: {config_path}")
            
            if not config_path.exists():
                raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {config_path}")
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logging.info("Załadowano konfigurację")
            
            setup_logger(self.config)
            logging.info("Skonfigurowano logger")
            
            self.binance = BinanceClient(self.api_key, self.api_secret)
            logging.info("Zainicjalizowano klienta Binance")
            
            self.cache = CacheManager('state')
            logging.info("Zainicjalizowano CacheManager")
            
            self.state = self.initialize_state()
            logging.info("Zainicjalizowano stan")
            
        except Exception as e:
            logging.error(f"Błąd podczas inicjalizacji: {str(e)}")
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
        """Calculate and adjust position sizes based on allocation config"""
        logging.info("Checking positions...")
        account_info = self.binance.fetch_account_info()
        if not account_info:
            logging.error("Failed to fetch account information")
            return
        
        equity = float(account_info["totalWalletBalance"])
        logging.info(f"Current balance: ${equity:.2f}")
        
        current_positions = {
            pos["symbol"]: {
                "size": float(pos["positionAmt"]),
                "notional": float(pos["notional"]),
                "unrealizedProfit": float(pos["unrealizedProfit"])
            }
            for pos in account_info["positions"]
            if float(pos["positionAmt"]) != 0
        }
        
        logging.info(f"Found {len(current_positions)} active positions")
        
        if not current_positions:
            logging.info("No active positions. Initializing allocations according to config...")
            for symbol, allocation in self.config["ALLOCATION"].items():
                try:
                    target_notional = equity * allocation
                    current_price = float(self.binance.get_symbol_price(symbol))
                    target_size = target_notional / current_price
                    
                    logging.info(f"Opening position for {symbol}:")
                    logging.debug(f"Allocation: {allocation*100}%")
                    logging.debug(f"Amount: ${target_notional:.2f}")
                    logging.debug(f"Size: {target_size:.6f}")
                    
                    order = self.binance.create_order(
                        symbol=symbol,
                        side="BUY",
                        quantity=target_size,
                        order_type="MARKET"
                    )
                    logging.info(f"Order executed: {order['orderId']}")
                    
                except Exception as e:
                    logging.error(f"Error opening position for {symbol}: {str(e)}")
        else:
            logging.info("Checking existing positions...")
            for symbol, allocation in self.config["ALLOCATION"].items():
                target_notional = equity * allocation
                current_pos = current_positions.get(symbol, {
                    "notional": 0, 
                    "unrealizedProfit": 0, 
                    "size": 0
                })
                
                logging.debug(f"Checking {symbol}:")
                logging.debug(f"Target value: ${target_notional:.2f}")
                logging.debug(f"Current value: ${current_pos['notional']:.2f}")
                logging.debug(f"Unrealized PnL: ${current_pos['unrealizedProfit']:.2f}")

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
        print("\nPobieram aktualny stan konta...")
        account_info = self.binance.fetch_account_info()
        if not account_info:
            print("Nie udało się pobrać informacji o koncie!")
            return
        
        equity = float(account_info["totalWalletBalance"])
        positions = {
            pos["symbol"]: pos 
            for pos in account_info["positions"]
            if float(pos["positionAmt"]) != 0
        }
        
        print(f"\n{'='*50}")
        print(f"Stan konta: ${equity:.2f}")
        print(f"Liczba aktywnych pozycji: {len(positions)}")
        print(f"{'='*50}\n")
        
        if positions:
            print("Aktywne pozycje:")
            for symbol, pos in positions.items():
                size = float(pos["positionAmt"])
                upnl = float(pos["unrealizedProfit"])
                print(f"{symbol}: {size:.4f} (PnL: ${upnl:.2f})")
        else:
            print("Brak aktywnych pozycji")

    def run(self):
        """Main strategy loop"""
        print("Rozpoczynam główną pętlę strategii...")
        try:
            print("Aktualizuję informacje o symbolach...")
            self.update_symbol_info(self.state)
            
            print("Obliczam rozmiary pozycji...")
            self.calculate_position_sizes()
            
            print("Wyświetlam status...")
            self.display_status()
            
            print("Zapisuję stan...")
            self.cache.save(self.state)
            
            print("Pierwszy cykl zakończony, przechodzę do pętli głównej...")
            
            while True:
                try:
                    self.update_symbol_info(self.state)
                    self.calculate_position_sizes()
                    self.display_status()
                    self.cache.save(self.state)
                    print(f"Czekam {self.config['UPDATE_INTERVAL']} sekund...")
                    time.sleep(self.config["UPDATE_INTERVAL"])
                except Exception as e:
                    print(f"Błąd w głównej pętli: {str(e)}")
                    logging.error(f"Błąd w głównej pętli: {str(e)}", exc_info=True)
                    time.sleep(5)
        except Exception as e:
            print(f"Błąd krytyczny w run(): {str(e)}")
            logging.error(f"Błąd krytyczny w run(): {str(e)}", exc_info=True)
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
    print("Rozpoczynam inicjalizację...")
    install_requirements()
    print("Pakiety zainstalowane, tworzę instancję BalancerStrategy...")
    try:
        strategy = BalancerStrategy()
        print("Instancja utworzona, uruchamiam strategię...")
        strategy.run()
    except Exception as e:
        print(f"Błąd krytyczny: {str(e)}")
        logging.error(f"Błąd krytyczny: {str(e)}", exc_info=True)
        sys.exit(1) 