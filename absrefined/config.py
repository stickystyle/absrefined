from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib as tomli
except ImportError:
    import tomli

DEFAULT_CONFIG_PATH = Path("config.toml") # Assuming run from root

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration from a TOML file."""
    
    search_path = config_path if config_path else DEFAULT_CONFIG_PATH
    
    # Try searching relative to this file if default path not found initially
    if not search_path.exists() and not config_path:
        # Assumes config.py is in absrefined/, so ../ is the root
        alt_path = Path(__file__).parent.parent / "config.toml" 
        if alt_path.exists():
            search_path = alt_path
        else:
             # If neither default nor relative path works, raise error with original paths
             raise ConfigError(f"Configuration file not found at {DEFAULT_CONFIG_PATH.resolve()} or relative path {alt_path.resolve()}")

    if not search_path.exists():
         raise ConfigError(f"Configuration file not found at {search_path.resolve()}")

    try:
        with open(search_path, "rb") as f:
            config_data = tomli.load(f)
        
        # Basic validation (expand as needed)
        if "audiobookshelf" not in config_data:
            raise ConfigError("Missing 'audiobookshelf' section in config")
        if "host" not in config_data["audiobookshelf"]:
            raise ConfigError("Missing 'host' in 'audiobookshelf' section")
        if "api_key" not in config_data["audiobookshelf"]:
            raise ConfigError("Missing 'api_key' in 'audiobookshelf' section")
            
        # Add validation for other expected sections/keys if critical
        # Example:
        # if "refiner" not in config_data:
        #     raise ConfigError("Missing 'refiner' section in config")
        # if "model_name" not in config_data["refiner"]:
        #     # Warn instead of error? Or provide default?
        #     print("Warning: Missing 'model_name' in 'refiner' section, using default.") 

        return config_data
        
    except tomli.TOMLDecodeError as e:
        raise ConfigError(f"Error parsing TOML file {search_path.resolve()}: {e}")
    except OSError as e:
        raise ConfigError(f"Error reading configuration file {search_path.resolve()}: {e}")


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads and returns the configuration dictionary."""
    # You could implement caching here if desired
    return load_config(config_path)

# Example of how it might be used elsewhere:
# from absrefined.config import get_config
# CONFIG = get_config()
# abs_host = CONFIG.get("audiobookshelf", {}).get("host") 