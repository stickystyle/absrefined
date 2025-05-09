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
            # 'data' is the raw dictionary parsed from the TOML file
            data = tomli.load(f) 
        
        # 'config_data' will be the processed and validated dictionary we return
        config_data = {}

        if "audiobookshelf" not in data:
            raise ConfigError("Missing \'audiobookshelf\' section in config")
        abs_config = data["audiobookshelf"]
        if "host" not in abs_config:
            raise ConfigError("Missing \'host\' in \'audiobookshelf\' section")
        if "api_key" not in abs_config:
            raise ConfigError("Missing \'api_key\' in \'audiobookshelf\' section")
        
        config_data["audiobookshelf"] = {
            "host": str(abs_config["host"]),
            "api_key": str(abs_config["api_key"]),
            "timeout": int(abs_config.get("timeout", 30)) # Default timeout
        }
        if config_data["audiobookshelf"]["timeout"] <= 0:
            raise ConfigError("audiobookshelf.timeout must be a positive integer.")

        refiner_config_raw = data.get("refiner", {})
        config_data["refiner"] = {
            "openai_api_url": str(refiner_config_raw.get("openai_api_url", "")),
            "openai_api_key": str(refiner_config_raw.get("openai_api_key", "")),
            "model_name": str(refiner_config_raw.get("model_name", "gpt-4o-mini")), # Default model
            "whisper_model_name": str(refiner_config_raw.get("whisper_model_name", "whisper-1")) # Default whisper
        }
        # Components (ChapterRefiner, AudioTranscriber) will raise errors if essential keys like api_key/url are missing.

        processing_config_raw = data.get("processing", {})
        config_data["processing"] = {
            "search_window_seconds": int(processing_config_raw.get("search_window_seconds", 60)),
            "download_path": str(processing_config_raw.get("download_path", "")), 
        }
        if config_data["processing"]["search_window_seconds"] <= 0:
            raise ConfigError("processing.search_window_seconds must be a positive integer.")

        logging_config_raw = data.get("logging", {})
        config_data["logging"] = {
            "level": str(logging_config_raw.get("level", "INFO")).upper(),
            "debug_files": bool(logging_config_raw.get("debug_files", False)),
        }
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config_data["logging"]["level"] not in valid_log_levels:
            raise ConfigError(f"logging.level must be one of {valid_log_levels}")

        # Read from raw 'data'
        costs_config_raw = data.get("costs", {})
        llm_prompt_cost_raw = costs_config_raw.get("llm_refinement_cost_per_million_prompt_tokens")
        llm_completion_cost_raw = costs_config_raw.get("llm_refinement_cost_per_million_completion_tokens")
        transcription_cost_raw = costs_config_raw.get("audio_transcription_cost_per_minute")

        config_data["costs"] = {
            "llm_refinement_cost_per_million_prompt_tokens": float(llm_prompt_cost_raw) if llm_prompt_cost_raw is not None else 0.0,
            "llm_refinement_cost_per_million_completion_tokens": float(llm_completion_cost_raw) if llm_completion_cost_raw is not None else 0.0,
            "audio_transcription_cost_per_minute": float(transcription_cost_raw) if transcription_cost_raw is not None else 0.0,
        }
        if config_data["costs"]["llm_refinement_cost_per_million_prompt_tokens"] < 0:
            raise ConfigError("costs.llm_refinement_cost_per_million_prompt_tokens cannot be negative.")
        if config_data["costs"]["llm_refinement_cost_per_million_completion_tokens"] < 0:
            raise ConfigError("costs.llm_refinement_cost_per_million_completion_tokens cannot be negative.")
        if config_data["costs"]["audio_transcription_cost_per_minute"] < 0:
            raise ConfigError("costs.audio_transcription_cost_per_minute cannot be negative.")

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