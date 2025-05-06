import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import queue
import os
import re
import logging
import tempfile
import atexit
import shutil
from pathlib import Path

# --- Import configuration loader ---
from absrefined.config import get_config, ConfigError

# --- Import your existing absrefined components ---
from absrefined.client import AudiobookshelfClient
from absrefined.transcriber import AudioTranscriber
from absrefined.refiner import ChapterRefiner
from absrefined.refinement_tool import ChapterRefinementTool
from absrefined.utils.timestamp import format_timestamp  # Assuming you have this
from absrefined.utils.url_utils import extract_item_id_from_url  # Added import

# --- Try importing audio libraries and set flags --- #
has_simpleaudio = False
# has_pydub = False # Removed
try:
    import simpleaudio as sa
    import wave  # simpleaudio uses this

    has_simpleaudio = True
except ImportError:
    print(
        "Warning: `simpleaudio` or `wave` library not found. Audio playback might be limited."
    )
    print("Please install it using: pip install simpleaudio")

# --- Keep track of temp paths to clean up on exit --- #
_temp_dirs_to_clean = set()


def _cleanup_temp_files():
    """Clean up temporary directories when the application exits."""
    # Use logger if available, otherwise print
    logger = (
        logging.getLogger("_cleanup_temp_files") if "logging" in globals() else None
    )

    if logger:
        logger.info(f"Cleaning up {len(_temp_dirs_to_clean)} temporary directories")
    else:
        print(f"Cleaning up {len(_temp_dirs_to_clean)} temporary directories")

    for temp_dir in _temp_dirs_to_clean:
        if not temp_dir or not os.path.exists(temp_dir):
            continue

        try:
            if logger:
                logger.info(f"Removing directory: {temp_dir}")
            else:
                print(f"Removing directory: {temp_dir}")

            shutil.rmtree(temp_dir)

            if logger:
                logger.debug(f"Successfully removed directory: {temp_dir}")
            else:
                print(f"Successfully removed directory: {temp_dir}")
        except OSError as e:
            error_message = f"Error removing directory {temp_dir}: {e}"
            if logger:
                logger.warning(error_message)
            else:
                print(error_message)


atexit.register(_cleanup_temp_files)

# --- Constants removed, will be from config ---


class AbsRefinedApp:
    def __init__(self, root):
        global _temp_dirs_to_clean  # Moved here
        self.root = root
        self.root.title("ABSRefined GUI")
        self.root.geometry("800x600")  # Adjust as needed

        # --- Initialize Logger FIRST --- #
        self.logger = logging.getLogger(self.__class__.__name__)
        # Logging level is set globally in the main block

        self.task_queue = queue.Queue()
        self.result_data = []
        self.current_item_id = (
            None  # Added to store item ID from last successful processing
        )

        # --- Load Configuration from TOML ---
        self.config = {}
        try:
            # Since gui.py and config.toml are in the root
            config_path = Path("config.toml")
            self.config = get_config(config_path)

            # Extract Audiobookshelf settings (critical)
            self.abs_host = self.config.get("audiobookshelf", {}).get("host")
            self.abs_api_key = self.config.get("audiobookshelf", {}).get("api_key")
            if not self.abs_host or not self.abs_api_key:
                raise ConfigError(
                    "Missing 'host' or 'api_key' in [audiobookshelf] section of config.toml"
                )

            # Extract Refiner settings
            refiner_config = self.config.get("refiner", {})
            self.llm_api_url = refiner_config.get("openai_api_url")
            self.llm_api_key = refiner_config.get("openai_api_key")
            self.default_model = refiner_config.get("model_name", "gpt-4o-mini")
            if not self.llm_api_url or not self.llm_api_key:
                # These are critical for the refiner to function
                raise ConfigError(
                    "Missing 'openai_api_url' or 'openai_api_key' in [refiner] section of config.toml"
                )

            # Extract Processing settings
            processing_config = self.config.get("processing", {})
            self.default_window = processing_config.get("search_window_seconds", 60)

            # Use system temp directory by default
            if (
                "download_path" not in processing_config
                or not processing_config["download_path"]
            ):
                # Create a unique subdirectory in the system temp dir
                temp_subdir = os.path.join(
                    tempfile.gettempdir(), f"absrefined_gui_{os.getpid()}"
                )
                self.download_path = Path(temp_subdir)
                # Record for cleanup on exit
                # global _temp_dirs_to_clean # Removed from here
                _temp_dirs_to_clean.add(str(self.download_path))
                self.logger.info(
                    f"Using system temp directory for downloads: {temp_subdir}"
                )
            else:
                self.download_path = Path(processing_config["download_path"])
                # Also add the configured path to be cleaned up
                # global _temp_dirs_to_clean # Removed from here
                _temp_dirs_to_clean.add(str(self.download_path))

            # Update config with the actual download path
            if "processing" not in self.config:
                self.config["processing"] = {}
            self.config["processing"]["download_path"] = str(self.download_path)

        except ConfigError as e:
            messagebox.showerror(
                "Configuration Error",
                f"Failed to load or parse config.toml:\n{e}\nPlease ensure config.toml exists and is valid in the root directory.",
            )
            root.destroy()
            return
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An unexpected error occurred during configuration loading:\n{e}",
            )
            root.destroy()
            return

        # Create the download path if it doesn't exist
        try:
            self.download_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using download path: {self.download_path.resolve()}")
        except OSError as e:
            messagebox.showerror(
                "Error",
                f"Could not create download directory {self.download_path}: {e}",
            )
            root.destroy()
            return

        # --- Cancellation Event --- #
        self.cancel_event = threading.Event()

        # --- Initialize Cost Config Variables ---
        costs_cfg = self.config.get("costs", {})
        self.llm_prompt_cost_config_val = costs_cfg.get(
            "llm_refinement_cost_per_million_prompt_tokens", 0.0
        )
        self.llm_completion_cost_config_val = costs_cfg.get(
            "llm_refinement_cost_per_million_completion_tokens", 0.0
        )
        self.transcription_cost_config_val = costs_cfg.get(
            "audio_transcription_cost_per_minute", 0.0
        )

        # --- GUI Frames ---
        self.input_frame = ttk.Frame(root, padding="10")
        self.input_frame.grid(row=0, column=0, sticky="ew")

        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.grid(row=1, column=0, sticky="ew")

        self.progress_frame = ttk.Frame(root, padding="10")
        self.progress_frame.grid(row=2, column=0, sticky="ew")

        self.results_frame = ttk.Frame(root, padding="10")
        self.results_frame.grid(row=3, column=0, sticky="nsew")

        self.action_frame = ttk.Frame(root, padding="10")
        self.action_frame.grid(row=4, column=0, sticky="ew")

        self.usage_cost_frame = ttk.Frame(
            root, padding="10"
        )  # New frame for usage/cost
        self.usage_cost_frame.grid(row=5, column=0, sticky="ew")

        # Configure resizing behavior
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)  # Allow results frame to expand

        # --- Input Fields ---
        ttk.Label(self.input_frame, text="Book URL:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.book_url_var = tk.StringVar()
        self.book_url_entry = ttk.Entry(
            self.input_frame, textvariable=self.book_url_var, width=60
        )
        self.book_url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(self.input_frame, text="Window (s):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.window_var = tk.StringVar(value=str(self.default_window))  # From config
        self.window_entry = ttk.Entry(
            self.input_frame, textvariable=self.window_var, width=10
        )
        self.window_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self.input_frame, text="Model:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.model_var = tk.StringVar(value=self.default_model)  # From config
        self.model_entry = ttk.Entry(
            self.input_frame, textvariable=self.model_var, width=30
        )
        self.model_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # --- ABS Credentials Fields Removed ---
        # ... (lines for abs_user_var, abs_pass_var, labels, entries, and disabling logic are deleted) ...

        # Configure input frame column resizing
        self.input_frame.columnconfigure(1, weight=1)

        # --- Control Elements ---
        self.dry_run_var = tk.BooleanVar(value=True)  # Default to dry run for safety
        self.dry_run_check = ttk.Checkbutton(
            self.control_frame,
            text="Dry Run (Don't update server)",
            variable=self.dry_run_var,
            command=self._update_button_states,
        )
        self.dry_run_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.process_button = ttk.Button(
            self.control_frame,
            text="Start Processing",
            command=self.start_processing_thread,
        )
        self.process_button.grid(row=0, column=1, padx=5, pady=5)

        self.cancel_button = ttk.Button(
            self.control_frame,
            text="Cancel",
            command=self.cancel_task,
            state=tk.DISABLED,
        )
        self.cancel_button.grid(row=0, column=2, padx=5, pady=5)

        self.control_frame.columnconfigure(1, weight=1)  # Push button to the right
        # Column 2 (Cancel button) should not expand

        # --- Progress Bar ---
        self.progress_label_var = tk.StringVar(value="Progress: Idle")
        self.progress_label = ttk.Label(
            self.progress_frame, textvariable=self.progress_label_var
        )
        self.progress_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.progress_bar = ttk.Progressbar(
            self.progress_frame, orient="horizontal", mode="determinate", length=400
        )
        self.progress_bar.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        self.progress_frame.columnconfigure(0, weight=1)

        # --- Results Area (Placeholder - will be populated dynamically) ---
        # Using a Canvas + Frame for scrollability
        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_scrollbar = ttk.Scrollbar(
            self.results_frame, orient="vertical", command=self.results_canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.results_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            ),
        )

        self.results_canvas_window = self.results_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )  # Store window id
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)

        # Mouse wheel scrolling for the results canvas
        self.results_canvas.bind_all(
            "<MouseWheel>", self._on_mousewheel
        )  # Bind to all for simplicity on macOS/Windows
        self.results_canvas.bind("<Button-4>", self._on_mousewheel)  # Linux scroll up
        self.results_canvas.bind("<Button-5>", self._on_mousewheel)  # Linux scroll down

        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        self.results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

        # Add headers to scrollable frame
        self._create_result_headers()

        # --- Action Button ---
        self.push_button = ttk.Button(
            self.action_frame,
            text="Push Selected to Server",
            command=self.push_to_server,
            state=tk.DISABLED,
        )
        self.push_button.pack(pady=5)  # Simple packing for now

        # --- Usage and Cost Display Labels ---
        self._setup_usage_cost_labels()

        # --- Start Queue Polling ---
        self.root.after(100, self.process_queue)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling for the results canvas."""
        if event.num == 5 or event.delta < 0:  # Scroll down
            self.results_canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:  # Scroll up
            self.results_canvas.yview_scroll(-1, "units")

    def _create_result_headers(self):
        """Creates the header row for the results display."""
        header_frame = ttk.Frame(
            self.scrollable_frame, padding="5 0"
        )  # Padding top/bottom only
        header_frame.grid(
            row=0, column=0, sticky="ew", columnspan=5
        )  # Span across all columns

        col_weights = {
            0: 0,
            1: 2,
            2: 1,
            3: 1,
            4: 0,
        }  # Adjust weights: Chapter title gets more space
        padx_val = 5

        # Apply check column (fixed width)
        ttk.Label(header_frame, text="Apply", anchor="center").grid(
            row=0, column=0, padx=padx_val
        )
        header_frame.columnconfigure(0, weight=col_weights[0])

        # Chapter title column (more weight)
        ttk.Label(header_frame, text="Chapter", anchor="w").grid(
            row=0, column=1, padx=padx_val, sticky="w"
        )
        header_frame.columnconfigure(1, weight=col_weights[1])

        # Original time column
        ttk.Label(header_frame, text="Original Time", anchor="w").grid(
            row=0, column=2, padx=padx_val, sticky="w"
        )
        header_frame.columnconfigure(2, weight=col_weights[2])

        # Refined time column
        ttk.Label(header_frame, text="Refined Time", anchor="w").grid(
            row=0, column=3, padx=padx_val, sticky="w"
        )
        header_frame.columnconfigure(3, weight=col_weights[3])

        # Play button column (fixed width)
        ttk.Label(header_frame, text="Play Audio", anchor="center").grid(
            row=0, column=4, padx=padx_val
        )
        header_frame.columnconfigure(4, weight=col_weights[4])

        # Separator below headers
        ttk.Separator(self.scrollable_frame, orient="horizontal").grid(
            row=1, column=0, sticky="ew", pady=2, columnspan=5
        )

    def _setup_usage_cost_labels(self):
        """Creates and grids the labels for usage and cost display."""
        frame = self.usage_cost_frame
        # StringVars for dynamic updates
        self.ref_prompt_tokens_var = tk.StringVar(value="N/A")
        self.ref_completion_tokens_var = tk.StringVar(value="N/A")
        self.ref_total_tokens_var = tk.StringVar(value="N/A")
        self.ref_cost_var = tk.StringVar(value="N/A")
        self.trans_duration_var = tk.StringVar(value="N/A")
        self.trans_cost_var = tk.StringVar(value="N/A")

        row = 0
        ttk.Label(
            frame, text="Usage & Cost Estimates:", font=("Helvetica", 10, "bold")
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 5))
        row += 1

        # Refinement LLM Usage
        ttk.Label(frame, text="Refinement (LLM)").grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(5, 0)
        )
        row += 1
        ttk.Label(frame, text="Prompt Tokens:").grid(row=row, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.ref_prompt_tokens_var).grid(
            row=row, column=1, sticky="w"
        )
        ttk.Label(frame, text="Completion Tokens:").grid(row=row, column=2, sticky="w")
        ttk.Label(frame, textvariable=self.ref_completion_tokens_var).grid(
            row=row, column=3, sticky="w"
        )
        row += 1
        ttk.Label(frame, text="Total Tokens:").grid(row=row, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.ref_total_tokens_var).grid(
            row=row, column=1, sticky="w"
        )
        ttk.Label(frame, text="Estimated Cost:").grid(row=row, column=2, sticky="w")
        ttk.Label(frame, textvariable=self.ref_cost_var).grid(
            row=row, column=3, sticky="w"
        )
        row += 1

        # Transcription Usage
        ttk.Label(frame, text="Transcription (Audio)").grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )
        row += 1
        ttk.Label(frame, text="Total Duration:").grid(row=row, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.trans_duration_var).grid(
            row=row, column=1, sticky="w"
        )
        ttk.Label(frame, text="Estimated Cost:").grid(row=row, column=2, sticky="w")
        ttk.Label(frame, textvariable=self.trans_cost_var).grid(
            row=row, column=3, sticky="w"
        )

        for i in range(4):
            frame.columnconfigure(i, weight=1)

    def _reset_usage_cost_labels(self):
        """Resets usage and cost labels to their initial state."""
        self.ref_prompt_tokens_var.set("N/A")
        self.ref_completion_tokens_var.set("N/A")
        self.ref_total_tokens_var.set("N/A")
        self.ref_cost_var.set("N/A")
        self.trans_duration_var.set("N/A")
        self.trans_cost_var.set("N/A")

    def _update_usage_cost_display(self, refinement_usage, transcription_usage):
        """Updates the StringVars for usage and cost display."""
        if refinement_usage:
            self.ref_prompt_tokens_var.set(
                str(refinement_usage.get("prompt_tokens", "N/A"))
            )
            self.ref_completion_tokens_var.set(
                str(refinement_usage.get("completion_tokens", "N/A"))
            )
            self.ref_total_tokens_var.set(
                str(refinement_usage.get("total_tokens", "N/A"))
            )
            if (
                self.llm_prompt_cost_config_val > 0
                or self.llm_completion_cost_config_val > 0
            ):
                self.ref_cost_var.set(
                    f"${refinement_usage.get('estimated_cost', 0.0):.4f}"
                )
            else:
                self.ref_cost_var.set("Not Configured")
        else:
            self.ref_prompt_tokens_var.set("N/A")
            self.ref_completion_tokens_var.set("N/A")
            self.ref_total_tokens_var.set("N/A")
            self.ref_cost_var.set("N/A")

        if transcription_usage:
            duration_s = transcription_usage.get("duration_seconds", 0.0)
            duration_m = transcription_usage.get("duration_minutes", 0.0)
            self.trans_duration_var.set(f"{duration_s:.2f}s ({duration_m:.2f}m)")
            if self.transcription_cost_config_val > 0:
                self.trans_cost_var.set(
                    f"${transcription_usage.get('estimated_cost', 0.0):.4f}"
                )
            else:
                self.trans_cost_var.set("Not Configured")
        else:
            self.trans_duration_var.set("N/A")
            self.trans_cost_var.set("N/A")

    def start_processing_thread(self):
        """Initiates the background processing task."""
        book_url = self.book_url_var.get().strip()
        if not book_url:
            messagebox.showerror("Error", "Book URL is required.")
            return

        # Extract server URL and item ID (uses self.abs_host for base if needed)
        server_url_from_gui, item_id_from_gui = self._extract_info_from_url(book_url)

        # Determine server URL to use: GUI entry > config.
        # Item ID must come from GUI.
        if not item_id_from_gui:
            messagebox.showerror(
                "Invalid Input",
                "Could not determine Book ID from URL. Ensure it's a valid Audiobookshelf item URL or just the ID.",
            )
            return

        self.current_item_id = item_id_from_gui  # Store for later use (e.g. push)

        # Use host from config as the definitive one for the client
        # server_url_to_use = server_url_from_gui or self.abs_host # Prioritize URL input
        # The client will use self.abs_host from the config. No need to determine here.

        try:
            window_size = int(self.window_var.get())
            if window_size <= 0:
                raise ValueError("Window size must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Window size must be a positive integer.")
            return

        # ... (validation for model_name) ...
        model_name = self.model_var.get().strip()
        if not model_name:  # Fallback to config if GUI field is empty
            model_name = self.default_model
            self.model_var.set(model_name)

        # Disable button, clear previous results, reset progress
        self.cancel_event.clear()  # Ensure cancel flag is reset before starting
        self.process_button.config(state=tk.DISABLED)
        self.push_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)  # Enable Cancel button
        self.result_data = []
        self.progress_bar["value"] = 0
        self.progress_label_var.set("Progress: Starting...")
        self._reset_usage_cost_labels()  # Reset usage/cost display

        # Gather args for the background task
        args = {
            # Pass the whole config object
            "config": self.config,
            "item_id": self.current_item_id,  # Use the stored item_id
            # "server_url" no longer needed here, client uses config
            "window_size": window_size,  # This is search_window_seconds
            "model_name": model_name,  # Can be overridden by GUI
            "dry_run": self.dry_run_var.get(),
            "temp_dir": str(self.download_path),  # Pass download_path from config
            # Credentials (llm_api_url, llm_api_key, abs_*) are now in self.config
        }

        # Run the core logic in a separate thread
        self.processing_thread = threading.Thread(
            target=self.run_refinement_task, args=(args,), daemon=True
        )
        self.processing_thread.start()

    def run_refinement_task(self, args):
        """The actual processing logic run in the background thread."""
        # Make logger accessible within thread if needed, or pass messages via queue
        try:
            config = args["config"]  # Get the full config object
            item_id = args["item_id"]
            search_window = args[
                "window_size"
            ]  # Renamed from "window_size" for clarity
            model_name = args["model_name"]
            dry_run = args["dry_run"]
            temp_dir = Path(args["temp_dir"])  # Ensure it's a Path object

            self.update_gui_progress(
                0, f"Connecting to {config['audiobookshelf']['host']}..."
            )
            if self.cancel_event.is_set():
                raise InterruptedError("Task cancelled")

            # Initialize components using the config object
            # Client, Transcriber, Refiner, Tool need to be updated to accept config
            client = AudiobookshelfClient(config=config)
            # Login is handled by client's __init__ or a separate method if it needs config
            # client.login() # Assuming login is part of client init or called if needed

            # Ensure temp_dir exists (already created in __init__, but good to have here if logic changes)
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Transcriber and Refiner might be initialized within ChapterRefinementTool
            # If so, ChapterRefinementTool's __init__ needs to handle the 'config'
            tool = ChapterRefinementTool(
                config=config,  # Pass the full config
                client=client,
                # Transcriber and Refiner will be created by the tool using config
                progress_callback=self.update_gui_progress,
                # cancel_event=self.cancel_event # Pass if tool supports it
            )
            # ... rest of the method, replacing direct arg use with config access where appropriate ...
            # For example, refiner's model name is now passed to tool, which passes to refiner.
            # The tool's process_item should use search_window and model_name from its arguments,
            # which are ultimately sourced from GUI / config.

            # The call to tool.process_item should reflect this,
            # it might take fewer direct arguments if they are derived from config inside the tool
            full_results = tool.process_item(
                item_id=item_id,
                search_window_seconds=search_window,  # Explicitly pass from GUI/args
                model_name_override=model_name,  # Explicitly pass from GUI/args
                dry_run=dry_run,  # Add the missing dry_run argument
            )

            refined_chapters_details = full_results.get("chapter_details", [])
            total_chapters = len(refined_chapters_details)

            if not refined_chapters_details:
                msg = "Processing complete, but no chapter details were returned by the tool."
                self.update_gui_progress(100, msg)
                self.queue_task(messagebox.showinfo, "Processing Info", msg)
                self.queue_task(self._update_button_states)  # Re-enable process button
                return

            has_refinements = False
            # Check cancellation within the loop for displaying results (quick check)
            for i, chapter_info in enumerate(refined_chapters_details):
                if self.cancel_event.is_set():
                    raise InterruptedError("Task cancelled")

                refined_time = chapter_info.get("refined_start")
                default_apply = (
                    refined_time is not None
                    and refined_time != chapter_info.get("original_start")
                )
                apply_var = tk.BooleanVar(value=default_apply)

                if default_apply:
                    has_refinements = True

                # Pass the full chapter_info dict which now includes chunk_path and window_start
                result_item = {
                    "original_time": chapter_info.get("original_start", 0),
                    "refined_time": refined_time,
                    "apply_var": apply_var,
                    "chapter_data": chapter_info,
                }
                self.result_data.append(result_item)
                self.queue_task(self._add_result_row, result_item, i)

            # Final progress update and enable push button if changes exist
            final_message = "Processing Complete."
            if not has_refinements:
                final_message += " No significant changes detected."

            self.update_gui_progress(100, final_message)
            self.queue_task(self._update_button_states)

            # After processing, update the usage/cost display via queue
            ref_usage = full_results.get("refinement_usage")
            trans_usage = full_results.get("transcription_usage")
            self.queue_task(self._update_usage_cost_display, ref_usage, trans_usage)

        except InterruptedError:
            self.logger.info("Refinement task execution cancelled.")
            self.update_gui_progress(0, "Cancelled")
            self.queue_task(self._update_button_states)
        except Exception as e:
            error_message = f"An error occurred during processing: {e}"
            self.logger.exception(
                error_message
            )  # Log stack trace for unexpected errors
            self.queue_task(messagebox.showerror, "Processing Error", error_message)
            self.update_gui_progress(0, f"Error: {e}")
            self.queue_task(self._update_button_states)

    def _extract_info_from_url(self, url):
        # Use ABS host from config as fallback or for validation
        abs_host_config = (
            self.config.get("audiobookshelf", {}).get("host", "").strip("/")
        )
        final_server_url = None
        item_id = None

        if not url:
            return None, None  # No URL provided

        # Use the utility to extract item_id
        item_id = extract_item_id_from_url(url)

        if item_id:
            self.logger.info(
                f"Extracted item ID '{item_id}' using utility from URL '{url}'."
            )
            # Attempt to determine server_url if item_id was found in a path-like structure
            # This is a simplified approach; the original regex was more complex for server extraction.
            # If the url contains '/item/', assume the part before it could be the server.
            if "/item/" in url:
                # Try to parse the server part from the URL
                server_match = re.match(r"(https?://[^/]+)", url)
                if server_match:
                    final_server_url = server_match.group(1).strip("/")
                    self.logger.info(
                        f"Deduced server URL from input: {final_server_url}"
                    )

            # If server URL couldn't be deduced from input, or if input was just an ID,
            # fall back to config host.
            if not final_server_url:
                if abs_host_config:
                    final_server_url = abs_host_config
                    self.logger.info(
                        f"Using server URL from config: {final_server_url} for item ID '{item_id}'."
                    )
                else:
                    messagebox.showerror(
                        "Configuration Error",
                        "Item ID found, but server URL could not be determined from input and is missing in config.toml.",
                    )
                    return None, None  # Cannot proceed without server URL

            # Validate the final_server_url
            if not final_server_url.startswith(("http://", "https://")):
                messagebox.showerror(
                    "Configuration Error",
                    f"Determined server URL '{final_server_url}' must include http:// or https://. Please check input or config.toml.",
                )
                return None, None

            return final_server_url, item_id
        else:
            # item_id was not found by the utility
            messagebox.showerror(
                "Invalid Input",
                "Could not parse Item ID from URL using utility. Please use the full item URL (e.g., http://host/item/item-id) or just the item ID.",
            )
            return None, None

    def update_gui_progress(self, value, text):
        """Schedules a progress update in the main thread."""
        # Clamp value between 0 and 100
        value = max(0, min(100, int(value)))
        self.queue_task(self._set_progress, value, text)

    def queue_task(self, task, *args):
        """Adds a task (function and args) to the queue for the main thread."""
        self.task_queue.put((task, args))

    def process_queue(self):
        """Processes tasks from the queue in the main thread."""
        try:
            while True:
                task, args = self.task_queue.get_nowait()
                task(*args)  # Execute the task (e.g., update GUI widget)
                # Don't call update_idletasks here, it can cause performance issues if queue fills up
        except queue.Empty:
            pass  # No tasks left
        finally:
            # Reschedule polling
            self.root.after(100, self.process_queue)

    def _set_progress(self, value, text):
        """Updates the progress bar and label (called by process_queue)."""
        self.progress_bar["value"] = value
        self.progress_label_var.set(f"Progress: {text}")
        self.root.update_idletasks()  # Update UI specifically after progress changes

    def _clear_results_display(self):
        """Removes previous results from the scrollable frame, keeping headers."""
        # Clear internal data first
        self.result_data = []
        # Destroy widgets, skipping headers (row 0) and separator (row 1)
        for widget in self.scrollable_frame.winfo_children():
            grid_info = widget.grid_info()
            if (
                grid_info and grid_info["row"] > 1
            ):  # Check if grid_info exists and row > 1
                widget.destroy()
        # Reset scroll region after clearing (important!)
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _add_result_row(self, result_item, row_index):
        """Adds a single chapter result row to the GUI (called by process_queue)."""
        frame = self.scrollable_frame  # Add row to the inner frame
        # Calculate actual grid row: headers are 0, separator is 1, so data starts at 2
        start_row = row_index + 2

        # result_item now contains the full details including chunk_path and window_start
        apply_var = result_item["apply_var"]
        refined_time = result_item["refined_time"]
        original_time = result_item["original_time"]
        chapter_data = result_item[
            "chapter_data"
        ]  # This holds id, title, chunk_path, window_start etc.
        chunk_path = chapter_data.get("chunk_path")
        window_start = chapter_data.get("window_start_time")

        # --- Create Widgets ---
        apply_check = ttk.Checkbutton(frame, variable=apply_var)

        title = chapter_data.get(
            "title", f"Chapter {row_index + 1}"
        )  # User-friendly 1-based index
        title_label = ttk.Label(
            frame, text=title, anchor="w", wraplength=200
        )  # Wrap long titles

        orig_ts_str = format_timestamp(original_time)
        orig_label = ttk.Label(frame, text=orig_ts_str, anchor="w")

        refined_ts_str = (
            format_timestamp(refined_time) if refined_time is not None else "---"
        )
        refined_label = ttk.Label(frame, text=refined_ts_str, anchor="w")

        # Determine the absolute time to play (refined if available, else original)
        absolute_play_time = refined_time if refined_time is not None else original_time

        play_button = ttk.Button(
            frame,
            text="▶",
            width=3,
            # Pass chunk_path, window_start, and the absolute time to the playback function
            command=lambda cp=chunk_path,
            ws=window_start,
            apt=absolute_play_time: self.play_audio_segment(cp, ws, apt),
        )

        # --- Grid Widgets ---
        padx_val = 5
        apply_check.grid(row=start_row, column=0, padx=padx_val, sticky="ew")
        title_label.grid(row=start_row, column=1, padx=padx_val, sticky="w")
        orig_label.grid(row=start_row, column=2, padx=padx_val, sticky="w")
        refined_label.grid(row=start_row, column=3, padx=padx_val, sticky="w")
        play_button.grid(row=start_row, column=4, padx=padx_val, sticky="ew")

        # --- Configure State ---
        can_apply = refined_time is not None and refined_time != original_time
        if not can_apply:
            apply_check.config(state=tk.DISABLED)
            apply_var.set(False)  # Ensure it's unchecked if disabled

        # Disable play button if chunk_path or window_start is missing (shouldn't happen for ch > 0)
        if not chunk_path or window_start is None:
            play_button.config(state=tk.DISABLED)
            # Log warning only for chapters after the first (index > 0)
            if row_index > 0:  # Use row_index which is available here
                self.logger.warning(
                    f"Missing chunk_path or window_start for chapter {title}. Playback disabled."
                )

    def _update_button_states(self, task_finished=True):
        """Re-enables process button and enables/disables push/cancel buttons based on state."""

        if task_finished:
            # Task completed, failed, or cancelled - reset buttons for idle state
            self.process_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            self.cancel_event.clear()  # Clear event on task completion/failure/cancel

            # Determine if Push button should be enabled based on current results:
            can_push = False
            if not self.dry_run_var.get() and self.result_data:
                for item in self.result_data:
                    if (
                        item["refined_time"] is not None
                        and item["refined_time"] != item["original_time"]
                    ):
                        can_push = True
                        break
            if can_push:
                self.push_button.config(state=tk.NORMAL)
            else:
                self.push_button.config(state=tk.DISABLED)
        else:
            # Task is starting - disable process/push, enable cancel
            self.process_button.config(state=tk.DISABLED)
            self.push_button.config(state=tk.DISABLED)
            self.cancel_button.config(state=tk.NORMAL)

    def play_audio_segment(self, chunk_path, window_start, absolute_start_time):
        """Plays a short audio segment from the downloaded file starting at start_time."""
        if not chunk_path:
            messagebox.showwarning(
                "Playback Failed", "Audio file path is not available."
            )
            return
        if not os.path.exists(chunk_path):
            messagebox.showerror(
                "Playback Error", f"Audio file not found: {chunk_path}"
            )
            # Potentially disable relevant buttons here if needed
            return

        if not has_simpleaudio:
            messagebox.showerror(
                "Playback Error",
                "`simpleaudio` library is required for playback but not found.",
            )
            return

        try:
            # --- Play using simpleaudio from the specific chunk path --- #
            self.logger.debug(
                f"Attempting to play WAV segment from chunk: {chunk_path}"
            )
            if not os.path.exists(chunk_path):
                messagebox.showerror(
                    "Playback Error", f"Audio chunk file not found: {chunk_path}"
                )
                return

            # Calculate time relative to the chunk start
            relative_start_time = absolute_start_time - window_start
            if relative_start_time < 0:
                # This might happen if refined time is slightly before window start
                self.logger.warning(
                    f"Calculated relative start time ({relative_start_time:.3f}s) is negative. Playing from start of chunk."
                )
                relative_start_time = 0.0

            wave_obj = sa.WaveObject.from_wave_file(chunk_path)
            num_channels = wave_obj.num_channels
            bytes_per_sample = wave_obj.bytes_per_sample
            sample_rate = wave_obj.sample_rate

            # Use wave module to read only the segment needed
            with wave.open(chunk_path, "rb") as wf:
                chunk_duration = wf.getnframes() / float(sample_rate)
                seek_frame = int(relative_start_time * sample_rate)

                # Clamp seek frame to chunk boundaries
                seek_frame = max(0, min(seek_frame, wf.getnframes()))

                if seek_frame >= wf.getnframes():
                    messagebox.showwarning(
                        "Playback",
                        "Start time is at or beyond the end of this audio chunk.",
                    )
                    return

                wf.setpos(seek_frame)
                # Calculate frames to read for the desired duration
                duration = 5  # Play for ~5 seconds
                frames_to_read = int(duration * sample_rate)
                remaining_frames = wf.getnframes() - seek_frame
                frames_to_read = min(frames_to_read, remaining_frames)

                if frames_to_read <= 0:
                    messagebox.showwarning(
                        "Playback",
                        "No audio data to play at the specified time in this chunk.",
                    )
                    return

                audio_data_segment = wf.readframes(frames_to_read)

            if audio_data_segment:
                # Stop any currently playing audio first
                sa.stop_all()
                self.logger.info(
                    f"Playing {duration}s segment from chunk '{os.path.basename(chunk_path)}' at {absolute_start_time:.2f}s (relative: {relative_start_time:.2f}s)"
                )
                play_obj = sa.play_buffer(
                    audio_data_segment, num_channels, bytes_per_sample, sample_rate
                )
            else:
                messagebox.showwarning(
                    "Playback", "Calculated audio segment is empty or read failed."
                )

        except wave.Error as e:
            messagebox.showerror(
                "Playback Error", f"Error reading WAV chunk file ({chunk_path}): {e}"
            )
        except FileNotFoundError:
            # This might happen if the original file disappears between checks
            messagebox.showerror(
                "Playback Error",
                f"Audio chunk file not found during playback attempt: {chunk_path}",
            )
        except Exception as e:
            # Catch-all for other unexpected simpleaudio/playback errors
            self.logger.exception(f"Unexpected playback error: {e}")  # Log stack trace
            messagebox.showerror(
                "Playback Error", f"An unexpected error occurred during playback: {e}"
            )

    def push_to_server(self):
        """Collects selected chapters and pushes updates to the ABS server."""
        if self.dry_run_var.get():
            # This check should ideally prevent the button from being enabled,
            # but double-check here for safety.
            messagebox.showwarning(
                "Dry Run", "Dry Run is enabled. Cannot push changes."
            )
            return

        selected_chapters_to_update = []
        for item in self.result_data:
            # Check if the apply box is checked AND if it's a valid change
            if (
                item["apply_var"].get()
                and item["refined_time"] is not None
                and item["refined_time"] != item["original_time"]
            ):
                chapter_id = item["chapter_data"].get("id")
                new_start_time = item["refined_time"]
                if chapter_id:
                    # Ensure start time is non-negative
                    selected_chapters_to_update.append(
                        {"id": chapter_id, "start": max(0.0, new_start_time)}
                    )

        if not selected_chapters_to_update:
            messagebox.showinfo(
                "No Selection", "No applicable chapters selected to push."
            )
            return

        # Confirmation dialog
        confirm_message = (
            f"Push {len(selected_chapters_to_update)} chapter updates to the server?\n"
            "This action cannot be undone easily."
        )
        confirm = messagebox.askyesno("Confirm Update", confirm_message)
        if not confirm:
            return

        # Disable buttons during push
        self.push_button.config(state=tk.DISABLED)
        self.process_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)  # Enable cancel for push operation
        self.cancel_event.clear()  # Reset cancel flag for this operation
        self.progress_label_var.set("Progress: Pushing to server...")
        self.progress_bar["value"] = 0

        # item_id is now self.current_item_id stored during processing
        if not self.current_item_id:
            messagebox.showerror(
                "Error",
                "Cannot push: Item ID not available. Please process a book first.",
            )
            self._update_button_states()
            return

        # Client uses config, so no need for server_url, username, password here in args
        update_args = {
            "config": self.config,  # Pass the main config object
            "item_id": self.current_item_id,
            "chapters": selected_chapters_to_update,
        }
        # Run the update in a thread to prevent GUI freeze
        self.update_thread = threading.Thread(
            target=self.run_server_update_task, args=(update_args,), daemon=True
        )
        self.update_thread.start()

    def run_server_update_task(self, args):
        """Background task to perform the actual server update."""
        try:
            config = args["config"]
            item_id = args["item_id"]
            chapters_to_push = args["chapters"]  # Renamed for clarity

            if self.cancel_event.is_set():
                raise InterruptedError("Task cancelled")

            # Client initialized with config will handle auth
            client = AudiobookshelfClient(config=config)

            self.update_gui_progress(
                50, f"Sending {len(chapters_to_push)} updates to server..."
            )
            if self.cancel_event.is_set():
                raise InterruptedError("Task cancelled")

            # Client method needs to accept item_id and list of chapter dicts
            # Example: client.update_chapters_start_times(item_id, chapters_to_push)
            success = client.update_chapters_start_time(
                item_id, chapters_to_push
            )  # Assuming this method exists and uses config for auth

            if self.cancel_event.is_set():
                raise InterruptedError("Task cancelled")

            if success:
                self.update_gui_progress(100, "Server update successful.")
                self.queue_task(
                    messagebox.showinfo,
                    "Success",
                    "Selected chapter times updated on the server.",
                )
            else:
                raise Exception(
                    "Server update method returned failure (check client logs/API response or client's internal error handling)."
                )

        except InterruptedError:
            self.logger.info("Server update task execution cancelled.")
            self.update_gui_progress(0, "Cancelled Update")
            # Leave buttons disabled until user starts new process
            self.queue_task(self._update_button_states, True)
        except AttributeError as e:
            # Added specific handling for missing client method
            self.logger.error(f"Client method missing: {e}")
            error_msg = f"Client is missing a required method: {e}. Please check client implementation."
            self.update_gui_progress(0, "Error: Client method missing")
            self.queue_task(messagebox.showerror, "Update Failed", error_msg)
            self.queue_task(self._update_button_states, True)
        except Exception as e:
            self.logger.exception(f"Error updating server: {e}")  # Log stack trace
            self.update_gui_progress(0, f"Error updating server: {e}")
            self.queue_task(
                messagebox.showerror, "Update Failed", f"Failed to update server: {e}"
            )
            self.queue_task(self._update_button_states, True)
        finally:
            # Ensure buttons are reset correctly after task finishes or is cancelled
            # The calls within except blocks already handle this for specific cases.
            # If an unhandled exception occurred before those, or if normal completion,
            # we might still need a general call here, but it's largely covered.
            # For now, relying on except blocks. A general call here might be redundant
            # or could interfere if specific state (like cancelled) is desired.
            pass

    def cancel_task(self):
        """Signals the background task to cancel."""
        self.logger.info("Cancel requested by user.")
        self.cancel_event.set()  # Set the event flag
        self.cancel_button.config(
            state=tk.DISABLED
        )  # Disable cancel button immediately
        self.progress_label_var.set("Progress: Cancelling...")  # Update status


# --- Main Execution ---
if __name__ == "__main__":
    # --- Load config to get logging level ---
    # Initial minimal config load just for logging, before full app init
    # This avoids logging before config is parsed if app itself logs in __init__
    # App's __init__ will then load the full config.
    log_level_from_config = "INFO"  # Default
    temp_config_for_logging = {}
    try:
        # Try to load config just for the logging level
        # config.py will search for config.toml in default locations
        temp_config_for_logging = get_config()
        log_level_from_config = (
            temp_config_for_logging.get("logging", {}).get("level", "INFO").upper()
        )
    except ConfigError as e:
        # Log to console if config fails early, as logger might not be set up
        print(f"Config warning (for logging init): {e}. Using default INFO level.")
    except Exception as e:  # Catch any other error during this pre-load
        print(
            f"Unexpected error during logging config pre-load: {e}. Using default INFO level."
        )

    logging.basicConfig(
        level=log_level_from_config,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Set up main window ---
    main_root = tk.Tk()
    app = AbsRefinedApp(main_root)  # App __init__ handles full config load & validation

    if not main_root.winfo_exists():
        # This means app.root.destroy() was called in __init__ (e.g. due to config error)
        logging.error(
            "Application initialization failed, likely due to configuration issues. Exiting."
        )
        # messagebox might have already been shown by __init__
        exit(1)

    logging.info(f"Starting ABSRefined GUI with log level {log_level_from_config}.")
    main_root.mainloop()
    logging.info("ABSRefined GUI closed.")
