import tkinter as tk
from tkinter import ttk  # For themed widgets like Progressbar
from tkinter import messagebox, filedialog # For messages and file dialogs (temp dir)
import threading
import queue
import os
import re # Added for URL parsing
import logging # Add logging import
from getpass import getpass # Added for password input fallback
from dotenv import load_dotenv
import tempfile # Add tempfile for temporary WAV conversion
import atexit # To help clean up temporary files
import shutil # For directory cleanup

# --- Import your existing absrefined components ---
# (Adjust paths if gui.py is not in the root)
from absrefined.client import AudiobookshelfClient
from absrefined.transcriber import AudioTranscriber
from absrefined.refiner import ChapterRefiner
from absrefined.refinement_tool import ChapterRefinementTool
from absrefined.utils.timestamp import format_timestamp # Assuming you have this

# --- Try importing audio libraries and set flags --- #
has_simpleaudio = False
# has_pydub = False # Removed
try:
    import simpleaudio as sa
    import wave # simpleaudio uses this
    has_simpleaudio = True
except ImportError:
    print("Warning: `simpleaudio` or `wave` library not found. Audio playback might be limited.")
    print("Please install it using: pip install simpleaudio")

# --- Keep track of temporary files to delete on exit --- #
# No longer tracking individual files, cleaning directory instead
# _temp_files_to_delete = set()

def _cleanup_temp_files():
    # Use logger if available, otherwise print
    logger = logging.getLogger("_cleanup_temp_files") if 'logging' in globals() else None
    temp_dir_to_clean = "temp_gui" # The directory used by the GUI

    message = f"Cleaning up temporary directory: {temp_dir_to_clean}"
    if logger:
        logger.info(message)
    else:
        print(message)

    if os.path.isdir(temp_dir_to_clean):
        try:
            shutil.rmtree(temp_dir_to_clean)
            if logger:
                logger.debug(f"  Successfully removed directory: {temp_dir_to_clean}")
            else:
                print(f"  Successfully removed directory: {temp_dir_to_clean}")
        except OSError as e:
            error_message = f"  Error removing directory {temp_dir_to_clean}: {e}"
            if logger:
                 logger.warning(error_message)
            else:
                 print(error_message)
    else:
         message = f"  Temporary directory not found: {temp_dir_to_clean}"
         if logger:
              logger.info(message)
         else:
              print(message)

atexit.register(_cleanup_temp_files)

# --- Constants (Consider moving to a config file later) ---
DEFAULT_WINDOW = 15
DEFAULT_MODEL = "gpt-4o-mini" # Or fetch from env

class AbsRefinedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ABSRefined GUI")
        self.root.geometry("800x600") # Adjust as needed

        self.task_queue = queue.Queue()
        self.result_data = [] # To store chapter results {orig_time, refined_time, apply_var, chapter_data}
        self.audio_file_path = None # To store the path of the downloaded audio file

        # --- Load Environment Variables ---
        load_dotenv()
        # You might want more robust handling for missing env vars in a GUI
        self.llm_api_url = os.getenv("OPENAI_API_URL")
        self.llm_api_key = os.getenv("OPENAI_API_KEY")
        self.abs_username = os.getenv("ABS_USERNAME")
        self.abs_password = os.getenv("ABS_PASSWORD")
        self.default_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

        # --- Initialize Logger for the App --- #
        self.logger = logging.getLogger(self.__class__.__name__)
        # Note: Basic config should be set outside the class, see main block

        # --- Cancellation Event --- #
        self.cancel_event = threading.Event()

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

        # Configure resizing behavior
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1) # Allow results frame to expand

        # --- Input Fields ---
        ttk.Label(self.input_frame, text="Book URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.book_url_var = tk.StringVar()
        self.book_url_entry = ttk.Entry(self.input_frame, textvariable=self.book_url_var, width=60)
        self.book_url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(self.input_frame, text="Window (s):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.window_var = tk.StringVar(value=str(DEFAULT_WINDOW))
        self.window_entry = ttk.Entry(self.input_frame, textvariable=self.window_var, width=10)
        self.window_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w") # Align left

        ttk.Label(self.input_frame, text="Model:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar(value=self.default_model)
        self.model_entry = ttk.Entry(self.input_frame, textvariable=self.model_var, width=30)
        self.model_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w") # Align left

        # --- ABS Credentials (only enabled if not in .env) --- #
        cred_row_start = 3
        self.abs_user_var = tk.StringVar()
        self.abs_pass_var = tk.StringVar()

        ttk.Label(self.input_frame, text="ABS User:").grid(row=cred_row_start, column=0, padx=5, pady=5, sticky="w")
        self.abs_user_entry = ttk.Entry(self.input_frame, textvariable=self.abs_user_var, width=30)
        self.abs_user_entry.grid(row=cred_row_start, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(self.input_frame, text="ABS Pass:").grid(row=cred_row_start + 1, column=0, padx=5, pady=5, sticky="w")
        self.abs_pass_entry = ttk.Entry(self.input_frame, textvariable=self.abs_pass_var, width=30, show="*")
        self.abs_pass_entry.grid(row=cred_row_start + 1, column=1, padx=5, pady=5, sticky="w")

        # Disable credential fields if loaded from environment
        if self.abs_username:
            self.abs_user_entry.insert(0, "(from .env)")
            self.abs_user_entry.config(state=tk.DISABLED)
        if self.abs_password:
             self.abs_pass_entry.insert(0, "********")
             self.abs_pass_entry.config(state=tk.DISABLED)

        # Configure input frame column resizing
        self.input_frame.columnconfigure(1, weight=1)

        # --- Control Elements ---
        self.dry_run_var = tk.BooleanVar(value=True) # Default to dry run for safety
        self.dry_run_check = ttk.Checkbutton(self.control_frame, text="Dry Run (Don't update server)", variable=self.dry_run_var, command=self._update_button_states)
        self.dry_run_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.process_button = ttk.Button(self.control_frame, text="Start Processing", command=self.start_processing_thread)
        self.process_button.grid(row=0, column=1, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.control_frame, text="Cancel", command=self.cancel_task, state=tk.DISABLED)
        self.cancel_button.grid(row=0, column=2, padx=5, pady=5)

        self.control_frame.columnconfigure(1, weight=1) # Push button to the right
        # Column 2 (Cancel button) should not expand

        # --- Progress Bar ---
        self.progress_label_var = tk.StringVar(value="Progress: Idle")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_label_var)
        self.progress_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate", length=400)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        self.progress_frame.columnconfigure(0, weight=1)

        # --- Results Area (Placeholder - will be populated dynamically) ---
        # Using a Canvas + Frame for scrollability
        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.results_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )

        self.results_canvas_window = self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw") # Store window id
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)

        # Mouse wheel scrolling for the results canvas
        self.results_canvas.bind_all("<MouseWheel>", self._on_mousewheel) # Bind to all for simplicity on macOS/Windows
        self.results_canvas.bind("<Button-4>", self._on_mousewheel) # Linux scroll up
        self.results_canvas.bind("<Button-5>", self._on_mousewheel) # Linux scroll down


        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        self.results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

        # Add headers to scrollable frame
        self._create_result_headers()

        # --- Action Button ---
        self.push_button = ttk.Button(self.action_frame, text="Push Selected to Server", command=self.push_to_server, state=tk.DISABLED)
        self.push_button.pack(pady=5) # Simple packing for now

        # --- Start Queue Polling ---
        self.root.after(100, self.process_queue)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling for the results canvas."""
        if event.num == 5 or event.delta < 0: # Scroll down
            self.results_canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0: # Scroll up
            self.results_canvas.yview_scroll(-1, "units")

    def _create_result_headers(self):
        """Creates the header row for the results display."""
        header_frame = ttk.Frame(self.scrollable_frame, padding="5 0") # Padding top/bottom only
        header_frame.grid(row=0, column=0, sticky="ew", columnspan=5) # Span across all columns

        col_weights = {0: 0, 1: 2, 2: 1, 3: 1, 4: 0} # Adjust weights: Chapter title gets more space
        padx_val = 5

        # Apply check column (fixed width)
        ttk.Label(header_frame, text="Apply", anchor="center").grid(row=0, column=0, padx=padx_val)
        header_frame.columnconfigure(0, weight=col_weights[0])

        # Chapter title column (more weight)
        ttk.Label(header_frame, text="Chapter", anchor="w").grid(row=0, column=1, padx=padx_val, sticky="w")
        header_frame.columnconfigure(1, weight=col_weights[1])

        # Original time column
        ttk.Label(header_frame, text="Original Time", anchor="w").grid(row=0, column=2, padx=padx_val, sticky="w")
        header_frame.columnconfigure(2, weight=col_weights[2])

        # Refined time column
        ttk.Label(header_frame, text="Refined Time", anchor="w").grid(row=0, column=3, padx=padx_val, sticky="w")
        header_frame.columnconfigure(3, weight=col_weights[3])

        # Play button column (fixed width)
        ttk.Label(header_frame, text="Play Audio", anchor="center").grid(row=0, column=4, padx=padx_val)
        header_frame.columnconfigure(4, weight=col_weights[4])

        # Separator below headers
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(row=1, column=0, sticky='ew', pady=2, columnspan=5)


    def start_processing_thread(self):
        """Initiates the background processing task."""
        book_url = self.book_url_var.get()
        if not book_url:
            messagebox.showerror("Error", "Book URL is required.")
            return

        server_url, item_id = self._extract_info_from_url(book_url)
        if not server_url or not item_id:
             messagebox.showerror("Error", "Could not extract server URL and item ID from the Book URL. Please check the format (e.g., http://server.com/item/item-id).")
             return

        try:
            window_size = int(self.window_var.get())
            if window_size <= 0:
                 raise ValueError("Window size must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Window size must be a positive integer.")
            return

        # Check for essential env vars OR GUI input
        username_to_use = self.abs_username
        password_to_use = self.abs_password

        if not username_to_use:
            username_to_use = self.abs_user_var.get()
            if not username_to_use:
                 messagebox.showerror("Missing Input", "ABS Username is required (or set ABS_USERNAME in .env).")
                 return

        if not password_to_use:
            password_to_use = self.abs_pass_var.get()
            if not password_to_use:
                 messagebox.showerror("Missing Input", "ABS Password is required (or set ABS_PASSWORD in .env).")
                 return

        if not self.llm_api_url or not self.llm_api_key:
             messagebox.showerror("Missing Config", "OpenAI API URL (OPENAI_API_URL) or Key (OPENAI_API_KEY) not found in environment variables (.env).")
             return
        # REMOVED check for self.abs_username/password here, handled above
        # if not self.abs_username or not self.abs_password:
        #      messagebox.showerror(...)
        #      return

        # Disable button, clear previous results, reset progress
        self.cancel_event.clear() # Ensure cancel flag is reset before starting
        self.process_button.config(state=tk.DISABLED)
        self.push_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL) # Enable Cancel button
        self.result_data = []
        self.audio_file_path = None # Reset audio path
        self.progress_bar["value"] = 0
        self.progress_label_var.set("Progress: Starting...")

        # Gather args for the background task
        args = {
            "server_url": server_url, # Pass extracted info
            "item_id": item_id,       # Pass extracted info
            "window_size": window_size,
            "model": self.model_var.get(),
            "dry_run": self.dry_run_var.get(),
            "temp_dir": "temp_gui", # Or let user choose
            # Pass necessary credentials/API info
            "llm_api_url": self.llm_api_url,
            "llm_api_key": self.llm_api_key,
            "abs_username": username_to_use,
            "abs_password": password_to_use,
        }

        # Run the core logic in a separate thread
        self.processing_thread = threading.Thread(target=self.run_refinement_task, args=(args,), daemon=True)
        self.processing_thread.start()

    def run_refinement_task(self, args):
        """The actual processing logic run in the background thread."""
        # Make logger accessible within thread if needed, or pass messages via queue
        try:
            # --- Setup ---
            server_url = args["server_url"]
            item_id = args["item_id"]
            self.update_gui_progress(0, f"Connecting to {server_url}...")
            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

            client = AudiobookshelfClient(server_url, verbose=False) # GUI handles feedback
            try:
                 # Add cancellation check before potentially long operation
                 if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")
                 client.login(args["abs_username"], args["abs_password"])
                 self.update_gui_progress(5, "Login successful.")
            except Exception as login_err:
                 raise ConnectionError(f"Failed to login to Audiobookshelf: {login_err}")

            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

            temp_dir = args["temp_dir"]
            os.makedirs(temp_dir, exist_ok=True)

            refiner = ChapterRefiner(
                args["llm_api_url"],
                args["model"],
                window_size=args["window_size"],
                verbose=False,
                llm_api_key=args["llm_api_key"],
            )

            transcriber = AudioTranscriber(api_key=args["llm_api_key"], verbose=False, debug=False)

            # --- Refinement Tool --- #
            # Pass the cancellation event to the tool if the tool supports it
            # For now, we check the event *between* calls to the tool's methods
            tool = ChapterRefinementTool(
                client,
                transcriber,
                refiner,
                verbose=False,
                temp_dir=temp_dir,
                dry_run=True,
                debug=False,
                progress_callback=self.update_gui_progress # Pass the GUI's progress updater
                # Pass cancel_event if tool is modified: cancel_event=self.cancel_event
            )

            self.update_gui_progress(10, f"Processing item: {item_id}...")
            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

            # --- Call process_item --- #
            # Note: Cancellation *during* process_item requires modifying the tool itself.
            # Here, we can only cancel *before* this potentially long call.
            full_results = tool.process_item(item_id)

            # Check immediately after the main processing call
            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

            # --- Process Results for GUI --- #
            if full_results.get("error"):
                 raise Exception(f"Processing failed: {full_results['error']}")

            # Store audio file path for playback (though playback now uses chunks)
            # self.audio_file_path = full_results.get("audio_file_path")
            # ... (rest of result processing)

            refined_chapters_details = full_results.get("chapter_details", [])
            total_chapters = len(refined_chapters_details)

            if not refined_chapters_details:
                 msg = "Processing complete, but no chapter details were returned by the tool."
                 self.update_gui_progress(100, msg)
                 self.queue_task(messagebox.showinfo, "Processing Info", msg)
                 self.queue_task(self._update_button_states) # Re-enable process button
                 return

            has_refinements = False
            # Check cancellation within the loop for displaying results (quick check)
            for i, chapter_info in enumerate(refined_chapters_details):
                if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

                refined_time = chapter_info.get('refined_start')
                default_apply = refined_time is not None and refined_time != chapter_info.get('original_start')
                apply_var = tk.BooleanVar(value=default_apply)

                if default_apply:
                    has_refinements = True

                # Pass the full chapter_info dict which now includes chunk_path and window_start
                result_item = {
                    "original_time": chapter_info.get('original_start', 0),
                    "refined_time": refined_time,
                    "apply_var": apply_var,
                    "chapter_data": chapter_info
                }
                self.result_data.append(result_item)
                self.queue_task(self._add_result_row, result_item, i)

            # Final progress update and enable push button if changes exist
            final_message = "Processing Complete."
            if not has_refinements:
                final_message += " No significant changes detected."

            self.update_gui_progress(100, final_message)
            self.queue_task(self._update_button_states)

        except InterruptedError:
             self.logger.info("Refinement task execution cancelled.")
             self.update_gui_progress(0, "Cancelled")
             self.queue_task(self._update_button_states)
        except ConnectionError as e:
             self.queue_task(messagebox.showerror, "Connection Error", f"{e}")
             self.update_gui_progress(0, f"Error: {e}")
             self.queue_task(self._update_button_states)
        except Exception as e:
             error_message = f"An error occurred during processing: {e}"
             self.logger.exception(error_message) # Log stack trace for unexpected errors
             self.queue_task(messagebox.showerror, "Processing Error", error_message)
             self.update_gui_progress(0, f"Error: {e}")
             self.queue_task(self._update_button_states)

    def _extract_info_from_url(self, url):
        """Helper to extract server URL and item ID."""
        # Reuse regex from main.py or improve it
        match = re.search(r"(https?://[^/]+)(?:/[^/]+)*/item/([a-zA-Z0-9\-]+)", url)
        if match:
            return match.group(1), match.group(2)
        # Try extracting just the server URL if item ID isn't present (less ideal)
        match_server = re.search(r"(https?://[^/]+)", url)
        if match_server:
            return match_server.group(1), None # Return None for item_id if not found
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
                task(*args) # Execute the task (e.g., update GUI widget)
                # Don't call update_idletasks here, it can cause performance issues if queue fills up
        except queue.Empty:
            pass # No tasks left
        finally:
            # Reschedule polling
            self.root.after(100, self.process_queue)

    def _set_progress(self, value, text):
        """Updates the progress bar and label (called by process_queue)."""
        self.progress_bar["value"] = value
        self.progress_label_var.set(f"Progress: {text}")
        self.root.update_idletasks() # Update UI specifically after progress changes


    def _clear_results_display(self):
         """Removes previous results from the scrollable frame, keeping headers."""
         # Clear internal data first
         self.result_data = []
         # Destroy widgets, skipping headers (row 0) and separator (row 1)
         for widget in self.scrollable_frame.winfo_children():
             grid_info = widget.grid_info()
             if grid_info and grid_info['row'] > 1: # Check if grid_info exists and row > 1
                 widget.destroy()
         # Reset scroll region after clearing (important!)
         self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))


    def _add_result_row(self, result_item, row_index):
        """Adds a single chapter result row to the GUI (called by process_queue)."""
        frame = self.scrollable_frame # Add row to the inner frame
        # Calculate actual grid row: headers are 0, separator is 1, so data starts at 2
        start_row = row_index + 2

        # result_item now contains the full details including chunk_path and window_start
        apply_var = result_item["apply_var"]
        refined_time = result_item["refined_time"]
        original_time = result_item["original_time"]
        chapter_data = result_item["chapter_data"] # This holds id, title, chunk_path, window_start etc.
        chunk_path = chapter_data.get("chunk_path")
        window_start = chapter_data.get("window_start")

        # --- Create Widgets ---
        apply_check = ttk.Checkbutton(frame, variable=apply_var)

        title = chapter_data.get('title', f"Chapter {row_index + 1}") # User-friendly 1-based index
        title_label = ttk.Label(frame, text=title, anchor="w", wraplength=200) # Wrap long titles

        orig_ts_str = format_timestamp(original_time)
        orig_label = ttk.Label(frame, text=orig_ts_str, anchor="w")

        refined_ts_str = format_timestamp(refined_time) if refined_time is not None else "---"
        refined_label = ttk.Label(frame, text=refined_ts_str, anchor="w")

        # Determine the absolute time to play (refined if available, else original)
        absolute_play_time = refined_time if refined_time is not None else original_time

        play_button = ttk.Button(frame, text="▶", width=3,
                                 # Pass chunk_path, window_start, and the absolute time to the playback function
                                 command=lambda cp=chunk_path, ws=window_start, apt=absolute_play_time:
                                          self.play_audio_segment(cp, ws, apt))

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
            apply_var.set(False) # Ensure it's unchecked if disabled

        # Disable play button if chunk_path or window_start is missing (shouldn't happen for ch > 0)
        if not chunk_path or window_start is None:
             play_button.config(state=tk.DISABLED)
             # Log warning only for chapters after the first (index > 0)
             if row_index > 0: # Use row_index which is available here
                  self.logger.warning(f"Missing chunk_path or window_start for chapter {title}. Playback disabled.")


    def _update_button_states(self, task_finished=True):
        """Re-enables process button and enables/disables push/cancel buttons based on state."""

        if task_finished:
            # Task completed, failed, or cancelled - reset buttons for idle state
            self.process_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            self.cancel_event.clear() # Clear event on task completion/failure/cancel

            # Determine if Push button should be enabled based on current results:
            can_push = False
            if not self.dry_run_var.get() and self.result_data:
                 for item in self.result_data:
                      if item["refined_time"] is not None and item["refined_time"] != item["original_time"]:
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
            messagebox.showwarning("Playback Failed", "Audio file path is not available.")
            return
        if not os.path.exists(chunk_path):
            messagebox.showerror("Playback Error", f"Audio file not found: {chunk_path}")
            self.audio_file_path = None # Mark as unavailable
            # Potentially disable relevant buttons here if needed
            return

        if not has_simpleaudio:
             messagebox.showerror("Playback Error", "`simpleaudio` library is required for playback but not found.")
             return

        try:
            # --- Play using simpleaudio from the specific chunk path --- #
            self.logger.debug(f"Attempting to play WAV segment from chunk: {chunk_path}")
            if not os.path.exists(chunk_path):
                 messagebox.showerror("Playback Error", f"Audio chunk file not found: {chunk_path}")
                 return

            # Calculate time relative to the chunk start
            relative_start_time = absolute_start_time - window_start
            if relative_start_time < 0:
                # This might happen if refined time is slightly before window start
                self.logger.warning(f"Calculated relative start time ({relative_start_time:.3f}s) is negative. Playing from start of chunk.")
                relative_start_time = 0.0

            wave_obj = sa.WaveObject.from_wave_file(chunk_path)
            num_channels = wave_obj.num_channels
            bytes_per_sample = wave_obj.bytes_per_sample
            sample_rate = wave_obj.sample_rate

            # Use wave module to read only the segment needed
            with wave.open(chunk_path, 'rb') as wf:
                chunk_duration = wf.getnframes() / float(sample_rate)
                seek_frame = int(relative_start_time * sample_rate)

                # Clamp seek frame to chunk boundaries
                seek_frame = max(0, min(seek_frame, wf.getnframes()))

                if seek_frame >= wf.getnframes():
                    messagebox.showwarning("Playback", "Start time is at or beyond the end of this audio chunk.")
                    return

                wf.setpos(seek_frame)
                # Calculate frames to read for the desired duration
                duration = 5 # Play for ~5 seconds
                frames_to_read = int(duration * sample_rate)
                remaining_frames = wf.getnframes() - seek_frame
                frames_to_read = min(frames_to_read, remaining_frames)

                if frames_to_read <= 0:
                     messagebox.showwarning("Playback", "No audio data to play at the specified time in this chunk.")
                     return

                audio_data_segment = wf.readframes(frames_to_read)

            if audio_data_segment:
                # Stop any currently playing audio first
                sa.stop_all()
                self.logger.info(f"Playing {duration}s segment from chunk '{os.path.basename(chunk_path)}' at {absolute_start_time:.2f}s (relative: {relative_start_time:.2f}s)")
                play_obj = sa.play_buffer(audio_data_segment, num_channels, bytes_per_sample, sample_rate)
            else:
                messagebox.showwarning("Playback", "Calculated audio segment is empty or read failed.")

        except wave.Error as e:
             messagebox.showerror("Playback Error", f"Error reading WAV chunk file ({chunk_path}): {e}")
        except FileNotFoundError:
            # This might happen if the original file disappears between checks
            messagebox.showerror("Playback Error", f"Audio chunk file not found during playback attempt: {chunk_path}")
            self.audio_file_path = None
        except Exception as e:
            # Catch-all for other unexpected simpleaudio/playback errors
            self.logger.exception(f"Unexpected playback error: {e}") # Log stack trace
            messagebox.showerror("Playback Error", f"An unexpected error occurred during playback: {e}")


    def push_to_server(self):
        """Collects selected chapters and pushes updates to the ABS server."""
        if self.dry_run_var.get():
             # This check should ideally prevent the button from being enabled,
             # but double-check here for safety.
             messagebox.showwarning("Dry Run", "Dry Run is enabled. Cannot push changes.")
             return

        selected_chapters_to_update = []
        for item in self.result_data:
            # Check if the apply box is checked AND if it's a valid change
            if item["apply_var"].get() and item["refined_time"] is not None and item["refined_time"] != item["original_time"]:
                chapter_id = item["chapter_data"].get("id")
                new_start_time = item["refined_time"]
                if chapter_id:
                    # Ensure start time is non-negative
                    selected_chapters_to_update.append({"id": chapter_id, "start": max(0.0, new_start_time)})

        if not selected_chapters_to_update:
            messagebox.showinfo("No Selection", "No applicable chapters selected to push.")
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
        self.cancel_button.config(state=tk.NORMAL) # Enable cancel for push operation
        self.cancel_event.clear() # Reset cancel flag for this operation
        self.progress_label_var.set("Progress: Pushing to server...")
        self.progress_bar["value"] = 0

        # Needs server/item ID again
        book_url = self.book_url_var.get()
        server_url, item_id = self._extract_info_from_url(book_url)
        if not server_url or not item_id:
             messagebox.showerror("Error", "Cannot push: Could not determine server/item ID.")
             self._update_button_states() # Re-enable buttons
             return

        # Run the update in a thread to prevent GUI freeze
        update_args = {
            "server_url": server_url,
            "item_id": item_id,
            "chapters": selected_chapters_to_update,
            "abs_username": self.abs_username,
            "abs_password": self.abs_password,
        }
        self.update_thread = threading.Thread(target=self.run_server_update_task, args=(update_args,), daemon=True)
        self.update_thread.start()


    def run_server_update_task(self, args):
        """Background task to perform the actual server update."""
        try:
            # Add cancellation check before potentially long operations
            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

            client = AudiobookshelfClient(args["server_url"], verbose=False)
            # Add cancellation check before potentially long operations
            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")
            client.login(args["abs_username"], args["abs_password"])

            # --- Need client.update_chapters_start_time method ---
            self.update_gui_progress(50, f"Sending {len(args['chapters'])} updates to server...")

            # Add cancellation check before potentially long operations
            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

            # *** This method needs to be added to abs_client.py ***
            # It should ideally take just item_id and the list of {'id': '...', 'start': ...} dicts
            success = client.update_chapters_start_time(args["item_id"], args["chapters"])

            # Check for cancellation immediately after the call returns
            if self.cancel_event.is_set(): raise InterruptedError("Task cancelled")

            if success:
                self.update_gui_progress(100, "Server update successful.")
                self.queue_task(messagebox.showinfo, "Success", "Selected chapter times updated on the server.")
            else:
                raise Exception("Server update method returned failure (check client logs/API response).")

        except InterruptedError:
             self.logger.info("Server update task execution cancelled.")
             self.update_gui_progress(0, "Cancelled Update")
             # Leave buttons disabled until user starts new process
             self.queue_task(self._update_button_states, task_finished=True)
        except AttributeError:
             error_msg = "The `update_chapters_start_time` method is not implemented in AudiobookshelfClient."
             self.update_gui_progress(0, "Error: Client method missing")
             self.queue_task(messagebox.showerror, "Update Failed", error_msg)
             self.queue_task(self._update_button_states, task_finished=True)
        except Exception as e:
            self.logger.exception(f"Error updating server: {e}") # Log stack trace
            self.update_gui_progress(0, f"Error updating server: {e}")
            self.queue_task(messagebox.showerror, "Update Failed", f"Failed to update server: {e}")
            self.queue_task(self._update_button_states, task_finished=True)
        finally:
            # Ensure buttons are reset correctly after task finishes or is cancelled
            # Pass flag to indicate task finished to _update_button_states
            # This is now handled within the except/success blocks to ensure proper state
            # self.queue_task(self._update_button_states, task_finished=True)
            pass

    def cancel_task(self):
        """Signals the background task to cancel."""
        self.logger.info("Cancel requested by user.")
        self.cancel_event.set() # Set the event flag
        self.cancel_button.config(state=tk.DISABLED) # Disable cancel button immediately
        self.progress_label_var.set("Progress: Cancelling...") # Update status


# --- Main Execution ---
if __name__ == "__main__":
    # --- Basic Logging Setup --- #
    logging.basicConfig(level=logging.INFO, # Change to DEBUG for more verbose logs
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # Silence noisy libraries if needed
    # logging.getLogger("pydub").setLevel(logging.WARNING)

    # Add library dependency check here?
    if not has_simpleaudio:
        # Already printed warning, maybe show messagebox?
        pass # For now, rely on print warnings
    # if not has_pydub:
    #     pass # Rely on print warnings

    root = tk.Tk()
    app = AbsRefinedApp(root)
    root.mainloop() 