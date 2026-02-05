#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import os
import threading
from pathlib import Path
import json
import logging
import queue

# Import existing pipeline logic
try:
    # Add script directory to sys.path to ensure imports work if run from project root
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    import dataset_pipeline
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import dataset_pipeline: {e}")
    sys.exit(1)

# Configure Logging to be captured by GUI
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

class DatasetPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Processing Pipeline")
        self.root.geometry("800x750")
        
        # Styles
        self.style = ttk.Style()
        self.style.configure("TButton", padding=5)
        self.style.configure("TLabel", padding=5)

        # Variables
        self.defaults = dataset_pipeline.PipelinePaths.defaults()
        
        self.input_path = tk.StringVar(value=str(self.defaults.input_raw))
        self.clean_path = tk.StringVar(value=str(self.defaults.output_clean))
        self.dedup_path = tk.StringVar(value=str(self.defaults.output_dedup))
        self.train_path = tk.StringVar(value=str(self.defaults.output_train))
        self.bert_path = tk.StringVar(value=str(self.defaults.output_train).replace(".csv", "_bert.csv"))
        
        self.convert_json = tk.BooleanVar(value=False)
        self.validate_json = tk.BooleanVar(value=False)
        self.generate_bert = tk.BooleanVar(value=False)

        # Queue for thread-safe logging
        self.log_queue = queue.Queue()
        self.setup_logging()

        # UI Layout
        self.create_widgets()
        self.root.after(100, self.process_log_queue)

    def setup_logging(self):
        # Setup logger to capture output from dataset_pipeline (assuming it uses print, but we can redirect or capture logging)
        # Note: The original script uses print(). We'll redirect stdout/stderr or modifying script calls to logging not trivial without editing.
        # But wait, the original script mainly prints. We can override print locally or capture stdout.
        pass # We will handle capturing in the run method or by redirecting sys.stdout

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- File Configuration Section ---
        config_frame = ttk.LabelFrame(main_frame, text="File Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=5)

        self.create_path_entry(config_frame, "Input Raw CSV:", self.input_path, 0, True)
        self.create_path_entry(config_frame, "Output Clean CSV:", self.clean_path, 1)
        self.create_path_entry(config_frame, "Output Dedup CSV:", self.dedup_path, 2)
        self.create_path_entry(config_frame, "Output Train CSV:", self.train_path, 3)
        self.create_path_entry(config_frame, "Output BERT CSV:", self.bert_path, 4)

        # --- Options Section ---
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(options_frame, text="Convert Response to JSON (Pipeline)", variable=self.convert_json).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options_frame, text="Validate Response JSON", variable=self.validate_json).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options_frame, text="Generate Extra BERT Dataset (separate file)", variable=self.generate_bert).pack(side=tk.LEFT, padx=10)

        # --- Actions Section ---
        action_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        action_frame.pack(fill=tk.X, pady=5)

        ttk.Button(action_frame, text="1. Clean Only", command=lambda: self.run_task("clean")).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(action_frame, text="2. Dedup Only", command=lambda: self.run_task("dedup")).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(action_frame, text="3. Split Only", command=lambda: self.run_task("split")).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(action_frame, text="4. Validate Only", command=lambda: self.run_task("validate")).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        separator = ttk.Separator(action_frame, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        run_all_btn = ttk.Button(action_frame, text="🚀 Run ALL Pipeline", command=lambda: self.run_task("all"))
        run_all_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- Log Output Section ---
        log_frame = ttk.LabelFrame(main_frame, text="Execution Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_area = scrolledtext.ScrolledText(log_frame, state='disabled', height=15)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def create_path_entry(self, parent, label_text, var, row, is_input=False):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        
        if is_input:
            btn = ttk.Button(parent, text="Browse", command=lambda: self.browse_file(var, True))
        else:
            btn = ttk.Button(parent, text="Browse", command=lambda: self.browse_file(var, False))
        btn.grid(row=row, column=2, padx=5, pady=2)
        
        parent.columnconfigure(1, weight=1)

    def browse_file(self, var, open_mode):
        if open_mode:
            path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        else:
            path = filedialog.asksaveasfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],  defaultextension=".csv")
        
        if path:
            var.set(path)

    def log(self, message):
        self.log_queue.put(message)

    def process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_area.config(state='normal')
            self.log_area.insert(tk.END, str(msg) + "\n")
            self.log_area.see(tk.END)
            self.log_area.config(state='disabled')
        self.root.after(100, self.process_log_queue)

    def run_task(self, command):
        # Validate paths
        path_vars = {
            "Input Raw": self.input_path,
            "Output Clean": self.clean_path,
            "Output Dedup": self.dedup_path,
            "Output Train": self.train_path,
            "Output BERT": self.bert_path
        }
        
        for name, var in path_vars.items():
            val = var.get().strip()
            if not val:
                messagebox.showerror("Configuration Error", f"{name} path cannot be empty.")
                return
            p = Path(val)
            if p.is_dir():
                 messagebox.showerror("Configuration Error", f"{name} path '{val}' is a directory. Please specify a file path (e.g., end with .csv).")
                 return
                 
        paths = dataset_pipeline.PipelinePaths(
            input_raw=Path(self.input_path.get()),
            output_clean=Path(self.clean_path.get()),
            output_dedup=Path(self.dedup_path.get()),
            output_train=Path(self.train_path.get()),
        )
        
        convert_json = self.convert_json.get()
        validate_json = self.validate_json.get()
        gen_bert = self.generate_bert.get()
        bert_out_path = Path(self.bert_path.get())

        # Threading for non-blocking UI
        t = threading.Thread(target=self.execute_logic, args=(command, paths, convert_json, validate_json, gen_bert, bert_out_path))
        t.start()

    def execute_logic(self, command, paths, convert_json, validate_json, gen_bert, bert_out_path):
        # Redirect stdout to capture print() calls from the pipeline
        class TitleRedirector:
            def __init__(self, gui_logger):
                self.gui_logger = gui_logger
            def write(self, s):
                if s.strip():
                    self.gui_logger.log(s.strip())
            def flush(self):
                pass
        
        old_stdout = sys.stdout
        sys.stdout = TitleRedirector(self)
        
        try:
            self.log("-" * 50)
            self.log(f"Starting command: {command}")
            
            if command == "all":
                dataset_pipeline.run_all(paths, convert_response_json=convert_json, validate_response_json=validate_json)
            elif command == "clean":
                dataset_pipeline.clean_dataset(paths.input_raw, paths.output_clean, convert_response_json=convert_json)
                print(f"✅ Clean done: {paths.output_clean}")
            elif command == "dedup":
                dataset_pipeline.deduplicate_patterns(paths.output_clean, paths.output_dedup)
                print(f"✅ Dedup done: {paths.output_dedup}")
            elif command == "split":
                dataset_pipeline.split_patterns(paths.output_dedup, paths.output_train)
                print(f"✅ Split done: {paths.output_train}")
            elif command == "validate":
                dataset_pipeline.validate_dataset(paths.output_train, validate_response_json=validate_json)
                print(f"✅ Validate OK: {paths.output_train}")

            # Extra Step: BERT Generation
            # If "all" ran or "split" ran (since split produces train file), we can do BERT gen
            allowed_triggers = ["all", "split"] 
            if gen_bert and (command in allowed_triggers or (command == "validate")): # Allow if user just validates too? Maybe better only if new data generated
                 # But user might want to run just bert generation? No separate button for that in plan.
                 # Let's run it if generated or if command is ALL.
                 if Path(paths.output_train).exists():
                     self.generate_bert_dataset(paths.output_train, bert_out_path)
                 else:
                     print("⚠️ Convert to BERT skipped: Train file not found.")

            self.log("🏁 Task Completed Successfully.\n")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", str(e))
        finally:
            sys.stdout = old_stdout

    def generate_bert_dataset(self, input_csv, output_csv):
        print(f"\n🔄 Generating BERT Dataset (Extra Step)...")
        print(f"   Input: {input_csv}")
        print(f"   Output: {output_csv}")
        
        try:
            # We reuse the logic: Read -> Encure JSON -> Write
            # Since dataset_train is already in correct structure, we just need to ensure responses are JSON.
            
            # Using dataset_pipeline's flexible reader
            df = dataset_pipeline._read_csv_flexible(input_csv)
            df = dataset_pipeline._ensure_columns(df)
            df = dataset_pipeline._normalize_fields(df)
            
            # Create a helper to reuse the convert logic from pipeline
            # Note: convert_json.py logic is identical to dataset_pipeline._convert_response_to_json
            df["response"] = df["response"].apply(dataset_pipeline._convert_response_to_json)
            
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"   ✅ BERT Dataset Created: {output_csv}")
            
        except Exception as e:
             print(f"   ❌ Failed to generate BERT dataset: {e}")
             raise e

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetPipelineGUI(root)
    root.mainloop()
