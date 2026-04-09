#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import queue
import sys
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable

"""GUI for scripts/dataset_pipeline.py.

Goals:
- Cleaner UI (tabbed, consistent spacing)
- Safe background execution (no UI freeze)
- Capture prints into GUI log
- Optional config load/save
"""


# Import existing pipeline logic
try:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import dataset_pipeline
except (ImportError, SystemExit) as e:
    messagebox.showerror("Import Error", f"Could not import dataset_pipeline: {e}")
    raise SystemExit(1)

@dataclass
class GUIConfig:
    input_raw: str
    output_clean: str
    output_dedup: str
    output_train: str
    output_bert: str
    convert_response_json: bool = False
    validate_response_json: bool = False
    generate_bert: bool = False


class _QueueWriter:
    """File-like object to redirect stdout/stderr into a queue."""

    def __init__(self, write_cb: Callable[[str], None]):
        self._write_cb = write_cb

    def write(self, s: str) -> None:
        if not s:
            return
        # Keep newlines but avoid spamming empty lines
        for line in str(s).splitlines():
            if line.strip():
                self._write_cb(line.rstrip())

    def flush(self) -> None:
        return

class DatasetPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Processing Pipeline")
        self.root.minsize(860, 680)

        self._is_running = False
        self._stdout_lock = threading.Lock()

        # Styles
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure("TButton", padding=(10, 6))
        self.style.configure("TLabel", padding=(2, 2))
        self.style.configure("Header.TLabel", font=("TkDefaultFont", 10, "bold"))

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

        self.status_var = tk.StringVar(value="Ready")

        # Queue for thread-safe logging
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        # UI Layout
        self._buttons: list[ttk.Button] = []
        self.create_widgets()
        self.root.after(100, self.process_log_queue)

    def create_widgets(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # Top row: title + config buttons
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Dataset Processing Pipeline", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        ttk.Button(header, text="Reset Defaults", command=self.reset_defaults).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(header, text="Load Config", command=self.load_config).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(header, text="Save Config", command=self.save_config).grid(row=0, column=3, padx=(8, 0))

        # Tabs
        notebook = ttk.Notebook(outer)
        notebook.grid(row=1, column=0, sticky="nsew")

        tab_paths = ttk.Frame(notebook, padding=10)
        tab_options = ttk.Frame(notebook, padding=10)
        tab_run = ttk.Frame(notebook, padding=10)
        tab_logs = ttk.Frame(notebook, padding=10)

        notebook.add(tab_paths, text="Paths")
        notebook.add(tab_options, text="Options")
        notebook.add(tab_run, text="Run")
        notebook.add(tab_logs, text="Logs")

        # --- Paths tab ---
        tab_paths.columnconfigure(1, weight=1)
        self.create_path_entry(tab_paths, "Input Raw CSV", self.input_path, 0, is_input=True)
        self.create_path_entry(tab_paths, "Output Train CSV", self.train_path, 1)

        tip = ttk.Label(
            tab_paths,
            text=(
                "Tip: Output paths boleh diarahkan ke folder lain. "
                "Pipeline akan membuat folder jika belum ada."
            ),
        )
        tip.grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # --- Options tab ---
        tab_options.columnconfigure(0, weight=1)
        opts = ttk.LabelFrame(tab_options, text="Dataset Options", padding=10)
        opts.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(opts, text="Convert response to JSON (text/list)", variable=self.convert_json).grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Checkbutton(opts, text="Validate response JSON (auto-convert enabled)", variable=self.validate_json).grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Checkbutton(opts, text="Generate extra BERT CSV after run", variable=self.generate_bert).grid(
            row=2, column=0, sticky="w", pady=3
        )

        # --- Run tab ---
        tab_run.columnconfigure(0, weight=1)

        run_box = ttk.LabelFrame(tab_run, text="Pipeline Actions", padding=10)
        run_box.grid(row=0, column=0, sticky="ew")
        run_box.columnconfigure((0, 1, 2), weight=1)

        self._add_button(run_box, "Run ALL", lambda: self.run_task("all"), row=0, col=0)
        self._add_button(run_box, "Clean", lambda: self.run_task("clean"), row=0, col=1)
        self._add_button(run_box, "Dedup", lambda: self.run_task("dedup"), row=0, col=2)
        self._add_button(run_box, "Split", lambda: self.run_task("split"), row=1, col=0)
        self._add_button(run_box, "Validate", lambda: self.run_task("validate"), row=1, col=1)
        self._add_button(run_box, "Generate BERT", lambda: self.run_task("bert"), row=1, col=2)

        progress_box = ttk.Frame(tab_run)
        progress_box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        progress_box.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_box, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_box, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(6, 0))

        # --- Logs tab ---
        tab_logs.columnconfigure(0, weight=1)
        tab_logs.rowconfigure(1, weight=1)

        log_controls = ttk.Frame(tab_logs)
        log_controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        log_controls.columnconfigure(0, weight=1)

        ttk.Button(log_controls, text="Clear", command=self.clear_log).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(log_controls, text="Copy", command=self.copy_log).grid(row=0, column=2, padx=(8, 0))

        self.log_area = scrolledtext.ScrolledText(tab_logs, state="disabled", height=18, wrap=tk.WORD)
        self.log_area.grid(row=1, column=0, sticky="nsew")

        # Status bar
        status = ttk.Frame(outer)
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        status.columnconfigure(0, weight=1)
        ttk.Separator(status, orient=tk.HORIZONTAL).grid(row=0, column=0, sticky="ew", columnspan=2)
        ttk.Label(status, textvariable=self.status_var).grid(row=1, column=0, sticky="w")

    def _add_button(self, parent, text: str, command: Callable[[], None], *, row: int, col: int) -> None:
        btn = ttk.Button(parent, text=text, command=command)
        btn.grid(row=row, column=col, sticky="ew", padx=6, pady=6)
        self._buttons.append(btn)

    def create_path_entry(self, parent, label_text, var, row, is_input=False):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=(10, 8), pady=4)

        btn = ttk.Button(parent, text="Browse…", command=lambda: self.browse_file(var, open_mode=is_input))
        btn.grid(row=row, column=2, padx=(0, 4), pady=4)

    def browse_file(self, var, open_mode: bool):
        if open_mode:
            path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        else:
            path = filedialog.asksaveasfilename(
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                defaultextension=".csv",
            )
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

    def clear_log(self):
        self.log_area.config(state="normal")
        self.log_area.delete("1.0", tk.END)
        self.log_area.config(state="disabled")

    def copy_log(self):
        text = self.log_area.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Log copied to clipboard")

    def _ui(self, fn: Callable[[], None]) -> None:
        self.root.after(0, fn)

    def _set_running(self, running: bool, status: str) -> None:
        self._is_running = running
        self.status_var.set(status)
        for b in self._buttons:
            b.configure(state=("disabled" if running else "normal"))
        if running:
            try:
                self.progress.start(12)
            except Exception:
                pass
        else:
            try:
                self.progress.stop()
            except Exception:
                pass

    def reset_defaults(self):
        self.defaults = dataset_pipeline.PipelinePaths.defaults()
        self.input_path.set(str(self.defaults.input_raw))
        self.clean_path.set(str(self.defaults.output_clean))
        self.dedup_path.set(str(self.defaults.output_dedup))
        self.train_path.set(str(self.defaults.output_train))
        self.bert_path.set(str(self.defaults.output_train).replace(".csv", "_bert.csv"))
        self.status_var.set("Defaults restored")

    def _current_config(self) -> GUIConfig:
        return GUIConfig(
            input_raw=self.input_path.get().strip(),
            output_clean=self.clean_path.get().strip(),
            output_dedup=self.dedup_path.get().strip(),
            output_train=self.train_path.get().strip(),
            output_bert=self.bert_path.get().strip(),
            convert_response_json=bool(self.convert_json.get()),
            validate_response_json=bool(self.validate_json.get()),
            generate_bert=bool(self.generate_bert.get()),
        )

    def save_config(self):
        cfg = self._current_config()
        path = filedialog.asksaveasfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            defaultextension=".json",
            initialfile="dataset_pipeline_config.json",
        )
        if not path:
            return
        Path(path).write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"Config saved: {path}")

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Load Config", f"Invalid JSON: {e}")
            return

        # Apply config (tolerate missing keys)
        self.input_path.set(str(raw.get("input_raw", self.input_path.get())))
        self.clean_path.set(str(raw.get("output_clean", self.clean_path.get())))
        self.dedup_path.set(str(raw.get("output_dedup", self.dedup_path.get())))
        self.train_path.set(str(raw.get("output_train", self.train_path.get())))
        self.bert_path.set(str(raw.get("output_bert", self.bert_path.get())))
        self.convert_json.set(bool(raw.get("convert_response_json", self.convert_json.get())))
        self.validate_json.set(bool(raw.get("validate_response_json", self.validate_json.get())))
        self.generate_bert.set(bool(raw.get("generate_bert", self.generate_bert.get())))

        self.status_var.set(f"Config loaded: {path}")

    def run_task(self, command):
        if self._is_running:
            return

        cfg = self._current_config()

        # Basic validation
        try:
            paths = dataset_pipeline.PipelinePaths(
                input_raw=Path(cfg.input_raw),
                output_clean=Path(cfg.output_clean),
                output_dedup=Path(cfg.output_dedup),
                output_train=Path(cfg.output_train),
            )
            bert_out_path = Path(cfg.output_bert)
        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))
            return

        def err(msg: str) -> None:
            messagebox.showerror("Configuration Error", msg)

        # Ensure non-empty
        for label, p in {
            "Input Raw": paths.input_raw,
            "Output Clean": paths.output_clean,
            "Output Dedup": paths.output_dedup,
            "Output Train": paths.output_train,
            "Output BERT": bert_out_path,
        }.items():
            if not str(p).strip():
                err(f"{label} path cannot be empty")
                return
            if p.exists() and p.is_dir():
                err(f"{label} points to a directory. Please select a CSV file path.")
                return

        # Existence rules (command-specific)
        if command in {"all", "clean"} and not paths.input_raw.exists():
            err(f"Input raw CSV not found: {paths.input_raw}")
            return
        if command == "dedup" and not paths.output_clean.exists():
            err(f"Clean CSV not found (run Clean first): {paths.output_clean}")
            return
        if command == "split" and not paths.output_dedup.exists():
            err(f"Dedup CSV not found (run Dedup first): {paths.output_dedup}")
            return
        if command in {"validate", "bert"} and not paths.output_train.exists():
            err(f"Train CSV not found (run Split/All first): {paths.output_train}")
            return

        self.clear_log()
        self._set_running(True, f"Running: {command}…")

        t = threading.Thread(
            target=self.execute_logic,
            args=(command, paths, cfg.convert_response_json, cfg.validate_response_json, cfg.generate_bert, bert_out_path),
            daemon=True,
        )
        t.start()

    def execute_logic(self, command, paths, convert_json, validate_json, gen_bert, bert_out_path):
        writer = _QueueWriter(self.log)

        # Redirect stdout/stderr *temporarily*.
        with self._stdout_lock:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = writer
            sys.stderr = writer

            try:
                print("-" * 60)
                print(f"Command: {command}")
                print(f"Input : {paths.input_raw}")
                print(f"Clean : {paths.output_clean}")
                print(f"Dedup : {paths.output_dedup}")
                print(f"Train : {paths.output_train}")
                print(f"BERT  : {bert_out_path}")
                print("-" * 60)

                # If user asks to validate JSON, auto-enable conversion (same behavior as dataset_pipeline.run_all)
                if validate_json and not convert_json:
                    print("ℹ️  validate JSON enabled → auto-enabling convert JSON")
                    convert_json = True

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
                elif command == "bert":
                    dataset_pipeline.generate_bert_dataset(
                        paths.output_train,
                        bert_out_path,
                        ensure_response_json=True,
                        validate_response_json=validate_json,
                    )
                    print(f"✅ BERT dataset written: {bert_out_path}")

                # Optional post-step: generate BERT dataset after all/split/validate
                if gen_bert and command in {"all", "split", "validate"}:
                    if Path(paths.output_train).exists():
                        dataset_pipeline.generate_bert_dataset(
                            paths.output_train,
                            bert_out_path,
                            ensure_response_json=True,
                            validate_response_json=validate_json,
                        )
                        print(f"✅ Extra BERT dataset written: {bert_out_path}")
                    else:
                        print("⚠️  Extra BERT skipped: train CSV not found")

                print("\n🏁 Completed")

                self._ui(lambda: self._set_running(False, "Done"))

            except Exception as e:
                import traceback

                print(f"❌ Error: {e}")
                print(traceback.format_exc())

                # Capture 'e' ke local var sebelum closure — Python (PEP 3110)
                # menghapus variabel except setelah blok selesai, sehingga
                # closure yang dipanggil via Tkinter event loop akan NameError.
                _err_msg = str(e)

                def show():
                    self._set_running(False, "Failed")
                    messagebox.showerror("Pipeline Error", _err_msg)

                self._ui(show)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    # (generate_bert_dataset is now provided by dataset_pipeline.generate_bert_dataset)

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetPipelineGUI(root)
    root.mainloop()
