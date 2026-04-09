#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset GUI Application
Tkinter-based GUI for 4-step dataset processing pipeline
Based on Figma design
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_pipeline_processor import DatasetPipelineProcessor, ProcessingConfig, create_sample_dataset


class DatasetPipelineGUI:
    """Main GUI Application for Dataset Processing Pipeline"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Processing Pipeline")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Configure style
        self.setup_styles()
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "dataset"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook (tabs)
        style = ttk.Style()
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        self.tab_logs = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="Step 1: Remove Duplicates")
        self.notebook.add(self.tab2, text="Step 2: Validate & Fix")
        self.notebook.add(self.tab3, text="Step 3: Split Patterns")
        self.notebook.add(self.tab4, text="Step 4: Convert & Split")
        self.notebook.add(self.tab_logs, text="Logs & Results")
        
        # Build tabs
        self.create_tab1()
        self.create_tab2()
        self.create_tab3()
        self.create_tab4()
        self.create_tab_logs()
        
        # Processing state
        self.processor = None
        self.is_processing = False
        self.current_df = None
    
    def setup_styles(self):
        """Setup tkinter styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    
    def create_tab1(self):
        """Create Tab 1: Remove Duplicates"""
        frame = ttk.LabelFrame(self.tab1, text="Step 1: Remove Duplicate Patterns", padding=15)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Description
        desc = ttk.Label(frame, text="Remove duplicate patterns within each row\nDelimiter: '|' (pipe character)", justify=tk.LEFT)
        desc.pack(anchor=tk.W, pady=(0, 10))
        
        # Input file selection
        input_frame = ttk.LabelFrame(frame, text="Input File", padding=10)
        input_frame.pack(fill=tk.X, pady=10)
        
        self.input_file_var = tk.StringVar(value=str(self.data_dir / "data_mentah.csv"))
        ttk.Label(input_frame, text="Input CSV:").pack(anchor=tk.W)
        input_entry = ttk.Entry(input_frame, textvariable=self.input_file_var, width=60)
        input_entry.pack(anchor=tk.W, pady=5)
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(anchor=tk.W, pady=5)
        ttk.Button(button_frame, text="Browse", command=self.browse_input_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Create Sample", command=self.create_sample).pack(side=tk.LEFT, padx=5)
        
        # Info display
        info_frame = ttk.LabelFrame(frame, text="Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # self.tab1_info = ttk.Label(
        #     info_frame,
        #     text="No file loaded yet",
        #     justify=tk.LEFT,
        #     wraplength=200,
        #     width=30,
        #     anchor=tk.NW
        # )
        # self.tab1_info.pack(anchor=tk.NW, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Preview", command=self.preview_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Process Step 1", command=lambda: self.run_step(1)).pack(side=tk.LEFT, padx=5)
    
    def create_tab2(self):
        """Create Tab 2: Validate & Fix"""
        frame = ttk.LabelFrame(self.tab2, text="Step 2: Validate & Fix Data", padding=15)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        desc = ttk.Label(frame, text="Validate data integrity and fix formatting issues\nRemove invalid rows, normalize data types", justify=tk.LEFT)
        desc.pack(anchor=tk.W, pady=(0, 10))
        
        # Validation options
        options_frame = ttk.LabelFrame(frame, text="Validation Options", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        self.validate_empty_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Remove empty patterns", variable=self.validate_empty_var).pack(anchor=tk.W)
        
        self.validate_type_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Normalize response types", variable=self.validate_type_var).pack(anchor=tk.W)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(frame, text="Validation Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.tab2_info = ttk.Label(stats_frame, text="Statistics will appear here after validation", justify=tk.LEFT, wraplength=400)
        self.tab2_info.pack(anchor=tk.NW, pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Process Step 2", command=lambda: self.run_step(2)).pack(side=tk.LEFT, padx=5)
    
    def create_tab3(self):
        """Create Tab 3: Split Patterns"""
        frame = ttk.LabelFrame(self.tab3, text="Step 3: Split Patterns", padding=15)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        desc = ttk.Label(frame, text="Split multiple patterns into separate rows\nEach pattern becomes a new dataset row", justify=tk.LEFT)
        desc.pack(anchor=tk.W, pady=(0, 10))
        
        # Split options
        options_frame = ttk.LabelFrame(frame, text="Split Options", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Pattern Delimiter:").pack(anchor=tk.W)
        self.delimiter_var = tk.StringVar(value="|")
        ttk.Entry(options_frame, textvariable=self.delimiter_var, width=5).pack(anchor=tk.W, pady=5)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(frame, text="Split Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.tab3_info = ttk.Label(stats_frame, text="Split statistics will appear here", justify=tk.LEFT, wraplength=400)
        self.tab3_info.pack(anchor=tk.NW, pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Process Step 3", command=lambda: self.run_step(3)).pack(side=tk.LEFT, padx=5)
    
    def create_tab4(self):
        """Create Tab 4: Convert & Split"""
        frame = ttk.LabelFrame(self.tab4, text="Step 4: Convert Responses & Create Datasets", padding=15)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        desc = ttk.Label(frame, text="Convert responses to JSON format and create train/validation split\nGenerate LSTM and BERT datasets", justify=tk.LEFT)
        desc.pack(anchor=tk.W, pady=(0, 10))
        
        # Split options
        options_frame = ttk.LabelFrame(frame, text="Dataset Options", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Validation Split (%):").pack(anchor=tk.W)
        self.split_var = tk.StringVar(value="20")
        ttk.Entry(options_frame, textvariable=self.split_var, width=5).pack(anchor=tk.W, pady=5)
        
        self.convert_json_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Convert responses to JSON", variable=self.convert_json_var).pack(anchor=tk.W, pady=5)
        
        # Output paths
        output_frame = ttk.LabelFrame(frame, text="Output Paths", padding=10)
        output_frame.pack(fill=tk.X, pady=10)
        
        lstm_path = str(self.data_dir / "lstm" / "dataset_training_lstm.csv")
        bert_path = str(self.data_dir / "bert" / "dataset_training_bert.csv")
        
        ttk.Label(output_frame, text="LSTM Dataset:", font=('Arial', 9)).pack(anchor=tk.W)
        ttk.Label(output_frame, text=lstm_path, font=('Courier', 8), foreground='blue').pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(output_frame, text="BERT Dataset:", font=('Arial', 9)).pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(output_frame, text=bert_path, font=('Courier', 8), foreground='blue').pack(anchor=tk.W)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(frame, text="Final Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.tab4_info = ttk.Label(stats_frame, text="Final statistics will appear here", justify=tk.LEFT, wraplength=500)
        self.tab4_info.pack(anchor=tk.NW, pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Process Step 4", command=lambda: self.run_step(4)).pack(side=tk.LEFT, padx=5)
    
    def create_tab_logs(self):
        """Create Logs Tab"""
        frame = ttk.Frame(self.tab_logs, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(frame, text="Processing Logs & Results", font=('Arial', 12, 'bold'))
        title.pack(anchor=tk.W, pady=(0, 10))
        
        # Logs display
        self.logs_text = scrolledtext.ScrolledText(frame, height=25, width=120, wrap=tk.WORD, font=('Courier', 9))
        self.logs_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Logs", command=self.save_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Complete Pipeline", command=self.run_complete_pipeline).pack(side=tk.LEFT, padx=5)
    
    def browse_input_file(self):
        """Browse for input file"""
        filename = filedialog.askopenfilename(
            title="Select Input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(self.data_dir)
        )
        if filename:
            self.input_file_var.set(filename)
            self.preview_input()
    
    def create_sample(self):
        """Create sample dataset"""
        sample_path = str(self.data_dir / "data_mentah.csv")
        create_sample_dataset(sample_path)
        self.input_file_var.set(sample_path)
        self.log_message(f"✓ Sample dataset created: {sample_path}")
        self.preview_input()
    
    def preview_input(self):
        """Preview input file"""
        try:
            import pandas as pd
            input_file = self.input_file_var.get()
            if not Path(input_file).exists():
                messagebox.showerror("Error", f"File not found: {input_file}")
                return
            
            df = pd.read_csv(input_file)
            info_text = f"File: {Path(input_file).name}\n"
            info_text += f"Rows: {len(df)}\n"
            info_text += f"Columns: {', '.join(df.columns)}\n\n"
            info_text += "First 3 rows:\n"
            for i in range(min(3, len(df))):
                info_text += f"\nRow {i+1}:\n"
                for col in df.columns:
                    val = str(df.iloc[i][col])[:50]
                    info_text += f"  {col}: {val}\n"
            
            # self.tab1_info.config(text=info_text)
            self.log_message(f"✓ Loaded: {input_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview: {str(e)}")
            self.log_message(f"✗ Error: {str(e)}")
    
    def run_step(self, step: int):
        """Run a specific processing step"""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return
        
        self.is_processing = True
        thread = threading.Thread(target=self._run_step_thread, args=(step,), daemon=True)
        thread.start()
    
    def _run_step_thread(self, step: int):
        """Thread function for running steps"""
        try:
            input_file = self.input_file_var.get()
            if not Path(input_file).exists():
                self.log_message(f"✗ Input file not found: {input_file}")
                self.is_processing = False
                return
            
            # Create config
            config = ProcessingConfig(
                input_file=input_file,
                output_lstm=str(self.data_dir / "lstm" / "dataset_training_lstm.csv"),
                output_bert=str(self.data_dir / "bert" / "dataset_training_bert.csv")
            )
            
            self.processor = DatasetPipelineProcessor(config)
            
            self.log_message(f"\n{'='*60}")
            self.log_message(f"Processing Step {step}")
            self.log_message(f"{'='*60}\n")
            
            if step == 1:
                df = self.processor.step_1_remove_duplicates(input_file)
                if df is not None:
                    stats = self.processor.stats["step_1"]
                    # self.update_tab1_info(stats)
                    self.current_df = df
                    self.log_message(f"✓ Step 1 completed")
                    self.log_message(json.dumps(stats, indent=2))
            
            elif step == 2:
                if self.current_df is None:
                    self.log_message("✗ No data from Step 1. Run Step 1 first.")
                else:
                    df = self.processor.step_2_validate_fix(self.current_df)
                    if df is not None:
                        stats = self.processor.stats["step_2"]
                        self.update_tab2_info(stats)
                        self.current_df = df
                        self.log_message(f"✓ Step 2 completed")
                        self.log_message(json.dumps(stats, indent=2))
            
            elif step == 3:
                if self.current_df is None:
                    self.log_message("✗ No data from Step 2. Run Steps 1-2 first.")
                else:
                    df = self.processor.step_3_split_patterns(self.current_df)
                    if df is not None:
                        stats = self.processor.stats["step_3"]
                        self.update_tab3_info(stats)
                        self.current_df = df
                        self.log_message(f"✓ Step 3 completed")
                        self.log_message(json.dumps(stats, indent=2))
            
            elif step == 4:
                if self.current_df is None:
                    self.log_message("✗ No data from Step 3. Run Steps 1-3 first.")
                else:
                    lstm_path, bert_path = self.processor.step_4_convert_responses(self.current_df)
                    if lstm_path and bert_path:
                        stats = self.processor.stats["step_4"]
                        self.update_tab4_info(stats)
                        self.log_message(f"✓ Step 4 completed")
                        self.log_message(json.dumps(stats, indent=2))
                        self.log_message(f"\n✓ Output files created:")
                        self.log_message(f"  LSTM: {lstm_path}")
                        self.log_message(f"  BERT: {bert_path}")
        
        except Exception as e:
            self.log_message(f"✗ Error: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
        
        finally:
            self.is_processing = False
    
    def run_complete_pipeline(self):
        """Run complete pipeline"""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return
        
        thread = threading.Thread(target=self._run_complete_pipeline_thread, daemon=True)
        thread.start()
    
    def _run_complete_pipeline_thread(self):
        """Thread function for complete pipeline"""
        try:
            self.is_processing = True
            input_file = self.input_file_var.get()
            
            if not Path(input_file).exists():
                self.log_message(f"✗ Input file not found: {input_file}")
                return
            
            config = ProcessingConfig(
                input_file=input_file,
                output_lstm=str(self.data_dir / "lstm" / "dataset_training_lstm.csv"),
                output_bert=str(self.data_dir / "bert" / "dataset_training_bert.csv"),
                validation_split=float(self.split_var.get()) / 100.0
            )
            
            processor = DatasetPipelineProcessor(config)
            
            self.log_message(f"\n{'='*60}")
            self.log_message("Running Complete Pipeline")
            self.log_message(f"{'='*60}\n")
            
            result = processor.process_all()
            
            if result["success"]:
                self.log_message("\n✓ PIPELINE COMPLETED SUCCESSFULLY!")
                self.log_message(f"\nOutput Files:")
                self.log_message(f"  LSTM: {result['lstm_output']}")
                self.log_message(f"  BERT: {result['bert_output']}")
                self.log_message(f"\nStatistics:")
                self.log_message(json.dumps(result['stats'], indent=2))
                messagebox.showinfo("Success", "Pipeline completed successfully!")
            else:
                self.log_message(f"\n✗ Pipeline failed: {result['error']}")
                messagebox.showerror("Error", f"Pipeline failed: {result['error']}")
        
        except Exception as e:
            self.log_message(f"✗ Error: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            messagebox.showerror("Error", str(e))
        
        finally:
            self.is_processing = False
    
    def update_tab1_info(self, stats: dict):
        """Update Tab 1 information"""
        text = f"Rows Before: {stats.get('rows_before', 0)}\n"
        text += f"Rows After: {stats.get('rows_after', 0)}\n"
        text += f"Total Patterns Before: {stats.get('patterns_before', 0)}\n"
        text += f"Total Patterns After: {stats.get('patterns_after', 0)}\n"
        text += f"Duplicates Removed: {stats.get('duplicates_removed', 0)}\n"
        text += f"Rows with Duplicates: {stats.get('rows_with_duplicates', 0)}"
        # self.tab1_info.config(text=text)
    
    def update_tab2_info(self, stats: dict):
        """Update Tab 2 information"""
        text = f"Rows Before: {stats.get('rows_before', 0)}\n"
        text += f"Rows After: {stats.get('rows_after', 0)}\n"
        text += f"Rows Removed: {stats.get('rows_removed', 0)}\n"
        text += f"Valid Intents: {stats.get('valid_intents', 0)}\n"
        text += f"Response Types: {', '.join(stats.get('valid_response_types', []))}"
        self.tab2_info.config(text=text)
    
    def update_tab3_info(self, stats: dict):
        """Update Tab 3 information"""
        text = f"Rows Before: {stats.get('rows_before', 0)}\n"
        text += f"Rows After: {stats.get('rows_after', 0)}\n"
        text += f"New Rows Created: {stats.get('rows_created', 0)}\n"
        text += f"Average Patterns per Row: {stats.get('avg_patterns_per_row', 0):.2f}"
        self.tab3_info.config(text=text)
    
    def update_tab4_info(self, stats: dict):
        """Update Tab 4 information"""
        text = f"Total Rows: {stats.get('total_rows', 0)}\n"
        text += f"Train Rows: {stats.get('train_rows', 0)}\n"
        text += f"Validation Rows: {stats.get('val_rows', 0)}\n"
        text += f"Train Split: {stats.get('train_split_percentage', 0):.1f}%\n"
        text += f"Unique Intents: {stats.get('unique_intents', 0)}\n"
        text += f"Response Types: {', '.join(stats.get('response_types', []))}"
        self.tab4_info.config(text=text)
    
    def log_message(self, message: str):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_text.see(tk.END)
        self.root.update()
    
    def clear_logs(self):
        """Clear logs"""
        self.logs_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=str(self.data_dir)
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.logs_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Logs saved to: {filename}")


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = DatasetPipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
