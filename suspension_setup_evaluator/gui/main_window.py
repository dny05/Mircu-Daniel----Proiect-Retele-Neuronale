"""
Main GUI Window
Tkinter-based interface for the Suspension Setup Evaluator
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import TelemetryDataLoader
from preprocessing import TelemetryPreprocessor
from feature_extraction import FeatureExtractor
from models import create_model
from trainer import ModelTrainer
from evaluator import SetupEvaluator
import logging

logger = logging.getLogger(__name__)


class SuspensionSetupGUI:
    """Main GUI application"""
    
    def __init__(self, root):
        """Initialize GUI"""
        self.root = root
        self.root.title("Formula Student - Suspension Setup Evaluator")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.data_loader = TelemetryDataLoader()
        self.preprocessor = TelemetryPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        self.current_data = None
        self.current_features = None
        self.classifier_model = None
        self.evaluator = None
        
        # Setup GUI
        self.setup_ui()
        self.check_trained_model()
        
        logger.info("GUI initialized")
    
    def setup_ui(self):
        """Setup UI components"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_evaluation_tab()
        self.create_results_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_data_tab(self):
        """Create data loading and preprocessing tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📁 Data Loading")
        
        # File selection
        file_frame = ttk.LabelFrame(tab, text="Select Telemetry File", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=80).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Load & Process", command=self.load_and_process).pack(side=tk.LEFT, padx=5)
        
        # Data info
        info_frame = ttk.LabelFrame(tab, text="Data Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=25, wrap=tk.WORD)
        self.data_info_text.pack(fill=tk.BOTH, expand=True)
    
    def create_training_tab(self):
        """Create model training tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎓 Model Training")
        
        # Training controls
        control_frame = ttk.LabelFrame(tab, text="Training Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Note: Training requires labeled data").pack()
        ttk.Label(control_frame, text="For now, load a pre-trained model or train on synthetic data").pack()
        
        ttk.Button(control_frame, text="Train on Sample Data", 
                  command=self.train_on_sample_data).pack(pady=10)
        ttk.Button(control_frame, text="Load Pre-trained Model", 
                  command=self.load_pretrained_model).pack(pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD)
        self.training_log.pack(fill=tk.BOTH, expand=True)
    
    def create_evaluation_tab(self):
        """Create evaluation tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔍 Evaluation")
        
        # Evaluation controls
        control_frame = ttk.LabelFrame(tab, text="Evaluation Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Loaded data will be evaluated automatically").pack()
        ttk.Button(control_frame, text="Run Evaluation", 
                  command=self.run_evaluation, state=tk.DISABLED).pack(pady=10)
        self.eval_button = control_frame.winfo_children()[-1]
        
        # Progress
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=5)
        
        # Evaluation output
        output_frame = ttk.LabelFrame(tab, text="Evaluation Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.eval_output = scrolledtext.ScrolledText(output_frame, height=20, wrap=tk.WORD)
        self.eval_output.pack(fill=tk.BOTH, expand=True)
    
    def create_results_tab(self):
        """Create results and recommendations tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Results & Recommendations")
        
        # Results display
        results_frame = ttk.LabelFrame(tab, text="Setup Recommendations", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=25, wrap=tk.WORD, font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Export button
        ttk.Button(tab, text="Export Report", command=self.export_report).pack(pady=10)
    
    def browse_file(self):
        """Open file browser"""
        filename = filedialog.askopenfilename(
            title="Select Telemetry CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="data/raw"
        )
        if filename:
            self.file_path_var.set(filename)
    
    def load_and_process(self):
        """Load and process telemetry file"""
        filepath = self.file_path_var.get()
        if not filepath:
            messagebox.showwarning("No File", "Please select a telemetry file first")
            return
        
        self.status_var.set("Loading and processing data...")
        self.data_info_text.delete(1.0, tk.END)
        
        def process():
            try:
                # Load data
                df = self.data_loader.load_csv(filepath)
                self.log_to_text(self.data_info_text, f"✓ Loaded {len(df)} samples\n\n")
                
                # Get data quality report
                quality = self.data_loader.validate_data_quality(df)
                self.log_to_text(self.data_info_text, "=== Data Quality Report ===\n")
                self.log_to_text(self.data_info_text, f"Total samples: {quality['total_samples']}\n")
                self.log_to_text(self.data_info_text, f"Duration: {quality['duration_seconds']:.2f} seconds\n")
                self.log_to_text(self.data_info_text, f"Sampling rate: {quality['sampling_rate_hz']:.2f} Hz\n\n")
                
                # Split data
                timestamps, susp_data, acc_data, rot_data = self.data_loader.split_data(df)
                
                # Combine all sensor data for processing
                import numpy as np
                all_data = np.hstack([susp_data, acc_data, rot_data])
                
                # Preprocess
                self.log_to_text(self.data_info_text, "=== Preprocessing ===\n")
                processed_windows, metadata = self.preprocessor.preprocess_pipeline(
                    all_data,
                    sampling_rate=quality['sampling_rate_hz']
                )
                
                self.log_to_text(self.data_info_text, f"✓ Created {metadata['n_windows']} windows\n")
                self.log_to_text(self.data_info_text, f"Window size: {metadata['window_size']} samples\n\n")
                
                # Extract features
                self.log_to_text(self.data_info_text, "=== Feature Extraction ===\n")
                features = self.feature_extractor.extract_features_from_windows(
                    processed_windows,
                    sampling_rate=quality['sampling_rate_hz']
                )
                
                self.log_to_text(self.data_info_text, f"✓ Extracted {features.shape[1]} features per window\n")
                self.log_to_text(self.data_info_text, f"Total feature matrix: {features.shape}\n\n")
                
                # Store data
                self.current_data = df
                self.current_features = features
                
                self.log_to_text(self.data_info_text, "✓ Data processing complete!\n")
                self.status_var.set("Data loaded and processed successfully")
                
                # Enable evaluation if model is loaded
                if self.classifier_model is not None:
                    self.eval_button.config(state=tk.NORMAL)
                    # Auto-run evaluation
                    self.root.after(100, self.run_evaluation)
                
            except Exception as e:
                self.log_to_text(self.data_info_text, f"\n❌ Error: {str(e)}\n")
                self.status_var.set("Error processing data")
                logger.error(f"Error processing data: {e}", exc_info=True)
        
        # Run in thread
        threading.Thread(target=process, daemon=True).start()
    
    def check_trained_model(self):
        """Check if trained model exists"""
        model_path = Path("data/models/best_model.pth")
        if model_path.exists():
            try:
                self.classifier_model = create_model("classifier")
                self.classifier_model.load_state_dict(
                    torch.load(model_path, map_location='cpu')['model_state_dict']
                )
                self.evaluator = SetupEvaluator(self.classifier_model)
                self.status_var.set("✓ Pre-trained model loaded")
                logger.info("Loaded pre-trained model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def train_on_sample_data(self):
        """Train model on synthetic sample data"""
        messagebox.showinfo("Training", 
                          "Training on sample data...\n"
                          "This may take a few minutes.")
        
        self.training_log.delete(1.0, tk.END)
        self.log_to_text(self.training_log, "Generating synthetic training data...\n")
        
        def train():
            try:
                import numpy as np
                import torch
                
                # Generate synthetic data
                n_samples = 1000
                n_features = 30
                X_train = np.random.randn(n_samples, n_features)
                # Simulate understeer (0) vs oversteer (1) with some logic
                y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
                
                self.log_to_text(self.training_log, f"Generated {n_samples} samples\n\n")
                
                # Create and train model
                self.classifier_model = create_model("classifier")
                trainer = ModelTrainer(self.classifier_model)
                trainer.setup_optimizer_and_loss("classifier")
                
                self.log_to_text(self.training_log, "Training model...\n")
                
                train_loader, val_loader = trainer.prepare_data(X_train, y_train)
                history = trainer.train(train_loader, val_loader, epochs=20)
                
                self.log_to_text(self.training_log, "\n✓ Training complete!\n")
                self.log_to_text(self.training_log, f"Final train loss: {history['train_loss'][-1]:.4f}\n")
                self.log_to_text(self.training_log, f"Final val loss: {history['val_loss'][-1]:.4f}\n")
                
                # Initialize evaluator
                self.evaluator = SetupEvaluator(self.classifier_model)
                
                self.status_var.set("✓ Model trained successfully")
                messagebox.showinfo("Success", "Model trained successfully!")
                
                # Enable evaluation
                if self.current_features is not None:
                    self.eval_button.config(state=tk.NORMAL)
                
            except Exception as e:
                self.log_to_text(self.training_log, f"\n❌ Training error: {str(e)}\n")
                logger.error(f"Training error: {e}", exc_info=True)
        
        threading.Thread(target=train, daemon=True).start()
    
    def load_pretrained_model(self):
        """Load a pre-trained model"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")],
            initialdir="data/models"
        )
        if filename:
            try:
                import torch
                self.classifier_model = create_model("classifier")
                checkpoint = torch.load(filename, map_location='cpu')
                self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
                self.evaluator = SetupEvaluator(self.classifier_model)
                
                messagebox.showinfo("Success", "Model loaded successfully!")
                self.status_var.set("✓ Model loaded")
                
                if self.current_features is not None:
                    self.eval_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def run_evaluation(self):
        """Run evaluation on loaded data"""
        if self.current_features is None:
            messagebox.showwarning("No Data", "Please load telemetry data first")
            return
        
        if self.evaluator is None:
            messagebox.showwarning("No Model", "Please train or load a model first")
            return
        
        self.status_var.set("Running evaluation...")
        self.eval_output.delete(1.0, tk.END)
        self.progress_var.set(0)
        
        def evaluate():
            try:
                self.progress_var.set(30)
                self.log_to_text(self.eval_output, "Evaluating telemetry data...\n\n")
                
                # Run evaluation
                report = self.evaluator.evaluate_telemetry_file(self.current_features)
                
                self.progress_var.set(80)
                
                # Display results
                self.display_results(report)
                
                self.progress_var.set(100)
                self.status_var.set("✓ Evaluation complete")
                
            except Exception as e:
                self.log_to_text(self.eval_output, f"\n❌ Evaluation error: {str(e)}\n")
                logger.error(f"Evaluation error: {e}", exc_info=True)
        
        threading.Thread(target=evaluate, daemon=True).start()
    
    def display_results(self, report):
        """Display evaluation results"""
        self.results_text.delete(1.0, tk.END)
        
        # Format report
        text = "=" * 60 + "\n"
        text += "  SUSPENSION SETUP EVALUATION REPORT\n"
        text += "=" * 60 + "\n\n"
        
        eval_data = report['evaluation']
        rec_data = report['recommendations']
        
        text += f"Detected Behavior: {rec_data['detected_behavior'].upper()}\n"
        text += f"Confidence: {rec_data['confidence']*100:.1f}%\n"
        text += f"Reliability: {'HIGH ✓' if rec_data['reliable'] else 'LOW ⚠'}\n\n"
        
        text += "-" * 60 + "\n"
        text += "ANALYSIS DETAILS\n"
        text += "-" * 60 + "\n"
        text += f"Windows analyzed: {eval_data['n_windows']}\n"
        text += f"Understeer windows: {eval_data['n_understeer']} ({eval_data['understeer_ratio']*100:.1f}%)\n"
        text += f"Oversteer windows: {eval_data['n_oversteer']} ({eval_data['oversteer_ratio']*100:.1f}%)\n\n"
        
        if rec_data['reliable']:
            text += "-" * 60 + "\n"
            text += "SETUP RECOMMENDATIONS\n"
            text += "-" * 60 + "\n"
            text += f"{rec_data['recommendations']['message']}\n\n"
            
            text += "Specific Actions:\n"
            for i, action in enumerate(rec_data['specific_actions'], 1):
                text += f"  {i}. {action}\n"
        else:
            text += "-" * 60 + "\n"
            text += "⚠ CONFIDENCE TOO LOW FOR RELIABLE RECOMMENDATIONS\n"
            text += "-" * 60 + "\n"
            text += rec_data['recommendations']['message'] + "\n"
        
        text += "\n" + "=" * 60 + "\n"
        
        self.results_text.insert(1.0, text)
        self.log_to_text(self.eval_output, text)
        
        # Store report for export
        self.last_report = report
    
    def export_report(self):
        """Export evaluation report"""
        if not hasattr(self, 'last_report'):
            messagebox.showwarning("No Report", "No evaluation report to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")],
            initialdir="data/logs"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.last_report, f, indent=2)
                messagebox.showinfo("Success", f"Report exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def log_to_text(self, widget, message):
        """Thread-safe logging to text widget"""
        def append():
            widget.insert(tk.END, message)
            widget.see(tk.END)
        
        self.root.after(0, append)


if __name__ == "__main__":
    import torch
    root = tk.Tk()
    app = SuspensionSetupGUI(root)
    root.mainloop()