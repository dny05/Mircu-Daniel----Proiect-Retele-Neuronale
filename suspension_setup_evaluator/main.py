"""
Main Application Entry Point
Launches the Suspension Setup Evaluator GUI
"""

import sys
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "gui"))

from main_window import SuspensionSetupGUI


def setup_logging():
    """Configure logging for the application"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'application.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_directories():
    """Ensure required directories exist"""
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "data/logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Main application entry point"""
    # Setup
    setup_logging()
    check_directories()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Suspension Setup Evaluator")
    
    # Create and run GUI
    try:
        root = tk.Tk()
        app = SuspensionSetupGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()