import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get scripts directory
SCRIPTS_DIR = Path(__file__).parent

# Define pipeline steps in order
PIPELINE_STEPS = [
    "1hapus_duplikat.py",
    "2fix_csv.py",
    "3fix_csv_malformed.py",
    "4validate_csv.py",
    "5data_splitter.py"
]

def run_script(script_name: str, python_path: str = None) -> bool:
    """
    Run a single script in the pipeline.
    
    Args:
        script_name: Name of the script to run
        python_path: Path to Python executable (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"❌ Script not found: {script_path}")
        return False
    
    logger.info(f"▶️  Running: {script_name}")
    
    try:
        # Use provided Python path or current Python
        python_cmd = python_path or sys.executable
        
        # Run the script with UTF-8 encoding
        result = subprocess.run(
            [python_cmd, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=SCRIPTS_DIR.parent  # Run from project root
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            logger.info(f"✅ Completed: {script_name}")
            return True
        else:
            logger.error(f"❌ Failed: {script_name}")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        return False

def run_pipeline(continue_on_error: bool = False, python_path: str = None) -> dict:
    """
    Run the complete data cleaning and validation pipeline.
    
    Args:
        continue_on_error: If True, continue pipeline even if a step fails
        python_path: Path to Python executable (optional)
        
    Returns:
        dict: Results summary with success/failure counts
    """
    logger.info("🚀 Memulai pipeline pembersihan dan validasi data...")
    logger.info(f"📋 Total steps: {len(PIPELINE_STEPS)}")
    
    results = {
        'total': len(PIPELINE_STEPS),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'steps': {}
    }
    
    for i, script_name in enumerate(PIPELINE_STEPS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Step {i}/{len(PIPELINE_STEPS)}: {script_name}")
        logger.info(f"{'='*60}")
        
        success = run_script(script_name, python_path)
        
        results['steps'][script_name] = 'success' if success else 'failed'
        
        if success:
            results['success'] += 1
        else:
            results['failed'] += 1
            if not continue_on_error:
                logger.error(f"\n⚠️  Pipeline stopped at step {i} due to error")
                logger.error(f"Use --continue-on-error flag to continue despite errors")
                results['skipped'] = len(PIPELINE_STEPS) - i
                break
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 Pipeline Summary")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful: {results['success']}/{results['total']}")
    logger.info(f"❌ Failed: {results['failed']}/{results['total']}")
    if results['skipped'] > 0:
        logger.info(f"⏭️  Skipped: {results['skipped']}/{results['total']}")
    
    if results['failed'] == 0:
        logger.info("\n🎉 Pipeline selesai dengan sukses!")
    else:
        logger.warning(f"\n⚠️  Pipeline selesai dengan {results['failed']} error(s)")
    
    return results

def list_steps():
    """List all pipeline steps."""
    logger.info("📋 Pipeline Steps:")
    for i, step in enumerate(PIPELINE_STEPS, 1):
        script_path = SCRIPTS_DIR / step
        exists = "✅" if script_path.exists() else "❌"
        logger.info(f"  {i}. {exists} {step}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data cleaning and validation pipeline"
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline even if a step fails'
    )
    parser.add_argument(
        '--python',
        type=str,
        help='Path to Python executable'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all pipeline steps'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_steps()
    elif args.dry_run:
        logger.info("🔍 Dry run mode - showing pipeline steps:")
        list_steps()
        logger.info("\nNo scripts will be executed.")
    else:
        results = run_pipeline(
            continue_on_error=args.continue_on_error,
            python_path=args.python
        )
        
        # Exit with error code if any step failed
        sys.exit(0 if results['failed'] == 0 else 1)