import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent

# Use a consolidated pipeline implementation (new)
try:
    # Ensure project root is importable when run from anywhere
    sys.path.insert(0, str(SCRIPTS_DIR.parent))
    from scripts.dataset_pipeline import PipelinePaths, clean_dataset, deduplicate_patterns, split_patterns, validate_dataset
except SystemExit as e:
    # dataset_pipeline already prints a helpful message (e.g. missing pandas)
    raise
except Exception as e:
    logger.error(
        "❌ Failed to import dataset pipeline. "
        "Pastikan dependency terinstall (contoh: 'python3 -m pip install pandas') dan jalankan dari project root. "
        f"Detail: {e}"
    )
    raise


def _default_paths() -> PipelinePaths:
    return PipelinePaths.defaults()


def _run_step(label: str, fn) -> bool:
    logger.info(f"▶️  {label}")
    try:
        fn()
        logger.info(f"✅ Completed: {label}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {label} ({e})")
        return False

def run_pipeline(continue_on_error: bool = False) -> dict:
    """
    Run the complete data cleaning and validation pipeline.
    
    Args:
        continue_on_error: If True, continue pipeline even if a step fails
        python_path: Path to Python executable (optional)
        
    Returns:
        dict: Results summary with success/failure counts
    """
    paths = _default_paths()
    logger.info("🚀 Memulai pipeline pembersihan dan validasi data (ringkas)...")
    logger.info(f"📂 Input : {paths.input_raw}")
    logger.info(f"📄 Clean : {paths.output_clean}")
    logger.info(f"📄 Dedup : {paths.output_dedup}")
    logger.info(f"📄 Train : {paths.output_train}")

    steps = [
        ("Clean dataset", lambda: clean_dataset(paths.input_raw, paths.output_clean, convert_response_json=False)),
        ("Deduplicate patterns", lambda: deduplicate_patterns(paths.output_clean, paths.output_dedup)),
        ("Split patterns (training format)", lambda: split_patterns(paths.output_dedup, paths.output_train)),
        ("Validate final dataset", lambda: validate_dataset(paths.output_train, validate_response_json=False)),
    ]

    logger.info(f"📋 Total steps: {len(steps)}")
    
    results = {
        'total': len(steps),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'steps': {}
    }

    for i, (label, fn) in enumerate(steps, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Step {i}/{len(steps)}: {label}")
        logger.info(f"{'='*60}")

        success = _run_step(label, fn)

        results['steps'][label] = 'success' if success else 'failed'
        
        if success:
            results['success'] += 1
        else:
            results['failed'] += 1
            if not continue_on_error:
                logger.error(f"\n⚠️  Pipeline stopped at step {i} due to error")
                logger.error(f"Use --continue-on-error flag to continue despite errors")
                results['skipped'] = len(steps) - i
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
    logger.info("📋 Pipeline Steps (ringkas):")
    for i, label in enumerate([
        "Clean dataset",
        "Deduplicate patterns",
        "Split patterns (training format)",
        "Validate final dataset",
    ], 1):
        logger.info(f"  {i}. ✅ {label}")

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
            continue_on_error=args.continue_on_error
        )
        
        # Exit with error code if any step failed
        sys.exit(0 if results['failed'] == 0 else 1)