#!/usr/bin/env python3
"""
Visualize NLU test results from CSV.
Generates multiple charts for analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

# Set style
sns.set_style("whitegrid")
rcParams['figure.figsize'] = (16, 12)
rcParams['font.size'] = 10


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load test results CSV."""
    df = pd.read_csv(csv_path)
    return df


def plot_overall_metrics(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot overall success rate and confidence."""
    metrics = {
        'Success Rate': (df['success'].sum() / len(df)) * 100,
        'Avg Confidence': (df['confidence'].mean() * 100),
        'Failed Tests': (1 - df['success'].sum() / len(df)) * 100,
    }
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Overall Test Metrics', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)


def plot_confidence_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot confidence score distribution."""
    ax.hist(df['confidence'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(df['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["confidence"].mean():.3f}')
    ax.set_xlabel('Confidence Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Confidence Score Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_method_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot method used distribution."""
    method_counts = df['method_used'].value_counts()
    colors = plt.cm.Set3(range(len(method_counts)))
    wedges, texts, autotexts = ax.pie(
        method_counts.values,
        labels=method_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontweight': 'bold'}
    )
    ax.set_title('Method Distribution', fontweight='bold', fontsize=12)


def plot_processing_time(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot processing time distribution."""
    ax.boxplot(
        [df[df['success']]['processing_time_ms'].dropna(),
         df[~df['success']]['processing_time_ms'].dropna()],
        labels=['Successful', 'Failed'],
        patch_artist=True,
        boxprops=dict(facecolor='#3498db', alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    ax.set_ylabel('Processing Time (ms)', fontweight='bold')
    ax.set_title('Processing Time: Successful vs Failed', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)


def plot_success_by_intent(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot success rate by intent."""
    intent_success = df.groupby('expected_intent').apply(
        lambda x: (x['success'].sum() / len(x) * 100) if len(x) > 0 else 0
    ).sort_values(ascending=False)
    
    # Filter out None/NaN intents and empty
    intent_success = intent_success[intent_success.index.notna() & (intent_success.index != '')]
    
    colors = ['#2ecc71' if x >= 80 else '#f39c12' if x >= 50 else '#e74c3c' for x in intent_success.values]
    bars = ax.barh(range(len(intent_success)), intent_success.values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(intent_success)))
    ax.set_yticklabels(intent_success.index, fontsize=9)
    ax.set_xlabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Success Rate by Intent', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 110)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, intent_success.values)):
        ax.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=9)
    
    ax.grid(axis='x', alpha=0.3)


def plot_success_vs_failed(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot success vs failed test counts."""
    counts = df['success'].value_counts()
    labels = ['Successful', 'Failed']
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(labels, [counts.get(True, 0), counts.get(False, 0)], color=colors, alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Test Results: Success vs Failed', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)


def plot_failed_tests_detail(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Show details of failed tests."""
    failed = df[~df['success']].copy()
    
    if len(failed) == 0:
        ax.text(0.5, 0.5, 'All tests passed! ✅', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Create a table of failed tests
    failed_display = failed[['text', 'expected_intent', 'predicted_intent', 'confidence', 'method_used']].copy()
    failed_display['confidence'] = failed_display['confidence'].apply(lambda x: f'{x:.3f}')
    failed_display.columns = ['Query', 'Expected', 'Got', 'Conf', 'Method']
    
    ax.axis('off')
    table = ax.table(
        cellText=failed_display.head(10).values,
        colLabels=failed_display.columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.1, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(failed_display.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, min(11, len(failed_display) + 1)):
        for j in range(len(failed_display.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax.set_title(f'Failed Tests (showing up to 10 of {len(failed)})', 
                 fontweight='bold', fontsize=12, pad=20)


def create_visualizations(csv_path: Path, output_dir: Path | None = None) -> None:
    """Create all visualizations."""
    df = load_results(csv_path)
    
    if output_dir is None:
        output_dir = csv_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    
    # Main title
    fig.suptitle('NLU Test Results Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Overall metrics (top left, spanning 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_overall_metrics(df, ax1)
    
    # Plot 2: Success vs Failed (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_success_vs_failed(df, ax2)
    
    # Plot 3: Confidence distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_confidence_distribution(df, ax3)
    
    # Plot 4: Method distribution (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_method_distribution(df, ax4)
    
    # Plot 5: Processing time (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_processing_time(df, ax5)
    
    # Plot 6: Success by intent (bottom, spanning all cols)
    ax6 = fig.add_subplot(gs[2, :2])
    plot_success_by_intent(df, ax6)
    
    # Plot 7: Failed tests detail (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    plot_failed_tests_detail(df, ax7)
    
    # Save figure
    output_file = output_dir / f"test_results_visualization_bertt.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_file}")
    
    # Generate summary statistics
    print("\n" + "="*70)
    print("TEST SUMMARY STATISTICS")
    print("="*70)
    print(f"Total Tests: {len(df)}")
    print(f"Successful: {df['success'].sum()} ({df['success'].sum()/len(df)*100:.1f}%)")
    print(f"Failed: {(~df['success']).sum()} ({(~df['success']).sum()/len(df)*100:.1f}%)")
    print(f"Average Confidence: {df['confidence'].mean():.4f}")
    print(f"Average Processing Time: {df['processing_time_ms'].mean():.2f}ms")
    print(f"Min Processing Time: {df['processing_time_ms'].min():.2f}ms")
    print(f"Max Processing Time: {df['processing_time_ms'].max():.2f}ms")
    
    print("\n" + "="*70)
    print("SUCCESS RATE BY INTENT")
    print("="*70)
    intent_success = df.groupby('expected_intent').apply(
        lambda x: (x['success'].sum() / len(x) * 100) if len(x) > 0 else 0
    ).sort_values(ascending=False)
    
    intent_success = intent_success[intent_success.index.notna() & (intent_success.index != '')]
    for intent, rate in intent_success.items():
        count = len(df[df['expected_intent'] == intent])
        print(f"  {intent:30s}: {rate:6.1f}% ({count:2d} tests)")
    
    print("\n" + "="*70)
    print("METHOD DISTRIBUTION")
    print("="*70)
    for method, count in df['method_used'].value_counts().items():
        print(f"  {method:25s}: {count:3d} tests ({count/len(df)*100:.1f}%)")
    
    if (~df['success']).sum() > 0:
        print("\n" + "="*70)
        print("FAILED TESTS")
        print("="*70)
        failed = df[~df['success']].copy()
        for idx, row in failed.iterrows():
            print(f"\n  Query: {row['text']}")
            print(f"  Expected: {row['expected_intent']}")
            print(f"  Got: {row['predicted_intent']} (conf: {row['confidence']:.3f})")
            print(f"  Method: {row['method_used']}")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        # Find the latest test results file
        logs_dir = Path(__file__).resolve().parents[1] / "logs"
        csv_files = list(logs_dir.glob("test_results_*.csv"))
        
        if not csv_files:
            print("❌ No test results found in logs/")
            sys.exit(1)
        
        csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    print(f"📊 Loading results from: {csv_path}")
    create_visualizations(csv_path)
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
