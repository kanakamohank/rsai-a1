#!/usr/bin/env python3
"""
Run Full UAP Pipeline

This script runs the complete UAP analysis pipeline:
1. Compute UAP (or use existing)
2. Run comprehensive analysis
3. Generate all visualizations and reports
"""
import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"\n✗ Error: Command failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ Completed in {elapsed_time:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run Full UAP Pipeline')
    parser.add_argument('--skip-uap', action='store_true',
                       help='Skip UAP computation (use existing)')
    parser.add_argument('--uap-path', type=str, default='./results/uap_final.pt',
                       help='Path to UAP perturbation')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: Use reduced samples for testing')
    parser.add_argument('--output-dir', type=str, default='./results/analysis',
                       help='Output directory for analysis')

    args = parser.parse_args()

    print("=" * 80)
    print("UNIVERSAL ADVERSARIAL PERTURBATIONS - FULL PIPELINE")
    print("=" * 80)
    print(f"Mode: {'Quick (Testing)' if args.quick else 'Full (Production)'}")
    print(f"UAP Path: {args.uap_path}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # Find Python executable
    python_exe = sys.executable

    # Step 1: Compute UAP (unless skipped)
    if not args.skip_uap:
        print("\n" + "=" * 80)
        print("STEP 1: COMPUTING UNIVERSAL ADVERSARIAL PERTURBATION")
        print("=" * 80)

        uap_cmd = [python_exe, 'main.py', '--compute-uap']

        if Path(args.uap_path).exists():
            print(f"\n⚠ UAP already exists at {args.uap_path}")
            response = input("Do you want to recompute? (y/N): ")
            if response.lower() != 'y':
                print("Skipping UAP computation...")
            else:
                run_command(uap_cmd, "Computing UAP...")
        else:
            run_command(uap_cmd, "Computing UAP...")

        print(f"\n✓ UAP saved to {args.uap_path}")
    else:
        print("\n⚡ Skipping UAP computation (using existing)")
        if not Path(args.uap_path).exists():
            print(f"✗ Error: UAP not found at {args.uap_path}")
            sys.exit(1)

    # Step 2: Run comprehensive analysis
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    analysis_cmd = [
        python_exe, 'analyze_uap.py',
        '--uap-path', args.uap_path,
        '--output-dir', args.output_dir
    ]

    if args.quick:
        analysis_cmd.append('--quick')

    run_command(analysis_cmd, "Running analysis...")

    # Step 3: Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  1. uap_perturbation.png          - UAP visualization")
    print("  2. gradient_correlation.png      - Gradient correlation matrix")
    print("  3. pca_variance.png              - PCA variance explained")
    print("  4. analysis_results.json         - All numerical results")
    print("\nNext steps:")
    print("  - Review visualizations in the output directory")
    print("  - Check analysis_results.json for detailed metrics")
    print("  - Use results to answer the two critical questions")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
