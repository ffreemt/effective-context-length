#!/usr/bin/env python3
"""
Effective Context Length CLI Tool

Main entry point for the effective context length testing tool.
Tests and determines the effective context length limits for OpenAI-compatible LLM endpoints.
"""

import argparse
import sys
import os
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path

# Add the package to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from effective_context_length import __version__, __description__
from rich.console import Console
from rich.table import Table
from rich.text import Text


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with all required argument groups."""
    
    parser = argparse.ArgumentParser(
        prog='effective-context-length',
        description=__description__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://api.openai.com/v1
  %(prog)s http://localhost:8000/v1 --model custom-model --max-tokens 128000
  %(prog)s https://api.example.com/v1 --strategy binary-search --output results.json
  %(prog)s https://api.openai.com/v1 --api-key sk-... --model gpt-4 --verbose
        """
    )
    
    # Version information
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'%(prog)s {__version__}'
    )
    
    # Positional arguments (now optional with default)
    parser.add_argument(
        'base_url', 
        nargs='?',
        default='http://127.0.0.1:3300/v1',
        help='Base URL of the OpenAI-compatible API endpoint (default: http://127.0.0.1:3300/v1)'
    )
    
    # Authentication argument group
    auth_group = parser.add_argument_group('Authentication')
    auth_group.add_argument(
        '--api-key', 
        default='sk-aMjTHX0fJGIC7jb34HRZSLC8QTtZACQw92VewXqJYiiP4Knb',
        help='API key for authentication (can also be set via environment variable)'
    )
    auth_group.add_argument(
        '--api-key-env', 
        default='OPENAI_API_KEY',
        help='Environment variable containing API key (default: OPENAI_API_KEY)'
    )
    
    # Model configuration argument group
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model', 
        default='v0-1.0-md',
        help='Model name to test (default: v0-1.0-md)'
    )
    model_group.add_argument(
        '--max-tokens', 
        type=int, 
        default=200000,
        help='Maximum context length to test (default: 200000)'
    )
    model_group.add_argument(
        '--min-tokens', 
        type=int, 
        default=1000,
        help='Minimum context length to start testing (default: 1000)'
    )
    
    # Testing strategy argument group
    strategy_group = parser.add_argument_group('Testing Strategy')
    strategy_group.add_argument(
        '--strategy', 
        choices=['gradual', 'binary-search', 'predefined', 'hybrid'],
        default='hybrid', 
        help='Testing strategy to use (default: hybrid)'
    )
    strategy_group.add_argument(
        '--step-size', 
        type=int, 
        default=2000,
        help='Step size for gradual strategy (default: 2000)'
    )
    strategy_group.add_argument(
        '--predefined-sizes', 
        nargs='+', 
        type=int,
        help='Predefined context sizes to test (space-separated list of integers)'
    )
    strategy_group.add_argument(
        '--samples-per-size', 
        type=int, 
        default=5,
        help='Number of test samples per context size (default: 5)'
    )
    
    # Request configuration argument group
    request_group = parser.add_argument_group('Request Configuration')
    request_group.add_argument(
        '--timeout', 
        type=int, 
        default=300,
        help='Request timeout in seconds (default: 300)'
    )
    request_group.add_argument(
        '--max-retries', 
        type=int, 
        default=3,
        help='Maximum number of retries per request (default: 3)'
    )
    request_group.add_argument(
        '--rate-limit', 
        type=float, 
        default=1.0,
        help='Rate limit in requests per second (default: 1.0)'
    )
    request_group.add_argument(
        '--temperature', 
        type=float, 
        default=0.1,
        help='Temperature for API requests (default: 0.1)'
    )
    
    # Output configuration argument group
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output', 
        '-o', 
        help='Output file path for results (if not specified, prints to stdout)'
    )
    output_group.add_argument(
        '--format', 
        choices=['json', 'csv', 'yaml', 'txt'],
        default='json', 
        help='Output format (default: json)'
    )
    output_group.add_argument(
        '--verbose', 
        '-v', 
        action='count', 
        default=0,
        help='Increase verbosity level (use -v, -vv, or -vvv for more detail)'
    )
    output_group.add_argument(
        '--quiet', 
        '-q', 
        action='store_true',
        help='Suppress progress output (only show final results)'
    )
    output_group.add_argument(
        '--no-progress', 
        action='store_true',
        help='Disable progress bar display'
    )
    
    # Analysis configuration argument group
    analysis_group = parser.add_argument_group('Analysis Configuration')
    analysis_group.add_argument(
        '--error-threshold', 
        type=float, 
        default=0.1,
        help='Error rate threshold for determining effective limit (default: 0.1)'
    )
    analysis_group.add_argument(
        '--confidence-level', 
        type=float, 
        default=0.95,
        help='Confidence level for statistical analysis (default: 0.95)'
    )
    analysis_group.add_argument(
        '--min-samples', 
        type=int, 
        default=3,
        help='Minimum number of samples required per context length (default: 3)'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate parsed arguments and check for logical consistency."""
    
    # Validate URL format
    if not args.base_url.startswith(('http://', 'https://')):
        raise ValueError("Base URL must start with http:// or https://")
    
    # Validate token range
    if args.min_tokens >= args.max_tokens:
        raise ValueError("min-tokens must be less than max-tokens")
    
    if args.min_tokens <= 0:
        raise ValueError("min-tokens must be positive")
    
    # Validate step size for gradual strategy
    if args.strategy == 'gradual' and args.step_size <= 0:
        raise ValueError("step-size must be positive for gradual strategy")
    
    # Validate predefined sizes
    if args.strategy == 'predefined':
        if not args.predefined_sizes:
            raise ValueError("predefined-sizes must be specified for predefined strategy")
        if any(size <= 0 for size in args.predefined_sizes):
            raise ValueError("All predefined sizes must be positive")
    
    # Validate samples per size
    if args.samples_per_size <= 0:
        raise ValueError("samples-per-size must be positive")
    
    # Validate timeout and retries
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    
    if args.max_retries < 0:
        raise ValueError("max-retries cannot be negative")
    
    # Validate rate limit
    if args.rate_limit <= 0:
        raise ValueError("rate-limit must be positive")
    
    # Validate temperature
    if not (0.0 <= args.temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    
    # Validate error threshold
    if not (0.0 <= args.error_threshold <= 1.0):
        raise ValueError("error-threshold must be between 0.0 and 1.0")
    
    # Validate confidence level
    if not (0.0 < args.confidence_level < 1.0):
        raise ValueError("confidence-level must be between 0.0 and 1.0 (exclusive)")
    
    # Validate min samples
    if args.min_samples <= 0:
        raise ValueError("min-samples must be positive")
    
    # Check for conflicting quiet/verbose options
    if args.quiet and args.verbose > 0:
        raise ValueError("Cannot use both --quiet and --verbose options")


def get_api_key(args: argparse.Namespace) -> Optional[str]:
    """Get API key from arguments or environment variables."""
    
    # First check command line argument
    if args.api_key:
        return args.api_key
    
    # Then check environment variable
    api_key = os.getenv(args.api_key_env)
    if api_key:
        return api_key
    
    # No API key found
    return None


async def run_context_length_test(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main function to run the context length test with full implementation.
    """
    from effective_context_length.tester import EffectiveContextLengthTester, TestStrategy
    from effective_context_length.analyzer import ResultsAnalyzer
    from effective_context_length.utils import setup_logging, ProgressReporter, Timer
    
    # Setup logging
    logger = setup_logging(args.verbose, args.quiet)
    
    print(f"🚀 Starting effective context length test...")
    print(f"📡 Endpoint: {args.base_url}")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Strategy: {args.strategy}")
    print(f"📏 Range: {args.min_tokens:,} - {args.max_tokens:,} tokens")
    
    # Get API key
    api_key = get_api_key(args)
    if not api_key:
        raise ValueError("No API key available")
    
    # Map strategy string to enum
    strategy_map = {
        'hybrid': TestStrategy.HYBRID,
        'gradual': TestStrategy.GRADUAL,
        'binary-search': TestStrategy.BINARY_SEARCH,
        'predefined': TestStrategy.PREDEFINED
    }
    strategy = strategy_map[args.strategy]
    
    # Configure tester
    config = {
        'min_tokens': args.min_tokens,
        'max_tokens': args.max_tokens,
        'step_size': args.step_size,
        'samples_per_size': args.samples_per_size,
        'error_threshold': args.error_threshold,
        'timeout': args.timeout,
        'rate_limit': args.rate_limit,
        'max_retries': args.max_retries,
        'temperature': args.temperature,
        'min_samples': args.min_samples,
        'predefined_sizes': args.predefined_sizes
    }
    
    # Progress tracking
    progress_reporter = None
    if not args.quiet and not args.no_progress:
        # Estimate total tests for progress tracking
        if args.strategy == 'predefined' and args.predefined_sizes:
            estimated_tests = len(args.predefined_sizes) * args.samples_per_size
        elif args.strategy == 'gradual':
            estimated_lengths = (args.max_tokens - args.min_tokens) // args.step_size + 1
            estimated_tests = estimated_lengths * args.samples_per_size
        else:
            # Conservative estimate for binary search and hybrid
            estimated_tests = 20 * args.samples_per_size
        
        progress_reporter = ProgressReporter(
            total=estimated_tests,
            description="Testing context lengths",
            quiet=args.quiet
        )
    
    def progress_callback(completed: int, total: int, message: str = ""):
        """Progress callback for the tester."""
        if progress_reporter:
            progress_reporter.current = completed
            progress_reporter.total = total
            progress_reporter.update(0, message)
    
    try:
        # Run the test
        with Timer("Context length test") as test_timer:
            async with EffectiveContextLengthTester(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                strategy=strategy,
                **config
            ) as tester:
                
                # Run the test
                session = await tester.run_test(progress_callback=progress_callback)
                
                # Analyze results
                analyzer = ResultsAnalyzer(confidence_level=args.confidence_level)
                analysis = analyzer.analyze_session(
                    session=session,
                    error_threshold=args.error_threshold,
                    min_samples=args.min_samples
                )
        
        if progress_reporter:
            progress_reporter.finish("Test completed successfully")
        
        # Print summary if not quiet
        if not args.quiet:
            print(f"\n✅ Test completed in {test_timer.duration:.1f} seconds")
            print(f"📊 Total requests: {session.total_requests}")
            print(f"✅ Successful: {session.successful_requests}")
            print(f"❌ Failed: {session.failed_requests}")
            
            if analysis.effective_context.effective_context_length:
                print(f"🎯 Effective context length: {analysis.effective_context.effective_context_length:,} tokens")
                if analysis.effective_context.recommended_safe_length:
                    print(f"🛡️  Recommended safe limit: {analysis.effective_context.recommended_safe_length:,} tokens")
            else:
                print("⚠️  Could not determine effective context length")
            
            # Show key recommendations
            if analysis.recommendations:
                print(f"\n💡 Key recommendations:")
                for i, rec in enumerate(analysis.recommendations[:3], 1):
                    print(f"   {i}. {rec}")
            
            # Display error rates by context length using rich formatting
            if analysis.errors.error_rate_by_context_length and not args.quiet:
                print()  # Add spacing
                console = Console()
                display_error_rate_table(analysis.errors.error_rate_by_context_length, console)
        
        # Export detailed results
        detailed_results = analyzer.export_detailed_results(analysis)
        
        # Add metadata
        detailed_results['metadata'] = {
            'tool_version': __version__,
            'test_timestamp': session.start_time.isoformat(),
            'base_url': args.base_url,
            'model': args.model,
            'strategy': args.strategy,
            'total_duration_seconds': test_timer.duration
        }
        
        return detailed_results
        
    except Exception as e:
        if progress_reporter:
            progress_reporter.finish(f"Test failed: {str(e)}")
        
        logger.error(f"Test failed: {str(e)}")
        
        # Return error result
        return {
            'metadata': {
                'tool_version': __version__,
                'base_url': args.base_url,
                'model': args.model,
                'strategy': args.strategy,
                'status': 'failed',
                'error': str(e)
            },
            'configuration': config,
            'results': {
                'error': str(e),
                'message': 'Test failed - see error details above'
            }
        }


def format_output(results: Dict[str, Any], format_type: str) -> str:
    """Format results according to the specified output format."""
    
    if format_type == 'json':
        import json
        return json.dumps(results, indent=2, default=str)
    
    elif format_type == 'yaml':
        try:
            import yaml
            return yaml.dump(results, default_flow_style=False, indent=2)
        except ImportError:
            print("Warning: PyYAML not installed, falling back to JSON format", file=sys.stderr)
            import json
            return json.dumps(results, indent=2, default=str)
    
    elif format_type == 'csv':
        # CSV format for detailed results
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Check if we have detailed analysis results
        if 'summary' in results:
            # Write summary first
            writer.writerow(['Metric', 'Value'])
            
            summary = results.get('summary', {})
            writer.writerow(['Strategy', summary.get('test_strategy', 'unknown')])
            writer.writerow(['Total Requests', summary.get('total_requests', 0)])
            writer.writerow(['Success Rate', f"{summary.get('overall_success_rate', 0):.2%}"])
            writer.writerow(['Effective Context Length', summary.get('effective_context_length', 'Not determined')])
            writer.writerow(['Recommended Safe Length', summary.get('recommended_safe_length', 'Not determined')])
            writer.writerow(['Average Response Time (ms)', f"{summary.get('average_response_time_ms', 0):.1f}"])
            writer.writerow(['Test Duration (s)', f"{summary.get('total_test_duration_seconds', 0):.1f}"])
            writer.writerow(['Error Rate', f"{summary.get('error_rate', 0):.2%}"])
            
            # Add recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                writer.writerow([])
                writer.writerow(['Recommendations'])
                for i, rec in enumerate(recommendations, 1):
                    writer.writerow([f'Recommendation {i}', rec])
        else:
            # Fallback for simple results
            writer.writerow(['Key', 'Value'])
            
            def write_dict(d, prefix=''):
                for key, value in d.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        write_dict(value, full_key)
                    else:
                        writer.writerow([full_key, str(value)])
            
            write_dict(results)
        
        return output.getvalue()
    
    elif format_type == 'txt':
        # Enhanced text format for comprehensive results
        output = []
        output.append("Effective Context Length Analysis Report")
        output.append("=" * 50)
        
        # Metadata
        metadata = results.get('metadata', {})
        output.append(f"Tool Version: {metadata.get('tool_version', 'unknown')}")
        output.append(f"Test Date: {metadata.get('test_timestamp', 'unknown')}")
        output.append(f"Endpoint: {metadata.get('base_url', 'unknown')}")
        output.append(f"Model: {metadata.get('model', 'unknown')}")
        output.append(f"Strategy: {metadata.get('strategy', 'unknown')}")
        output.append(f"Duration: {metadata.get('total_duration_seconds', 0):.1f} seconds")
        output.append("")
        
        # Summary
        summary = results.get('summary', {})
        if summary:
            output.append("Test Summary:")
            output.append("-" * 20)
            output.append(f"Total Requests: {summary.get('total_requests', 0):,}")
            output.append(f"Successful Requests: {summary.get('successful_requests', 0):,}")
            output.append(f"Failed Requests: {summary.get('failed_requests', 0):,}")
            output.append(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.1%}")
            output.append("")
            
            # Effective context length
            effective_length = summary.get('effective_context_length')
            if effective_length:
                output.append(f"🎯 Effective Context Length: {effective_length:,} tokens")
                safe_length = summary.get('recommended_safe_length')
                if safe_length:
                    output.append(f"🛡️  Recommended Safe Limit: {safe_length:,} tokens")
            else:
                output.append("⚠️  Effective context length could not be determined")
            output.append("")
            
            # Performance
            avg_time = summary.get('average_response_time_ms', 0)
            if avg_time > 0:
                output.append(f"⏱️  Average Response Time: {avg_time:.0f}ms")
            
            error_rate = summary.get('error_rate', 0)
            if error_rate > 0:
                output.append(f"❌ Error Rate: {error_rate:.1%}")
            output.append("")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            output.append("Recommendations:")
            output.append("-" * 20)
            for i, rec in enumerate(recommendations, 1):
                output.append(f"{i}. {rec}")
            output.append("")
        
        # Fallback for simple results
        if not summary and 'results' in results:
            output.append("Results:")
            output.append("-" * 20)
            results_data = results['results']
            if isinstance(results_data, dict):
                for key, value in results_data.items():
                    output.append(f"{key.replace('_', ' ').title()}: {value}")
            else:
                output.append(str(results_data))
        
        return "\n".join(output)
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def display_error_rate_table(error_rate_by_context_length: Dict[int, float], console: Console = None) -> None:
    """
    Display error rates by context length using rich formatting.
    
    Args:
        error_rate_by_context_length: Dictionary mapping context lengths to error rates
        console: Rich console instance (creates new one if None)
    """
    if console is None:
        console = Console()
    
    if not error_rate_by_context_length:
        console.print("[yellow]No error rate data available[/yellow]")
        return
    
    # Create table
    table = Table(title="Error Rates by Context Length")
    
    # Add columns
    table.add_column("Context Length", justify="right", style="cyan")
    table.add_column("Error Rate", justify="center", style="magenta")
    table.add_column("Status", justify="center")
    
    # Sort by context length
    sorted_lengths = sorted(error_rate_by_context_length.keys())
    
    for length in sorted_lengths:
        error_rate = error_rate_by_context_length[length]
        error_rate_percent = error_rate * 100
        
        # Format error rate with color coding
        if error_rate == 0:
            error_rate_text = f"[green]{error_rate_percent:.1f}%[/green]"
            status = "[green]✓ Perfect[/green]"
        elif error_rate < 0.05:  # < 5%
            error_rate_text = f"[green]{error_rate_percent:.1f}%[/green]"
            status = "[green]✓ Good[/green]"
        elif error_rate < 0.1:  # < 10%
            error_rate_text = f"[yellow]{error_rate_percent:.1f}%[/yellow]"
            status = "[yellow]⚠ Acceptable[/yellow]"
        elif error_rate < 0.2:  # < 20%
            error_rate_text = f"[orange_red1]{error_rate_percent:.1f}%[/orange_red1]"
            status = "[orange_red1]⚠ High[/orange_red1]"
        else:  # >= 20%
            error_rate_text = f"[red]{error_rate_percent:.1f}%[/red]"
            status = "[red]✗ Critical[/red]"
        
        table.add_row(
            f"{length:,}",
            error_rate_text,
            status
        )
    
    console.print(table)


def main() -> int:
    """Main entry point for the CLI application."""
    
    try:
        # Create and parse arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Validate arguments
        validate_arguments(args)
        
        # Get API key
        api_key = get_api_key(args)
        if not api_key:
            print(f"Error: No API key found. Please provide --api-key or set {args.api_key_env} environment variable.", 
                  file=sys.stderr)
            return 1
        
        # Set verbosity level
        if args.verbose >= 3:
            print(f"Debug: Using API key from {'command line' if args.api_key else 'environment variable'}")
            print(f"Debug: All arguments: {vars(args)}")
        
        # Run the test
        results = asyncio.run(run_context_length_test(args))
        
        # Format output
        formatted_output = format_output(results, args.format)
        
        # Write output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            if not args.quiet:
                print(f"Results written to: {output_path}")
        else:
            print(formatted_output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
    
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())