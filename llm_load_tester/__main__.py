#!/usr/bin/env python3
"""Main entry point for the benchmark CLI."""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .cli import (
    run_interactive_config,
    prompt_export_results,
    get_export_filename,
    console
)
from .benchmarker import LLMBenchmarker, LoadTestConfig
from .modalities import get_handler
from .modelarts_monitoring import ModelArtsCloudEyeConfig


def main() -> int:
    """Main entry point."""
    try:
        # Run interactive configuration
        config_dict = run_interactive_config()
        
        # Get modality handler
        modality_handler = get_handler(config_dict["modality"])
        
        # Build load test configuration
        cloud_eye_settings = config_dict.get("modelarts_cloud_eye")
        cloud_eye_config = (
            ModelArtsCloudEyeConfig(**cloud_eye_settings)
            if cloud_eye_settings
            else None
        )
        load_config = LoadTestConfig(
            endpoint=config_dict["endpoint"],
            api_route=config_dict["api_route"],
            api_key=config_dict.get("api_key"),
            model=config_dict["model"],
            concurrency=config_dict["concurrency"],
            target_rps=config_dict["target_rps"],
            total_requests=config_dict["total_requests"],
            warmup_requests=config_dict["warmup_requests"],
            modality_handler=modality_handler,
            modality_config=config_dict["modality_config"],
            metrics_url=config_dict.get("metrics_url"),
            metrics_scrape_interval_seconds=config_dict.get(
                "metrics_scrape_interval_seconds", 1.0
            ),
            metrics_api_key=config_dict.get("metrics_api_key"),
            modelarts_cloud_eye=cloud_eye_config,
        )
        
        # Run the benchmark
        benchmarker = LLMBenchmarker(load_config)
        result = asyncio.run(benchmarker.run())
        
        # Display results
        result.print_rich_table()
        
        # Handle export
        should_export, export_format = prompt_export_results()
        
        if should_export and export_format:
            filename = get_export_filename(export_format)
            filepath = Path(filename)
            
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if export_format == "json":
                    result.export_json(filepath)
                else:
                    result.export_csv(filepath)
                
                console.print(f"[green]✓ Results exported to: {filepath.absolute()}[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to export results: {e}[/red]")
        
        console.print()
        console.print(Panel(
            "[green]Benchmark complete![/green]",
            border_style="green"
        ))
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
