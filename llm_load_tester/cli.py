from __future__ import annotations

"""Interactive CLI interface for the benchmark tool."""

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm

from .modalities import get_handler


console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
DISPLAY_BENCHMARKS_DIR = Path("/") / PROJECT_ROOT.name / "benchmarks"


# Preset configurations
API_PRESETS = {
    "OpenAI Compatible (Local)": {
        "endpoint": "http://localhost:8000",
        "api_route": "v1/chat/completions"
    },
    "vLLM Default": {
        "endpoint": "http://localhost:8000",
        "api_route": "v1/chat/completions"
    },
    "Ollama": {
        "endpoint": "http://localhost:11434",
        "api_route": "v1/chat/completions"
    },
    "Moonshot AI": {
        "endpoint": "https://api.moonshot.ai",
        "api_route": "v1/chat/completions"
    },
    "Custom": None
}

PROVIDER_MODEL_HINTS = {
    "Moonshot AI": {
        "examples": [
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ],
        "default": "moonshot-v1-8k",
    }
}

TOKEN_PRESETS = [256, 512, 1024, 2048, 4096, 8192]

CONCURRENCY_PRESETS = [1, 2, 4, 8, 16, 32, 64]

RPS_PRESETS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

REQUEST_COUNT_PRESETS = [10, 50, 100, 200, 500, 1000]
API_ROUTE_PRESETS = ["v1/chat/completions", "v2/chat/completions", "custom"]


def print_header() -> None:
    """Print the application header."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Multi-Modal LLM Benchmark[/bold cyan]\n"
        "[dim]Asynchronous benchmarking tool for LLM APIs[/dim]",
        border_style="cyan"
    ))
    console.print()


def select_modality() -> str:
    """Prompt user to select test modality."""
    console.print("[bold]Select Test Modality:[/bold]")
    console.print("  1. Text - Text-based chat completions")
    console.print("  2. Image - Vision/multimodal with images")
    console.print("  3. Voice - Audio/voice inputs")
    console.print()
    
    valid_choices = {"1", "2", "3", "text", "image", "voice"}
    mapping = {"1": "text", "2": "image", "3": "voice"}
    
    while True:
        choice = Prompt.ask("Enter choice (1)", default="1").lower()
        if choice in valid_choices:
            return mapping.get(choice, choice)
        console.print("[red]Invalid choice. Please enter 1, 2, 3, text, image, or voice.[/red]")


def select_from_presets(
    title: str,
    presets: list[str] | dict[str, Any],
    allow_custom: bool = True,
    custom_prompt: str = "Enter custom value"
) -> str | dict[str, Any] | None:
    """Generic preset selector with custom option."""
    console.print(f"\n[bold]{title}[/bold]")
    
    if isinstance(presets, dict):
        items = list(presets.keys())
    else:
        items = list(presets)
    
    for i, item in enumerate(items, 1):
        if isinstance(presets, dict) and presets[item] is not None:
            details = presets[item]
            if isinstance(details, dict):
                detail_str = f" ({details.get('endpoint', '')})"
            else:
                detail_str = ""
            console.print(f"  {i}. {item}{detail_str}")
        else:
            console.print(f"  {i}. {item}")
    
    console.print()
    
    # Build choices
    num_choices = [str(i) for i in range(1, len(items) + 1)]
    name_choices = [item.lower() for item in items]
    choices = num_choices + name_choices
    
    valid_choices = set(str(i) for i in range(1, len(items) + 1)) | set(item.lower() for item in items)
    
    while True:
        choice = Prompt.ask("Enter choice (1)", default="1").lower()
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{len(items)} or the option name.[/red]")
    
    # Convert numeric choice to item name
    if choice.isdigit():
        selected = items[int(choice) - 1]
    else:
        selected = choice
        # Find the actual case-sensitive item
        for item in items:
            if item.lower() == selected:
                selected = item
                break
    
    # Handle dict presets
    if isinstance(presets, dict):
        value = presets.get(selected)
        if value is None:  # Custom selected
            return None
        return selected, value
    
    # Handle list presets
    if selected.lower() == "custom":
        return Prompt.ask(custom_prompt)
    
    return selected


def build_auth_headers(api_key: str | None) -> dict[str, str]:
    """Build HTTP headers for optional bearer authentication."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def mask_api_key(api_key: str | None) -> str:
    """Mask an API key while leaving the last 4 characters visible."""
    if not api_key:
        return "(none)"
    if len(api_key) <= 4:
        return "*" * len(api_key)
    return f"{'*' * (len(api_key) - 4)}{api_key[-4:]}"


async def validate_api_key(endpoint: str, api_key: str) -> tuple[bool, str]:
    """Validate bearer authentication against the models endpoint."""
    headers = build_auth_headers(api_key)
    models_url = f"{endpoint}/v1/models"

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(models_url, headers=headers) as response:
                if response.status == 200:
                    return True, "API key accepted"
                if response.status in (401, 403):
                    text = await response.text()
                    if "invalid authentication" in text.lower():
                        return False, "API key rejected: invalid authentication"
                    return False, f"API key rejected with status {response.status}"
                text = await response.text()
                return False, f"API key test returned status {response.status}: {text[:100]}"
    except Exception as e:
        return False, f"API key test failed: {str(e)[:100]}"


async def test_endpoint(
    endpoint: str,
    api_route: str,
    api_key: str | None = None
) -> tuple[bool, list[str] | None, str]:
    """Test connection to endpoint and fetch available models.
    
    Returns: (success, available_models, message)
    """
    url = f"{endpoint}/{api_route}"
    models_url = f"{endpoint}/v1/models"
    headers = build_auth_headers(api_key)
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Try to fetch models first - this validates the endpoint exists
            models = None
            try:
                async with session.get(models_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and isinstance(data["data"], list):
                            models = [m.get("id", m.get("name", "unknown")) for m in data["data"]]
                            return True, models, "Endpoint is reachable"
                    elif response.status in (401, 403):
                        return True, models, "Endpoint is reachable (authentication required for models list)"
            except Exception:
                pass  # Model endpoint might not exist
            
            # Test chat completions endpoint with a minimal request
            # Try with a common model name that's likely to exist
            test_models = ["gpt-3.5-turbo", "gpt-4", "default", "model"]
            last_error = ""
            
            for test_model in test_models:
                test_payload = {
                    "model": test_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1
                }
                
                try:
                    async with session.post(url, json=test_payload, headers=headers) as response:
                        if response.status == 200:
                            return True, models, "Endpoint is reachable"
                        elif response.status in (401, 403):
                            return True, models, "Endpoint is reachable (authentication required)"
                        elif response.status == 404:
                            # Model not found - try next model
                            text = await response.text()
                            if "does not exist" in text.lower() or "not found" in text.lower():
                                last_error = f"Model '{test_model}' not found on server"
                                continue
                            return True, models, "Endpoint is reachable (model may need to be specified)"
                        elif response.status == 422:
                            return True, models, "Endpoint is reachable (validation error - check parameters)"
                        else:
                            text = await response.text()
                            return False, models, f"Endpoint returned status {response.status}: {text[:100]}"
                except Exception as e:
                    return False, models, f"Connection failed: {str(e)[:100]}"
            
            # If all model tests failed with 404, the endpoint exists but needs correct model name
            if last_error:
                return True, models, f"Endpoint is reachable (server requires a valid model name)"
            
            return False, models, "Could not validate endpoint"
                
    except Exception as e:
        return False, None, f"Connection failed: {str(e)[:100]}"


def select_auth_mode() -> str | None:
    """Prompt for optional bearer authentication."""
    console.print("\n[bold]Authentication:[/bold]")
    console.print("  1. Continue with no bearer key")
    console.print("  2. Enter API key")
    console.print()

    valid_choices = {"1", "2"}
    while True:
        choice = Prompt.ask("Enter choice (1)", default="1")
        if choice in valid_choices:
            break
        console.print("[red]Invalid choice. Please enter 1 or 2.[/red]")

    if choice == "1":
        return None

    api_key = Prompt.ask("Enter API key", password=True).strip() or None
    if api_key:
        console.print(f"[green]Using bearer key ending in {mask_api_key(api_key)[-4:]}[/green]")
    return api_key


def select_api_route(default_route: str = "v1/chat/completions") -> str:
    """Prompt user to select an API route."""
    console.print("\n[bold]Select API Route:[/bold]")
    for i, route in enumerate(API_ROUTE_PRESETS, 1):
        if route == "custom":
            console.print(f"  {i}. Custom")
        else:
            console.print(f"  {i}. {route}")
    console.print()

    route_to_index = {route: str(i + 1) for i, route in enumerate(API_ROUTE_PRESETS)}
    default_choice = route_to_index.get(default_route, "1")
    valid_choices = set(route_to_index.values())

    while True:
        choice = Prompt.ask(f"Enter choice ({default_choice})", default=default_choice)
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{len(API_ROUTE_PRESETS)}.[/red]")

    selected = API_ROUTE_PRESETS[int(choice) - 1]
    if selected == "custom":
        return Prompt.ask("Enter custom API Route", default=default_route)
    return selected


def select_api_config() -> tuple[str, str, list[str] | None, str | None, str | None]:
    """Prompt user for API endpoint and route configuration.
    
    Returns: (endpoint, api_route, available_models, api_key, provider_name)
    """
    result = select_from_presets(
        "Select API Configuration Preset:",
        API_PRESETS,
        allow_custom=True
    )
    provider_name = None
    
    if result is None or isinstance(result, str):  # Custom selected
        console.print("\n[dim]Enter custom API configuration:[/dim]")
        endpoint = Prompt.ask(
            "Endpoint IP:Port",
            default="localhost:8000"
        )
        api_route = select_api_route("v1/chat/completions")
        # Ensure endpoint has http:// prefix
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
    else:
        name, config = result
        provider_name = name
        endpoint = config["endpoint"]
        api_route = select_api_route(config["api_route"])

    api_key = select_auth_mode()

    if api_key:
        console.print(f"\n[yellow]Testing API key {mask_api_key(api_key)}...[/yellow]")
        auth_ok, auth_message = asyncio.run(validate_api_key(endpoint, api_key))
        if auth_ok:
            console.print(f"[green]✓ {auth_message}[/green]")
        else:
            console.print(f"[red]✗ {auth_message}[/red]")
            console.print("[yellow]Warning: Continuing with the provided key anyway...[/yellow]")
    
    # Test the endpoint
    console.print(f"\n[yellow]Testing connection to {endpoint}...[/yellow]")
    success, models, message = asyncio.run(test_endpoint(endpoint, api_route, api_key))
    
    if success:
        console.print(f"[green]✓ {message}[/green]")
        if models:
            console.print(f"[green]✓ Found {len(models)} model(s)[/green]")
    else:
        console.print(f"[red]✗ {message}[/red]")
        console.print("[yellow]Warning: Endpoint test failed. Continuing anyway...[/yellow]")
    
    return endpoint, api_route, models, api_key, provider_name


def select_model(
    available_models: list[str] | None = None,
    provider_name: str | None = None
) -> str:
    """Prompt user to enter model name."""
    console.print("\n[bold]Enter Model Name:[/bold]")
    provider_hints = PROVIDER_MODEL_HINTS.get(provider_name or "")
    
    if available_models:
        console.print("\n[dim]Available models on this endpoint:[/dim]")
        display_models = available_models[:20]  # Show first 20
        for i, m in enumerate(display_models, 1):
            console.print(f"  {i}. {m}")
        if len(available_models) > 20:
            console.print(f"  ... and {len(available_models) - 20} more")
        console.print()
        console.print("[dim]Enter a number to select from above, or type a custom model name[/dim]")
        
        while True:
            choice = Prompt.ask("Model name or number", default="1")
            # Check if user entered a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    return available_models[idx]
                else:
                    console.print(f"[red]Invalid number. Please enter 1-{len(available_models)} or a model name.[/red]")
            else:
                # User entered a custom model name
                return choice
    else:
        if provider_hints:
            examples = ", ".join(provider_hints["examples"])
            console.print(f"[dim]Examples: {examples}[/dim]")
            default_model = provider_hints["default"]
        else:
            console.print("[dim]Examples: gpt-oss, meta-llama/Llama-2-7b-chat-hf, mistral-7b-instruct[/dim]")
            default_model = "gpt-oss"
        console.print()
        return Prompt.ask("Model name", default=default_model)


def select_token_length(prompt_text: str) -> int:
    """Prompt user to select token length with presets."""
    console.print(f"\n[bold]{prompt_text}[/bold]")
    
    for i, preset in enumerate(TOKEN_PRESETS, 1):
        console.print(f"  {i}. {preset} tokens")
    console.print(f"  {len(TOKEN_PRESETS) + 1}. Custom")
    console.print()
    
    max_choice = len(TOKEN_PRESETS) + 1
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (1)", default="1")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    if int(choice) == len(TOKEN_PRESETS) + 1:
        return IntPrompt.ask("Enter custom token count", default=512)
    
    return TOKEN_PRESETS[int(choice) - 1]


def select_concurrency() -> int:
    """Prompt user for concurrency level."""
    console.print("\n[bold]Select Concurrency Level:[/bold]")
    console.print("[dim](Number of simultaneous connections)[/dim]")
    
    for i, preset in enumerate(CONCURRENCY_PRESETS, 1):
        console.print(f"  {i}. {preset} concurrent")
    console.print(f"  {len(CONCURRENCY_PRESETS) + 1}. Custom")
    console.print()
    
    max_choice = len(CONCURRENCY_PRESETS) + 1
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (4)", default="4")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    if int(choice) == len(CONCURRENCY_PRESETS) + 1:
        return IntPrompt.ask("Enter custom concurrency", default=4)
    
    return CONCURRENCY_PRESETS[int(choice) - 1]


def select_rps() -> float:
    """Prompt user for target RPS."""
    console.print("\n[bold]Select Target RPS:[/bold]")
    console.print("[dim](Requests Per Second - use 0 for unlimited)[/dim]")
    
    for i, preset in enumerate(RPS_PRESETS, 1):
        console.print(f"  {i}. {preset} RPS")
    console.print(f"  {len(RPS_PRESETS) + 1}. Custom")
    console.print(f"  {len(RPS_PRESETS) + 2}. Unlimited (0)")
    console.print()
    
    max_choice = len(RPS_PRESETS) + 2
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (5)", default="5")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    choice_int = int(choice)
    if choice_int == len(RPS_PRESETS) + 1:
        while True:
            custom_rps = FloatPrompt.ask("Enter custom RPS", default=1.0)
            if custom_rps >= 0:
                return custom_rps
            console.print("[red]RPS must be 0 or greater.[/red]")
    elif choice_int == len(RPS_PRESETS) + 2:
        return 0.0
    
    return RPS_PRESETS[choice_int - 1]


def select_request_count() -> int:
    """Prompt user for total request count."""
    console.print("\n[bold]Select Total Request Count:[/bold]")
    
    for i, preset in enumerate(REQUEST_COUNT_PRESETS, 1):
        console.print(f"  {i}. {preset} requests")
    console.print(f"  {len(REQUEST_COUNT_PRESETS) + 1}. Custom")
    console.print()
    
    max_choice = len(REQUEST_COUNT_PRESETS) + 1
    valid_choices = set(str(i) for i in range(1, max_choice + 1))
    
    while True:
        choice = Prompt.ask("Enter choice (3)", default="3")
        if choice in valid_choices:
            break
        console.print(f"[red]Invalid choice. Please enter 1-{max_choice}.[/red]")
    
    if int(choice) == len(REQUEST_COUNT_PRESETS) + 1:
        return IntPrompt.ask("Enter custom request count", default=100)
    
    return REQUEST_COUNT_PRESETS[int(choice) - 1]


def configure_text_modality(
    endpoint: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Configure text modality specific settings."""
    config = {"modality": "text"}
    
    console.print(Panel(
        "[cyan]Text Modality Configuration[/cyan]",
        border_style="cyan"
    ))
    
    config["input_tokens"] = select_token_length("Select Input Token Length:")

    is_modelarts_maas = "modelarts-maas.com" in endpoint
    if is_modelarts_maas:
        console.print("\n[bold]Output Token Limit Semantics:[/bold]")
        console.print("  1. max_completion_tokens - cap reasoning + answer")
        console.print("  2. max_tokens - cap visible answer only")
        token_limit_choice = Prompt.ask("Enter choice (1)", choices=["1", "2"], default="1")
        config["output_token_parameter"] = (
            "max_completion_tokens" if token_limit_choice == "1" else "max_tokens"
        )
        config["output_tokens"] = select_token_length(
            f"Select Output Token Length ({config['output_token_parameter']}):"
        )

        console.print("\n[bold]Thinking Mode:[/bold]")
        console.print("  1. Provider default")
        console.print("  2. Enabled")
        console.print("  3. Disabled")
        thinking_choice = Prompt.ask(
            "Enter choice (1)", choices=["1", "2", "3"], default="1"
        )
        config["thinking_mode"] = {
            "1": "provider_default",
            "2": "enabled",
            "3": "disabled",
        }[thinking_choice]
    else:
        config["output_token_parameter"] = "max_tokens"
        config["output_tokens"] = select_token_length(
            "Select Output Token Length (max_tokens):"
        )
    
    return config


def configure_image_modality() -> dict[str, Any]:
    """Configure image modality specific settings."""
    config = {"modality": "image"}
    
    console.print(Panel(
        "[cyan]Image Modality Configuration[/cyan]\n"
        "[dim]Images will be randomly selected from the specified directory[/dim]",
        border_style="cyan"
    ))
    
    # Get image directory
    image_dir = Prompt.ask(
        "Enter path to image directory",
        default="./images"
    )
    config["image_directory"] = image_dir
    
    # Get max tokens for response
    config["max_tokens"] = select_token_length("Select Max Output Tokens:")
    
    return config


def configure_voice_modality() -> dict[str, Any]:
    """Configure voice modality specific settings."""
    config = {"modality": "voice"}
    
    console.print(Panel(
        "[cyan]Voice Modality Configuration[/cyan]\n"
        "[dim]Audio files will be randomly selected from the specified directory[/dim]",
        border_style="cyan"
    ))
    
    # Get audio directory
    audio_directory = Prompt.ask(
        "Enter path to sound directory",
        default="./sound"
    )
    config["audio_directory"] = audio_directory
    
    # Get max tokens for response
    config["max_tokens"] = select_token_length("Select Max Output Tokens:")
    
    return config


def configure_server_metrics(
    endpoint: str,
    api_key: str | None,
) -> dict[str, Any]:
    """Configure optional Prometheus/vLLM scenario metrics collection."""
    enabled = Confirm.ask(
        "Collect server metrics from a Prometheus /metrics endpoint?",
        default=False,
    )
    if not enabled:
        return {
            "metrics_url": None,
            "metrics_scrape_interval_seconds": 1.0,
            "metrics_api_key": None,
        }

    default_url = f"{endpoint.rstrip('/')}/metrics"
    metrics_url = Prompt.ask("Metrics endpoint URL", default=default_url).strip()
    if not metrics_url.startswith(("http://", "https://")):
        metrics_url = f"http://{metrics_url}"

    while True:
        scrape_interval = FloatPrompt.ask(
            "Metrics scrape interval in seconds",
            default=1.0,
        )
        if scrape_interval >= 0.05:
            break
        console.print("[red]Scrape interval must be at least 0.05 seconds.[/red]")

    metrics_api_key = None
    if api_key and Confirm.ask(
        "Use the API bearer key for the metrics endpoint?",
        default=True,
    ):
        metrics_api_key = api_key

    return {
        "metrics_url": metrics_url,
        "metrics_scrape_interval_seconds": scrape_interval,
        "metrics_api_key": metrics_api_key,
    }


def configure_modelarts_cloud_eye(endpoint: str) -> dict[str, Any]:
    """Configure optional MaaS monitoring through Huawei Cloud Eye."""
    console.print(
        "\n[dim]ModelArts MaaS does not expose public /metrics. "
        "Its service metrics are queried separately from Cloud Eye.[/dim]"
    )
    enabled = Confirm.ask(
        "Collect aggregated MaaS metrics from Cloud Eye after the benchmark?",
        default=False,
    )
    if not enabled:
        return {"modelarts_cloud_eye": None}

    hostname = urlparse(endpoint).hostname or ""
    region = "ap-southeast-1"
    if hostname.startswith("api-") and ".modelarts-maas.com" in hostname:
        inferred_region = hostname[4:].split(".modelarts-maas.com", 1)[0]
        if inferred_region:
            region = inferred_region
    default_endpoint = f"https://ces.{region}.myhuaweicloud.com"
    cloud_eye_endpoint = Prompt.ask(
        "Cloud Eye API endpoint", default=default_endpoint
    ).strip()
    if not cloud_eye_endpoint.startswith(("http://", "https://")):
        cloud_eye_endpoint = f"https://{cloud_eye_endpoint}"
    project_id = Prompt.ask("Huawei Cloud project ID").strip()
    iam_token = Prompt.ask(
        "Temporary IAM token (not the MaaS inference API key)", password=True
    ).strip()

    console.print("\n[bold]Cloud Eye MaaS Dimension:[/bold]")
    console.print("  1. maas_api_id")
    console.print("  2. maas_service_name")
    dimension_choice = Prompt.ask("Enter choice (1)", choices=["1", "2"], default="1")
    dimension_name = "maas_api_id" if dimension_choice == "1" else "maas_service_name"
    dimension_value = Prompt.ask(f"{dimension_name} value").strip()
    dimensions: list[tuple[str, str]] = [(dimension_name, dimension_value)]
    if Confirm.ask("Filter metrics to a specific MaaS API key ID?", default=False):
        dimensions.append(("maas_key_id", Prompt.ask("maas_key_id value").strip()))

    wait_seconds = FloatPrompt.ask(
        "Wait for Cloud Eye ingestion before querying (seconds)", default=0.0
    )
    return {
        "modelarts_cloud_eye": {
            "endpoint": cloud_eye_endpoint,
            "project_id": project_id,
            "iam_token": iam_token,
            "dimensions": tuple(dimensions),
            "ingestion_wait_seconds": max(0.0, wait_seconds),
        }
    }


def prompt_export_results() -> tuple[bool, str | None]:
    """Prompt user if they want to export results."""
    console.print()
    
    should_export = Confirm.ask(
        "Would you like to export the results to a file? (y/n)",
        default=True
    )
    
    if not should_export:
        return False, None
    
    console.print("\n[bold]Select Export Format:[/bold]")
    console.print("  1. JSON (includes full raw metrics)")
    console.print("  2. Excel workbook (requests and monitoring in separate sheets)")
    console.print()
    
    valid_choices = {"1", "2", "json", "xlsx", "excel"}
    while True:
        format_choice = Prompt.ask("Enter choice (1)", default="1").lower()
        if format_choice in valid_choices:
            break
        console.print("[red]Invalid choice. Please enter 1, 2, json, xlsx, or excel.[/red]")
    
    if format_choice in ("1", "json"):
        return True, "json"
    else:
        return True, "xlsx"


def get_export_filename(format_type: str) -> str:
    """Prompt user for export filename."""
    from datetime import datetime
    
    default_name = f"llm_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
    default_display_path = DISPLAY_BENCHMARKS_DIR / default_name
    
    filename = Prompt.ask(
        "Enter filename for export",
        default=str(default_display_path)
    )

    if filename.startswith(f"{DISPLAY_BENCHMARKS_DIR}/"):
        relative_name = Path(filename).relative_to(DISPLAY_BENCHMARKS_DIR)
        filepath = DEFAULT_BENCHMARKS_DIR / relative_name
    elif filename == str(DISPLAY_BENCHMARKS_DIR):
        filepath = DEFAULT_BENCHMARKS_DIR / default_name
    else:
        filepath = Path(filename).expanduser()

    if not filepath.is_absolute():
        if filepath.parent == Path("."):
            filepath = DEFAULT_BENCHMARKS_DIR / filepath.name
        else:
            filepath = (PROJECT_ROOT / filepath).resolve()

    if filepath.suffix.lower() != f".{format_type}":
        filepath = filepath.with_suffix(f".{format_type}")

    return str(filepath)


def run_interactive_config() -> dict[str, Any]:
    """Run the full interactive configuration."""
    print_header()
    
    config = {}
    
    # 1. Select modality
    modality = select_modality()
    config["modality"] = modality
    
    # 2. API Configuration (with connection test)
    endpoint, api_route, available_models, api_key, provider_name = select_api_config()
    config["endpoint"] = endpoint
    config["api_route"] = api_route
    config["api_key"] = api_key

    # 2b. Provider-specific monitoring. MaaS uses Cloud Eye; self-hosted
    # engines can expose Prometheus-compatible server metrics directly.
    if "modelarts-maas.com" in endpoint:
        config.update({
            "metrics_url": None,
            "metrics_scrape_interval_seconds": 1.0,
            "metrics_api_key": None,
        })
        config.update(configure_modelarts_cloud_eye(endpoint))
    else:
        config.update(configure_server_metrics(endpoint, api_key))
        config["modelarts_cloud_eye"] = None
    
    # 3. Model selection (with available models if fetched)
    model = select_model(available_models, provider_name)
    config["model"] = model
    
    # 4. Modality-specific configuration
    console.print()
    if modality == "text":
        modality_config = configure_text_modality(endpoint, model)
    elif modality == "image":
        modality_config = configure_image_modality()
    elif modality == "voice":
        modality_config = configure_voice_modality()
    else:
        modality_config = {"modality": modality}
    
    config["modality_config"] = modality_config
    config["modality_config"]["model"] = model
    
    # 5. Load testing parameters
    console.print()
    console.print(Panel(
        "[cyan]Load Testing Parameters[/cyan]",
        border_style="cyan"
    ))
    
    config["concurrency"] = select_concurrency()
    config["target_rps"] = select_rps()
    config["total_requests"] = select_request_count()
    should_warmup = Confirm.ask(
        "Send warm-up requests before benchmark? (5 requests)",
        default=True
    )
    config["warmup_requests"] = 5 if should_warmup else 0

    return config
