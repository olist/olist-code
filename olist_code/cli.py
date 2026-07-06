"""CLI entry point for Olist Code Adapter using Typer."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated, Literal

import typer
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .auth import AuthError, clear_tokens, load_tokens
from .auth import login as sso_login
from .config import (
    CONFIG_FILE,
    load_config,
    save_config,
    update_claude_json,
    update_claude_settings,
)
from .models import AdapterConfig, ModelConfig, SSOConfig
from .proxy import fetch_gateway_models

console = Console()

_OLIST_ART: list[str] = [
    "   ____  ___      __ ",
    "  / __ \\/ (_)____/ /_",
    " / / / / / / ___/ __/",
    "/ /_/ / / (__  ) /_  ",
    "\\____/_/_/____/\\__/  ",
    "                     ",
]

_CODE_ART: list[str] = [
    "   ______          __   ",
    "  / ____/___  ____/ /__ ",
    " / /   / __ \\/ __  / _ \\",
    "/ /___/ /_/ / /_/ /  __/",
    "\\____/\\____/\\__,_/\\___/ ",
    "                        ",
]


def _print_banner() -> None:
    from . import __version__

    olist_w = max(len(line) for line in _OLIST_ART)

    text = Text()
    for i, (ol, cl) in enumerate(zip(_OLIST_ART, _CODE_ART)):
        text.append(ol.ljust(olist_w), style="bold blue")
        text.append("  ")
        text.append(cl, style="bold")
        if i < len(_OLIST_ART) - 1:
            text.append("\n")

    console.print()
    console.print(Align.center(text))
    console.print(Align.center(Text(f"v{__version__}", style="dim")))
    console.print()


cli = typer.Typer(
    name="olist-code",
    help="Olist Code Client — Proxy that translates Anthropic Messages API to OpenAI Chat Completions",
    add_completion=False,
    no_args_is_help=True,
)


def _pick_model(config: AdapterConfig) -> str | None:
    try:
        models = asyncio.run(fetch_gateway_models(config))
    except Exception as exc:
        console.print(f"[yellow]Não foi possível buscar modelos do gateway:[/yellow] {exc}")
        return None

    if not models:
        console.print("[yellow]Nenhum modelo ativo encontrado no gateway.[/yellow]")
        return None

    table = Table(title="Modelos disponíveis no gateway", show_lines=False)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Modelo", style="bold")
    table.add_column("Provider", style="dim")

    for i, m in enumerate(models, 1):
        table.add_row(str(i), m.get("id", "?"), m.get("owned_by", ""))

    console.print(table)
    console.print()

    max_idx = len(models)
    while True:
        choice: str = typer.prompt(f"Selecione um modelo [1-{max_idx}]", default="1")
        if choice.strip().isdigit() and 1 <= int(choice.strip()) <= max_idx:
            break
        console.print(f"[red]Escolha um número entre 1 e {max_idx}.[/red]")

    selected = models[int(choice.strip()) - 1]
    model_name = selected.get("id", "")
    console.print(f"[green]Modelo selecionado:[/green] [bold]{model_name}[/bold]")
    return model_name


@cli.command()
def run(
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", "-u", help="Upstream API base URL (e.g. https://api.openai.com)"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", "-k", help="Upstream API key"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Local proxy port"),
    ] = None,
    tool_format: Annotated[
        str | None,
        typer.Option("--tool-format", help="Tool format: native or xml"),
    ] = None,
    model_opus: Annotated[
        str | None,
        typer.Option("--model-opus", help="Opus model name"),
    ] = None,
    model_sonnet: Annotated[
        str | None,
        typer.Option("--model-sonnet", help="Sonnet model name"),
    ] = None,
    model_haiku: Annotated[
        str | None,
        typer.Option("--model-haiku", help="Haiku model name"),
    ] = None,
    save: Annotated[
        bool,
        typer.Option("--save", "-s", help="Save config to disk for future runs"),
    ] = False,
    do_login: Annotated[
        bool,
        typer.Option("--login", "-l", help="Run the backoffice SSO login before starting"),
    ] = False,
):
    existing = load_config()

    if do_login:
        sso = (existing.sso if existing else None) or SSOConfig()
        console.print(f"Abrindo o navegador para autenticar em [bold]{sso.issuer}[/bold]...")
        try:
            sso_login(sso)
            console.print("[green]Login concluído.[/green]\n")
        except AuthError as exc:
            console.print(f"[red]Login falhou:[/red] {exc}")
            raise typer.Exit(code=1)

    resolved_tool_format: Literal["native", "xml"] = (
        "native" if (tool_format or (existing.tool_format if existing else "native")) == "native" else "xml"
    )

    config = AdapterConfig(
        base_url=base_url or (existing.base_url if existing else ""),
        api_key=api_key if api_key is not None else (existing.api_key if existing else ""),
        sso=existing.sso if existing else None,
        models=ModelConfig(
            opus=model_opus or (existing.models.opus if existing else ""),
            sonnet=model_sonnet or (existing.models.sonnet if existing else None),
            haiku=model_haiku or (existing.models.haiku if existing else None),
        ),
        tool_format=resolved_tool_format,
        port=port or (existing.port if existing else 3080),
    )

    errors: list[str] = []
    if not config.base_url:
        errors.append("base_url is required (use --base-url or save a config first)")
    if not config.api_key and load_tokens() is None:
        errors.append(
            "no credential found: use --api-key, or run [bold]olist-code login[/bold] to sign in with the backoffice SSO"
        )

    if errors:
        for err in errors:
            console.print(f"[red]Error:[/red] {err}")
        raise typer.Exit(code=1)

    if model_opus is None:
        picked = _pick_model(config)
        if picked:
            config.models.opus = picked
        elif not config.models.opus:
            console.print("[red]Opus model is required (use --model-opus)[/red]")
            raise typer.Exit(code=1)

    if save:
        save_config(config)
        console.print(f"[green]Config saved to {CONFIG_FILE}[/green]")

    update_claude_settings(config)
    update_claude_json()

    from .server import set_app_config

    set_app_config(config)

    console.print()
    console.print(
        Panel.fit(
            "[bold]Olist Code Adapter[/bold] starting...\n\n"
            + f"  Base URL:  {config.base_url}\n"
            + f"  Port:      {config.port}\n"
            + f"  Tool fmt:  {config.tool_format}\n"
            + f"  Models:    opus={config.models.opus}, "
            + f"sonnet={config.models.sonnet or '—'}, "
            + f"haiku={config.models.haiku or '—'}",
            title="[bold green]Proxy[/bold green]",
            border_style="green",
        )
    )
    console.print()

    _start_granian(config)


def _start_granian(config: AdapterConfig) -> None:
    try:
        from granian import Granian
        from granian.constants import Interfaces
        from granian.log import LogLevels
    except ImportError as exc:
        console.print(f"[red]Granian not installed:[/red] {exc}")
        console.print("Install it with: [bold]pip install granian[/bold] or [bold]uv add granian[/bold]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Serving on http://0.0.0.0:{config.port}[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    console.print()

    server = Granian(
        "olist_code.server:app",
        address="0.0.0.0",
        port=config.port,
        workers=1,
        interface=Interfaces.ASGI,
        log_level=LogLevels.info,
        log_access=True,
    )
    server.serve()


@cli.command()
def login(
    issuer: Annotated[
        str | None,
        typer.Option("--issuer", help="Keycloak realm URL"),
    ] = None,
    client_id: Annotated[
        str | None,
        typer.Option("--client-id", help="Keycloak public client id"),
    ] = None,
    callback_port: Annotated[
        int | None,
        typer.Option("--callback-port", help="Local port for the SSO callback"),
    ] = None,
):
    """Sign in with the backoffice SSO; the token is used as the gateway API key."""
    existing = load_config()
    base_sso = (existing.sso if existing else None) or SSOConfig()
    sso = SSOConfig(
        issuer=issuer or base_sso.issuer,
        client_id=client_id or base_sso.client_id,
        callback_port=callback_port or base_sso.callback_port,
    )

    console.print(f"Abrindo o navegador para autenticar em [bold]{sso.issuer}[/bold]...")
    try:
        sso_login(sso)
    except AuthError as exc:
        console.print(f"[red]Login falhou:[/red] {exc}")
        raise typer.Exit(code=1)

    if existing is not None:
        existing.sso = sso
        save_config(existing)

    console.print(
        Panel.fit(
            "[green]Login concluído![/green]\n\n"
            + "O token do SSO será usado como credencial nas chamadas ao gateway\n"
            + "e renovado automaticamente enquanto a sessão durar.",
            title="[bold green]SSO[/bold green]",
            border_style="green",
        )
    )


@cli.command()
def logout():
    """Remove the stored SSO tokens."""
    if clear_tokens():
        console.print("[green]Tokens removidos.[/green]")
    else:
        console.print("[yellow]Nenhuma sessão SSO encontrada.[/yellow]")


@cli.command()
def config_show():
    cfg = load_config()
    if cfg is None:
        console.print("[yellow]No configuration found.[/yellow]")
        console.print(f"  Config file: {CONFIG_FILE}")
        console.print("  Run [bold]olist-code-adapter init[/bold] to create one.")
        return

    console.print(
        Panel.fit(
            f"[bold]base_url:[/bold]  {cfg.base_url}\n"
            + f"[bold]api_key:[/bold]   {'*' * 8}{cfg.api_key[-4:] if len(cfg.api_key) > 4 else '****'}\n"
            + f"[bold]port:[/bold]      {cfg.port}\n"
            + f"[bold]tool_format:[/bold] {cfg.tool_format}\n"
            + f"[bold]models:[/bold]    opus={cfg.models.opus}, "
            + f"sonnet={cfg.models.sonnet or '—'}, "
            + f"haiku={cfg.models.haiku or '—'}",
            title="[bold blue]Saved Config[/bold blue]",
            border_style="blue",
        )
    )


@cli.command()
def config_reset():
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        console.print(f"[green]Config removed: {CONFIG_FILE}[/green]")
    else:
        console.print("[yellow]No config file found.[/yellow]")


@cli.command()
def init():
    console.print("[bold]Olist Code Adapter[/bold] — Configuration Wizard\n")

    while True:
        base_url: str = typer.prompt("Upstream API base URL", default="https://api.openai.com")
        if base_url.strip():
            break
        console.print("[red]Base URL is required.[/red]")

    api_key: str = typer.prompt(
        "API key (deixe em branco para usar o SSO do backoffice)",
        hide_input=True,
        default="",
        show_default=False,
    )

    while True:
        model_opus: str = typer.prompt("Opus model name", default="claude-sonnet-4-20250514")
        if model_opus.strip():
            break
        console.print("[red]Opus model name is required.[/red]")

    model_sonnet: str | None = typer.prompt("Sonnet model name (leave blank to skip)", default="") or None
    model_haiku: str | None = typer.prompt("Haiku model name (leave blank to skip)", default="") or None

    while True:
        port_str: str = typer.prompt("Proxy port", default="3080")
        if port_str.strip().isdigit():
            break
        console.print("[red]Port must be a number.[/red]")

    config = AdapterConfig(
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        sso=SSOConfig() if not api_key.strip() else None,
        models=ModelConfig(
            opus=model_opus.strip(),
            sonnet=model_sonnet.strip() if model_sonnet else None,
            haiku=model_haiku.strip() if model_haiku else None,
        ),
        port=int(port_str.strip()),
    )

    save_config(config)
    update_claude_settings(config)
    update_claude_json()

    next_steps = "  Run [bold]olist-code-adapter run[/bold] to start the proxy."
    if not config.api_key:
        next_steps = (
            "  Run [bold]olist-code login[/bold] to sign in with the backoffice SSO,\n"
            + "  then [bold]olist-code-adapter run[/bold] to start the proxy."
        )

    console.print()
    console.print(
        Panel.fit(
            "[green]Configuration saved![/green]\n\n"
            + f"  Config:  {CONFIG_FILE}\n"
            + f"  Port:    {config.port}\n\n"
            + next_steps,
            title="[bold green]Done[/bold green]",
            border_style="green",
        )
    )


@cli.command()
def version():
    from . import __version__

    console.print(f"[bold]Olist Code Adapter[/bold] v{__version__}")


def main() -> None:
    show_banner = len(sys.argv) <= 1 or (len(sys.argv) > 1 and sys.argv[1] in ("run", "--help", "-h"))
    if show_banner:
        _print_banner()
    cli()


if __name__ == "__main__":
    main()
