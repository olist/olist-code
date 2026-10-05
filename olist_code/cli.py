"""CLI entry point for Olist Code Adapter using Typer."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
from typing import Annotated, Literal, cast

import httpx
import typer
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .auth import AuthError, clear_tokens, load_tokens
from .auth import login as sso_login
from .config import (
    CLAUDE_SETTINGS_FILE,
    CONFIG_FILE,
    OPENCODE_SETTINGS_FILE,
    claude_env,
    claude_model_picker,
    load_config,
    restore_claude_settings,
    restore_opencode_settings,
    save_config,
    update_claude_json,
    update_claude_settings,
    update_opencode_settings,
)
from .logging_setup import LOG_FILE, build_log_config, granian_log_level, resolve_log_level
from .models import AdapterConfig, ModelConfig, SSOConfig
from .proxy import fetch_gateway_models

console = Console()

GATEWAY_BASE_URL = "https://olist-ai-gateway-ai.prd.olistcloud.com/"

Harness = Literal["claude", "opencode", "both"]
_HARNESS_CHOICES: tuple[Harness, ...] = ("claude", "opencode", "both")


def _resolve_harness(explicit: str | None, existing: AdapterConfig | None) -> Harness:
    value = (explicit or (existing.harness if existing else "both")).strip().lower()
    if value not in _HARNESS_CHOICES:
        console.print(f"[red]--harness inválido: {value!r}. Use claude, opencode ou both.[/red]")
        raise typer.Exit(code=1)
    return cast(Harness, value)


def _prompt_harness() -> Harness:
    console.print("Qual harness você usa?")
    console.print("  [cyan]1[/cyan] - Claude Code")
    console.print("  [cyan]2[/cyan] - opencode")
    console.print("  [cyan]3[/cyan] - ambos (both)")
    while True:
        choice: str = typer.prompt("Escolha [1-3]", default="1")
        if choice.strip() in {"1", "2", "3"}:
            break
        console.print("[red]Escolha 1, 2 ou 3.[/red]")
    return cast(Harness, ("claude", "opencode", "both")[int(choice.strip()) - 1])


def _apply_settings(config: AdapterConfig, isolated: bool = False, model_ids: list[str] | None = None) -> None:
    if config.harness in ("claude", "both"):
        if isolated:
            # `olist-code claude` carries the config per-process; drop anything a
            # previous plain `olist-code` left behind so bare `claude` comes back.
            restore_claude_settings()
        else:
            update_claude_settings(config, model_ids)
        update_claude_json()
    if config.harness in ("opencode", "both"):
        update_opencode_settings(config)


def _gateway_model_ids(config: AdapterConfig) -> list[str]:
    try:
        models = asyncio.run(fetch_gateway_models(config))
    except Exception as exc:
        console.print(f"[yellow]Não foi possível buscar modelos do gateway:[/yellow] {exc}")
        return []
    return [m["id"] for m in models if m.get("id")]


def _claude_settings_args(model_ids: list[str]) -> list[str]:
    """`claude` flags that list the gateway models in /model for this session only."""
    if not model_ids:
        return []
    return ["--settings", json.dumps({"modelPicker": claude_model_picker(model_ids)})]


def _require_config() -> AdapterConfig:
    cfg = load_config()
    if cfg is None or not cfg.models.opus:
        console.print("[red]Nenhuma config encontrada.[/red]")
        console.print("Rode [bold]olist-code[/bold] uma vez para fazer login e escolher o modelo.")
        raise typer.Exit(code=1)
    return cfg


def _require_proxy(config: AdapterConfig) -> None:
    try:
        httpx.get(f"http://localhost:{config.port}/health", timeout=2.0).raise_for_status()
    except Exception:
        console.print(f"[red]O proxy não está respondendo em http://localhost:{config.port}[/red]")
        console.print(
            "Rode [bold]olist-code[/bold] (ou [bold]olist-code standalone[/bold]) em outro terminal primeiro."
        )
        raise typer.Exit(code=1)


# Vars PyInstaller rewrites so the frozen binary finds its own bundled libraries. It
# stashes whatever was there before in <VAR>_ORIG.
_FROZEN_LOADER_VARS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "LIBPATH")


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Our env plus `extra`, with PyInstaller's loader paths undone.

    Left in place, the child would look for its shared libraries inside our unpacked
    bundle instead of its own install.
    """
    env = {**os.environ, **(extra or {})}
    if not getattr(sys, "frozen", False):
        return env

    env.pop("_MEIPASS2", None)
    for var in _FROZEN_LOADER_VARS:
        original = env.pop(f"{var}_ORIG", None)
        if original:
            env[var] = original
        else:
            env.pop(var, None)
    return env


def _launch(command: str, args: list[str], env: dict[str, str] | None = None) -> None:
    """Run `command` in the foreground, then exit with its status code."""
    exe = shutil.which(command)
    if exe is None:
        console.print(f"[red]`{command}` não encontrado no PATH.[/red]")
        raise typer.Exit(code=1)

    proc = subprocess.Popen([exe, *args], env=_child_env(env))
    while True:
        try:
            code = proc.wait()
            break
        except KeyboardInterrupt:
            # Ctrl+C already reached the child via the shared process group; it decides
            # when to exit, and we stay around to report its status.
            continue
    raise typer.Exit(code=code)

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
    no_args_is_help=False,
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


_PORT_OPTION = Annotated[int | None, typer.Option("--port", "-p", help="Local proxy port")]
_HARNESS_OPTION = Annotated[
    str | None,
    typer.Option("--harness", help="Which harness to configure: claude, opencode, or both"),
]
_LOG_LEVEL_OPTION = Annotated[
    str | None,
    typer.Option("--log-level", help="debug, info, warning or error (default: $OLIST_CODE_LOG_LEVEL or info)"),
]
_DEBUG_OPTION = Annotated[bool, typer.Option("--debug", help="Shortcut for --log-level debug")]


def _resolve_log_level(log_level: str | None, debug: bool) -> int:
    try:
        return resolve_log_level(log_level, debug)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@cli.callback(invoke_without_command=True)
def main_default(
    ctx: typer.Context,
    port: _PORT_OPTION = None,
    harness: _HARNESS_OPTION = None,
    log_level: _LOG_LEVEL_OPTION = None,
    debug: _DEBUG_OPTION = False,
):
    """Start the proxy: logs in and picks a model automatically the first time, then reuses the saved config."""
    if ctx.invoked_subcommand is not None:
        return
    _run_server(port, harness, isolated=False, log_level=_resolve_log_level(log_level, debug))


@cli.command("standalone")
def standalone_command(
    port: _PORT_OPTION = None,
    harness: _HARNESS_OPTION = None,
    log_level: _LOG_LEVEL_OPTION = None,
    debug: _DEBUG_OPTION = False,
):
    """Start the proxy without touching the global Claude Code settings.

    A plain `claude` keeps using your own Anthropic account; reach the gateway with
    `olist-code claude` instead. Also undoes what a previous plain `olist-code` wrote.
    """
    _run_server(port, harness, isolated=True, log_level=_resolve_log_level(log_level, debug))


@cli.command(
    "claude",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def claude_command(ctx: typer.Context):
    """Open Claude Code against the proxy. Any extra args are passed straight to `claude`."""
    config = _require_config()
    _require_proxy(config)
    update_claude_json()
    _launch("claude", [*_claude_settings_args(_gateway_model_ids(config)), *ctx.args], claude_env(config))


@cli.command(
    "opencode",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def opencode_command(ctx: typer.Context):
    """Open opencode against the proxy. Any extra args are passed straight to `opencode`."""
    config = _require_config()
    _require_proxy(config)
    update_opencode_settings(config)
    _launch("opencode", ctx.args)


def _run_server(port: int | None, harness: str | None, isolated: bool, log_level: int = logging.INFO) -> None:
    existing = load_config()

    if not (existing and existing.api_key) and load_tokens() is None:
        sso = (existing.sso if existing else None) or SSOConfig()
        console.print(f"Abrindo o navegador para autenticar em [bold]{sso.issuer}[/bold]...")
        try:
            sso_login(sso)
            console.print("[green]Login concluído.[/green]\n")
        except AuthError as exc:
            console.print(f"[red]Login falhou:[/red] {exc}")
            raise typer.Exit(code=1)

    resolved_harness: Harness | None = None
    if harness is not None:
        resolved_harness = _resolve_harness(harness, existing)
    elif existing is not None:
        resolved_harness = _resolve_harness(None, existing)
    else:
        resolved_harness = _prompt_harness()

    config = AdapterConfig(
        base_url=GATEWAY_BASE_URL,
        api_key=existing.api_key if existing else "",
        sso=existing.sso if existing else None,
        models=ModelConfig(
            opus=existing.models.opus if existing else "",
            sonnet=existing.models.sonnet if existing else None,
            haiku=existing.models.haiku if existing else None,
        ),
        tool_format=existing.tool_format if existing else "native",
        port=port or (existing.port if existing else 3080),
        harness=resolved_harness,
    )

    _ensure_port_free(config.port)

    needs_save = existing is None

    if not config.models.opus:
        picked = _pick_model(config)
        if not picked:
            console.print("[red]Não foi possível selecionar um modelo.[/red]")
            raise typer.Exit(code=1)
        config.models.opus = picked
        needs_save = True

    if needs_save:
        save_config(config)
        console.print(f"[green]Config salva em {CONFIG_FILE}[/green]")

    model_ids = _gateway_model_ids(config) if config.harness in ("claude", "both") and not isolated else None
    _apply_settings(config, isolated, model_ids)

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
            + f"haiku={config.models.haiku or '—'}\n"
            + f"  Logs:      {LOG_FILE} ({logging.getLevelName(log_level).lower()})\n"
            + "  Claude:    "
            + (
                "standalone — abra com `olist-code claude`"
                if isolated
                else "settings.json global — `claude` já aponta pro gateway"
            ),
            title="[bold green]Proxy[/bold green]",
            border_style="green",
        )
    )
    console.print()

    _start_granian(config, log_level)


def _ensure_port_free(port: int) -> None:
    """Fail if something already listens on the port.

    Granian binds with SO_REUSEPORT, so a second server would silently share the
    port with the old one; probing without SO_REUSEPORT detects that.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            # Ignore leftover TIME_WAIT connections; still fails on an active listener.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            console.print(f"[red]A porta {port} já está em uso.[/red]")
            console.print("Encerre o outro proxy (ou processo) usando a porta, ou use [bold]--port[/bold].")
            raise typer.Exit(code=1)


def _start_granian(config: AdapterConfig, log_level: int = logging.INFO) -> None:
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
        log_level=LogLevels(granian_log_level(log_level)),
        log_dictconfig=build_log_config(log_level, LOG_FILE),
        log_access=False,
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
def config():
    """Print the saved olist-code config (base URL, port, models, masked API key)."""
    cfg = load_config()
    if cfg is None:
        console.print("[yellow]No configuration found.[/yellow]")
        console.print(f"  Config file: {CONFIG_FILE}")
        console.print("  Run [bold]olist-code[/bold] to create one.")
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
def restore():
    """Undo everything olist-code wrote: Claude Code / opencode settings, local config, and SSO session."""
    restore_claude_settings()
    restore_opencode_settings()

    removed_local_config = CONFIG_FILE.exists()
    if removed_local_config:
        CONFIG_FILE.unlink()

    removed_session = clear_tokens()

    console.print(
        Panel.fit(
            "[green]Configurações originais restauradas.[/green]\n\n"
            + f"  Claude Code:  {CLAUDE_SETTINGS_FILE}\n"
            + f"  opencode:     {OPENCODE_SETTINGS_FILE}\n"
            + f"  Local config: {'removido' if removed_local_config else 'não existia'}\n"
            + f"  Sessão SSO:   {'removida' if removed_session else 'não existia'}",
            title="[bold green]Restore[/bold green]",
            border_style="green",
        )
    )


@cli.command()
def version():
    """Print the installed olist-code version."""
    from . import __version__

    console.print(f"[bold]Olist Code Adapter[/bold] v{__version__}")


@cli.command()
def update():
    """Download and install the latest olist-code release, replacing this binary."""
    install_url = "https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.sh"
    console.print(f"Baixando o instalador de [bold]{install_url}[/bold]...")
    try:
        script = httpx.get(install_url, timeout=10.0, follow_redirects=True).raise_for_status().text
    except Exception as exc:
        console.print(f"[red]Falha ao baixar o instalador:[/red] {exc}")
        raise typer.Exit(code=1)

    result = subprocess.run(["bash", "-c", script])
    raise typer.Exit(code=result.returncode)


def main() -> None:
    show_banner = len(sys.argv) <= 1 or sys.argv[1] in ("--help", "-h", "standalone")
    if show_banner:
        _print_banner()
    cli()


if __name__ == "__main__":
    main()
