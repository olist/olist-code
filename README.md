# olist-code

Proxy local que faz o Claude Code e o opencode apontarem para o Olist AI Gateway, sem precisar de Python, `pip` ou `uv` instalados na máquina.

## Instalação

**macOS / Linux**

```bash
curl -fsSL https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.sh | bash
```

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.ps1 | iex
```

Os scripts baixam o binário da [release](https://github.com/olist/olist-code/releases) mais recente pro seu SO.

## Uso

```bash
olist-code   # login e escolha de modelo automáticos na primeira vez; nas próximas, sobe o proxy direto
```

| Comando | Descrição |
|---|---|
| `olist-code` | Start the proxy: logs in and picks a model automatically the first time, then reuses the saved config. |
| `olist-code login` | Sign in with the backoffice SSO; the token is used as the gateway API key. |
| `olist-code logout` | Remove the stored SSO tokens. |
| `olist-code config` | Print the saved olist-code config (base URL, port, models, masked API key). |
| `olist-code restore` | Undo everything olist-code wrote to Claude Code / opencode settings and remove the local config. |
| `olist-code version` | Print the installed olist-code version. |

Use `olist-code --help` para a lista completa.
