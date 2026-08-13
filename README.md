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

Num terminal, sobe o proxy e deixa rodando:

```bash
olist-code server   # login e escolha de modelo automáticos na primeira vez
```

Pronto: `claude` e `opencode` já apontam pro gateway, como sempre foi.

### Usando em paralelo com a sua conta Anthropic

Por padrão o `olist-code` escreve no `~/.claude/settings.json`, então *todo* `claude` vai pro gateway — inclusive os que não passam pelo terminal (extensão de IDE, app desktop, `claude -p` em script). O preço é não conseguir usar a sua conta Anthropic ao mesmo tempo.

Se você tem conta própria e quer as duas:

```bash
olist-code server --isolated   # não toca no settings.json global
olist-code claude              # abre o Claude Code no gateway
olist-code opencode
```

Nesse modo a config do gateway vai só no processo que o `olist-code` abre, e `claude` puro continua na sua conta. A escolha fica salva na config — você só passa a flag uma vez (`--no-isolated` volta atrás).

Tudo depois do nome do harness vai direto pra ele: `olist-code claude --resume`.

| Comando | Descrição |
|---|---|
| `olist-code server` | Start the proxy: logs in and picks a model automatically the first time, then reuses the saved config. Também é o que `olist-code` sozinho faz. |
| `olist-code server --isolated` | Same, but leave the global Claude Code settings alone so a plain `claude` keeps using your own account. |
| `olist-code claude` | Open Claude Code against the proxy. Any extra args are passed straight to `claude`. |
| `olist-code opencode` | Open opencode against the proxy. Any extra args are passed straight to `opencode`. |
| `olist-code login` | Sign in with the backoffice SSO; the token is used as the gateway API key. |
| `olist-code logout` | Remove the stored SSO tokens. |
| `olist-code config` | Print the saved olist-code config (base URL, port, models, masked API key). |
| `olist-code restore` | Undo everything olist-code wrote to Claude Code / opencode settings and remove the local config. |
| `olist-code version` | Print the installed olist-code version. |

Use `olist-code --help` para a lista completa.
