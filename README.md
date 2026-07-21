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

Os scripts baixam o binário da [release](https://github.com/olist/olist-code/releases) mais recente. Como o repositório é privado, é preciso ter o `gh` CLI autenticado (`gh auth login`) ou definir a env var `GITHUB_TOKEN` com acesso de leitura ao repo.

## Uso

```bash
olist-code init   # configuração inicial (gateway, modelos, porta)
olist-code run    # sobe o proxy e aponta Claude Code / opencode para ele
```
