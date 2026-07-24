# olist-code

Proxy local que faz o Claude Code e o opencode apontarem para o Olist AI Gateway, sem precisar de Python, `pip` ou `uv` instalados na máquina.

## Instalação

Como o repositório é privado, é preciso ter o [`gh` CLI](https://cli.github.com) instalado e autenticado (`gh auth login`) **ou** uma env var `GITHUB_TOKEN` com acesso de leitura ao repo. Isso vale mesmo pra quem usa SSH no dia a dia com git — a autenticação do `gh`/`GITHUB_TOKEN` é separada da chave SSH, e é ela que dá acesso pra baixar os scripts e os binários das releases.

**macOS / Linux**

```bash
curl -fsSL -H "Authorization: token $(gh auth token)" \
  https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.sh | bash
```

**Windows (PowerShell)**

```powershell
$token = gh auth token
irm -Headers @{ Authorization = "token $token" } `
  https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.ps1 | iex
```

Sem `gh`, dá pra usar `GITHUB_TOKEN` no lugar de `$(gh auth token)` / `gh auth token`.

Os scripts baixam o binário da [release](https://github.com/olist/olist-code/releases) mais recente pro seu SO.

## Uso

```bash
olist-code init   # configuração inicial (gateway, modelos, porta)
olist-code run    # sobe o proxy e aponta Claude Code / opencode para ele
```
