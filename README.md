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
olist-code   # login e escolha de modelo automáticos na primeira vez
```

Pronto: `claude` e `opencode` já apontam pro gateway, como sempre foi.

### Usando em paralelo com a sua conta Anthropic

O `olist-code` escreve no `~/.claude/settings.json`, então *todo* `claude` vai pro gateway — inclusive os que não passam pelo terminal (extensão de IDE, app desktop, `claude -p` em script). O preço é não conseguir usar a sua conta Anthropic ao mesmo tempo.

Se você tem conta própria e quer as duas, suba o proxy em modo standalone:

```bash
olist-code standalone   # não toca no settings.json global
olist-code claude       # abre o Claude Code no gateway
olist-code opencode
```

Nesse modo a config do gateway vai só no processo que o `olist-code` abre, e `claude` puro continua na sua conta. Subir standalone também desfaz o que um `olist-code` comum tinha escrito, e vice-versa — dá pra alternar à vontade.

Tudo depois do nome do harness vai direto pra ele: `olist-code claude --resume`.

| Comando | Descrição |
|---|---|
| `olist-code` | Start the proxy: logs in and picks a model automatically the first time, then reuses the saved config. |
| `olist-code standalone` | Same, but leave the global Claude Code settings alone so a plain `claude` keeps using your own account. |
| `olist-code claude` | Open Claude Code against the proxy. Any extra args are passed straight to `claude`. |
| `olist-code opencode` | Open opencode against the proxy. Any extra args are passed straight to `opencode`. |
| `olist-code login` | Sign in with the backoffice SSO; the token is used as the gateway API key. |
| `olist-code logout` | Remove the stored SSO tokens. |
| `olist-code config` | Print the saved olist-code config (base URL, port, models, masked API key). |
| `olist-code restore` | Undo everything olist-code wrote to Claude Code / opencode settings and remove the local config. |
| `olist-code version` | Print the installed olist-code version. |

Use `olist-code --help` para a lista completa.

## Logs e depuração

O proxy escreve no terminal e em `~/.olist-code-adapter/logs/olist-code.log` (rotaciona a cada 5 MB, guarda 3 arquivos antigos). Cada linha tem horário, nível e um id curto da requisição (`[req=ab12cd]`), pra juntar tudo que aconteceu numa mesma chamada.

Por padrão sai uma linha por requisição: endpoint, modelo pedido → modelo no gateway, se foi streaming, status final, duração e tokens. Pra ver mais (ou menos):

```bash
olist-code --debug                 # atalho pra --log-level debug
olist-code standalone --log-level warning
OLIST_CODE_LOG_LEVEL=debug olist-code
```

A flag tem prioridade sobre a variável de ambiente; sem nenhuma, o nível é `info`. No Docker, use `OLIST_CODE_LOG_LEVEL`; lá o log vai só pro terminal (sem arquivo).

Quando algo dá errado, o log mostra o motivo: status e corpo do erro do gateway (truncado), timeouts e falhas de conexão (tipo do erro e tempo decorrido), erros no meio do streaming, respostas cortadas por filtro de conteúdo, e o traceback de erros internos. Junto vai o formato da requisição — quantidade de mensagens, tipos de bloco por mensagem, número de tools, `max_tokens`, tamanho do corpo.

O proxy nunca registra o conteúdo dos prompts, as entradas de ferramentas, headers ou tokens de acesso. A única coisa copiada do gateway é a mensagem de erro que ele devolve, truncada.
