"""Docker entrypoint: configures the proxy from env vars and starts granian."""

import os

from olist_code.logging_setup import build_log_config, granian_log_level, resolve_log_level
from olist_code.models import AdapterConfig, ModelConfig
from olist_code.server import set_app_config

config = AdapterConfig(
    base_url=os.environ["GATEWAY_URL"],
    api_key=os.environ.get("GATEWAY_KEY", ""),
    models=ModelConfig(
        opus=os.environ["MODEL"],
        sonnet=os.environ.get("MODEL_SONNET") or None,
        haiku=os.environ.get("MODEL_HAIKU") or None,
    ),
    port=int(os.environ.get("PORT", "3080")),
    harness="claude",
)

set_app_config(config)

if __name__ == "__main__":
    from granian import Granian
    from granian.constants import Interfaces
    from granian.log import LogLevels

    log_level = resolve_log_level(None, debug=False)
    Granian(
        "olist_code.server:app",
        address="0.0.0.0",
        port=config.port,
        workers=1,
        interface=Interfaces.ASGI,
        log_level=LogLevels(granian_log_level(log_level)),
        log_dictconfig=build_log_config(log_level),
        log_access=False,
    ).serve()
