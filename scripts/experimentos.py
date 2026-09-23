"""
experimentos.py — registro estruturado das rodadas, em experiments/registry.csv.

Formato fixado na seção 4 do CONVENTIONS.md. Fica aqui, e não dentro de um
script de etapa, porque toda etapa que produz número comparável escreve nesse
arquivo: a comparação de taxas de decimação hoje, a extração de features e o
treino do classificador depois.

Três regras que o formato pressupõe, e que estas funções sustentam:

- **O script escreve a própria linha.** Preencher à mão depende de alguém
  lembrar. Daí `git_short_hash`, que lê o commit corrente sozinho: sem ele a
  rodada não é reproduzível, que é a razão de a coluna existir.
- **A numeração é contínua.** `next_exp_number` lê o arquivo e continua de onde
  parou, em vez de recomeçar e criar ids repetidos.
- **Uma varredura de parâmetro gera uma linha por ponto**, todas com o mesmo
  commit e a mesma data. É o que permite plotar métrica × parâmetro.

Rodadas de verificação — reexecutar para conferir um gráfico, testar o ambiente
— NÃO entram: use a flag que desliga o registro. Linha repetida não é registro
a mais, é comparação estragada.
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path


REGISTRY_COLUMNS = ["id", "data", "etapa", "script", "git_commit", "parametros",
                    "dataset", "metricas", "responsavel", "notas"]


def git_short_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "sem-git"
    except Exception:
        return "sem-git"


def next_exp_number(path: Path) -> int:
    """Continua a numeração sequencial do registry (exp001, exp002, ...)."""
    if not path.exists():
        return 1
    n = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            m = re.fullmatch(r"exp(\d+)", (row.get("id") or "").strip())
            if m:
                n = max(n, int(m.group(1)))
    return n + 1


def kv(pairs: dict) -> str:
    """Formato chave=valor;chave=valor exigido pelas colunas parametros/metricas."""
    return ";".join(f"{k}={v}" for k, v in pairs.items() if v is not None)


def append_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    novo = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        if novo:
            w.writeheader()
        w.writerows(rows)