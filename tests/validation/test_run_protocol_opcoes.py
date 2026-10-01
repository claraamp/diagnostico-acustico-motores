"""
Combinações de opções que o run_protocol recusa antes de ler qualquer dado.

Rodam o script num subprocesso: o argparse sai com código 2 e a mensagem no
stderr, sem precisar de data/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "validation" / "run_protocol.py"


def _rodar(*opcoes):
    return subprocess.run([sys.executable, str(SCRIPT), *opcoes], cwd=RAIZ,
                          capture_output=True, text=True)


@pytest.mark.parametrize("opcoes, mensagem", [
    (["--protocolo", "B", "--fs", "25600", "--aumento"], "--aumento só vale na taxa de trabalho"),
    (["--protocolo", "B", "--tarefa", "multiclasse"], "o Protocolo B é binário"),
])
def test_opcoes_incompativeis_param_logo(opcoes, mensagem):
    r = _rodar(*opcoes)
    assert r.returncode == 2
    assert mensagem in r.stderr
