"""
Testes da tabela do Protocolo B com e sem aumento (run_tabela_aumento.py).

Usam `metrics.json` mínimos em tmp_path; não dependem de data/. Conferem que
(1) a tabela recusa rodadas não comparáveis, (2) o fator de expansão efetivo
desconta as variantes recusadas e (3) várias sementes viram média ± desvio.
"""

from __future__ import annotations

import json

import pytest

from validation import run_tabela_aumento as T

BASE = {"protocolo": "B", "tarefa": "binario", "modelo": "lda", "splits": "abc",
        "mfcc_janela_ms": 25, "mfcc_hop_ms": 10, "mfcc_n_mels": 20, "mfcc_n_coefs": 13,
        "permutado": False, "sem_c0": False, "norm_clipe": False}
AUM = {"aumento": "deslocamento+estiramento+ruido", "aumento_copias": 4,
       "aumento_modo_estir": "tempo", "aumento_desloc_max_s": 0.4,
       "aumento_estir_taxas": "0.95-1.05", "aumento_snr_db": "20.0-35.0"}


def _rodada(pasta, exp_id, parametros, ab_bpfo03=0.5, aceitas=(896, 890)):
    falhas = {"bpfi_0.3mm": 1.0, "bpfo_0.3mm": ab_bpfo03}
    media = sum(falhas.values()) / len(falhas)
    m = {
        "parametros": parametros,
        "resumo": {
            "sensibilidade_media": 2 * media - 1, "especificidade_media": 1.0,
            "acuracia_balanceada_media": media,
            "pior_falha": "bpfo_0.3mm", "pior_falha_acuracia_balanceada": ab_bpfo03,
            "pior_fold": "B_bpfo_0.3mm_bloco0", "pior_fold_acuracia_balanceada": ab_bpfo03,
            "por_falha": {k: {"acuracia_balanceada": v} for k, v in falhas.items()},
        },
        "folds": [{"n_treino": 224, "n_treino_aumento": a} for a in aceitas],
    }
    d = pasta / f"{exp_id}_protocoloB"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")


@pytest.fixture
def pasta(tmp_path):
    _rodada(tmp_path, "exp001", {**BASE, "aumento": "nenhum"}, aceitas=(0, 0))
    _rodada(tmp_path, "exp002", {**BASE, **AUM})                              # semente do config
    _rodada(tmp_path, "exp003", {**BASE, **AUM, "aumento_semente": 1}, ab_bpfo03=0.6)
    _rodada(tmp_path, "exp004", {**BASE, **AUM, "aumento": "ruido"})
    _rodada(tmp_path, "exp005", {**BASE, **AUM, "permutado": True})
    return tmp_path


def test_tabela_com_sementes_e_fator_efetivo(pasta):
    texto = T.montar("exp001", ["exp002", "exp003"], ["exp004"], pasta)
    assert "três técnicas (2 sementes)" in texto
    assert "0.775 ± 0.035" in texto                 # médias 0,75 e 0,80
    assert "0.500 (bpfo_0.3mm)" in texto            # pior caso entre as sementes
    fator = (224 + 896) / 224 / 2 + (224 + 890) / 224 / 2
    assert f"5 / {fator:.2f}" in texto
    assert "890–896" in texto
    assert "só ruído" in texto


@pytest.mark.parametrize("ref, completo, ablacoes", [
    ("exp002", ["exp002"], []),            # referência com aumento
    ("exp001", ["exp002", "exp004"], []),  # completo com parâmetros de aumento diferentes
    ("exp001", ["exp002", "exp002"], []),  # semente repetida
    ("exp001", ["exp002"], ["exp005"]),    # controle permutado
])
def test_recusa_rodadas_nao_comparaveis(pasta, ref, completo, ablacoes):
    with pytest.raises(SystemExit):
        T.montar(ref, completo, ablacoes, pasta)


def test_recusa_particao_diferente(pasta):
    _rodada(pasta, "exp006", {**BASE, **AUM, "splits": "outro", "aumento_semente": 2})
    with pytest.raises(SystemExit):
        T.montar("exp001", ["exp002", "exp006"], [], pasta)
