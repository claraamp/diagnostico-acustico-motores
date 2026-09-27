"""Garantias do registro de experimentos (scripts/experimentos.py)."""

import csv

import experimentos

LINHA = {c: "" for c in experimentos.REGISTRY_COLUMNS}


def _linha(exp_id: str, notas: str = "") -> dict:
    return {**LINHA, "id": exp_id, "notas": notas}


def test_append_depois_de_arquivo_sem_quebra_final(tmp_path):
    """Um registry salvo sem a quebra de linha final não pode engolir a próxima linha."""
    reg = tmp_path / "registry.csv"
    experimentos.append_registry(reg, [_linha("exp001", "primeira")])
    reg.write_bytes(reg.read_bytes().rstrip(b"\r\n"))        # como um editor faria
    experimentos.append_registry(reg, [_linha("exp002", "segunda")])

    with reg.open(encoding="utf-8", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    assert [l["id"] for l in linhas] == ["exp001", "exp002"]
    assert linhas[0]["notas"] == "primeira"
    assert experimentos.next_exp_number(reg) == 3


def test_append_normal_nao_cria_linha_vazia(tmp_path):
    reg = tmp_path / "registry.csv"
    experimentos.append_registry(reg, [_linha("exp001")])
    experimentos.append_registry(reg, [_linha("exp002")])
    texto = reg.read_bytes()
    assert b"\r\n\r\n" not in texto and b"\n\n" not in texto
    assert experimentos.next_exp_number(reg) == 3