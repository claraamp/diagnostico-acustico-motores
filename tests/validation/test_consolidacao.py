"""
Testes da consolidação dos resultados (run_consolidacao.py).

Usam `metrics.json` e um `registry.csv` mínimos em tmp_path; não dependem de
data/. Conferem as garantias que tornam as tabelas do relatório confiáveis:
(1) número sem linha no registry, ou com valor diferente, aborta; (2) rodada no
papel errado aborta; (3) a matriz de confusão soma os folds com a contagem certa;
(4) a comparação de taxas aceita outra partição, mas não outro modelo nem
features de outro commit.
"""

from __future__ import annotations

import csv
import json

import pytest

from validation import run_consolidacao as C

BASE = {"tarefa": "binario", "modelo": "lda", "splits": "abc",
        "features": "mfcc_dsp_media_desvio", "fs_hz": 12800, "segmento_s": 1.0,
        "segmentos_por_bloco": 10, "segmentos_descarte": 1, "mfcc_janela_ms": 25,
        "mfcc_hop_ms": 10, "mfcc_n_mels": 20, "mfcc_n_coefs": 13,
        "permutado": False, "sem_c0": False, "norm_clipe": False, "aumento": "nenhum"}
BLOCOS = (0, 1)
N_FALHA, N_NORMAL = 59, 10


def _rodada_b(sens_por_falha: dict[str, float]) -> dict:
    folds = []
    for f in C.FALHAS:
        for b in BLOCOS:
            s = sens_por_falha[f]
            folds.append({"nome": f"B_{f}_bloco{b}", "info": {"falha_de_fora": f, "bloco_normal": b},
                          "n_treino": 224, "n_teste": N_FALHA + N_NORMAL, "sensibilidade": s,
                          "especificidade": 1.0, "acuracia_balanceada": (s + 1) / 2,
                          "n_teste_falha": N_FALHA, "n_teste_normal": N_NORMAL})
    por = {f: {"sensibilidade": s, "especificidade": 1.0, "acuracia_balanceada": (s + 1) / 2}
           for f, s in sens_por_falha.items()}
    media = sum(p["acuracia_balanceada"] for p in por.values()) / 4
    pior = min(por, key=lambda f: por[f]["acuracia_balanceada"])
    return {"resumo": {
        "protocolo": "B", "acuracia_balanceada_media": media,
        "sensibilidade_media": sum(sens_por_falha.values()) / 4, "especificidade_media": 1.0,
        "pior_falha": pior, "pior_falha_acuracia_balanceada": por[pior]["acuracia_balanceada"],
        "pior_fold": f"B_{pior}_bloco0", "pior_fold_acuracia_balanceada": por[pior]["acuracia_balanceada"],
        "por_falha": por}, "folds": folds}


def _rodada_a(multiclasse: bool = False, ab: float = 1.0) -> dict:
    folds = []
    for b in BLOCOS:
        f = {"nome": f"A_bloco{b}", "info": {"bloco": b}, "n_treino": 240, "n_teste": 50,
             "acuracia_balanceada": ab}
        if multiclasse:
            f["recall_por_classe"] = {c: ab for c in C.CLASSES}
        else:
            f.update(sensibilidade=ab, especificidade=ab, n_teste_falha=40, n_teste_normal=10)
        folds.append(f)
    return {"resumo": {"protocolo": "A", "acuracia_balanceada_media": ab,
                       "acuracia_balanceada_desvio": 0.0, "acuracia_balanceada_min": ab},
            "folds": folds}


def _registrar(pasta, registry_rows, exp_id, parametros, corpo, nome="x"):
    d = pasta / f"{exp_id}_{nome}"
    d.mkdir()
    corpo = {**corpo, "id": exp_id, "parametros": parametros}
    (d / "metrics.json").write_text(json.dumps(corpo), encoding="utf-8")
    r = corpo["resumo"]
    mapa = C.METRICAS_B if parametros["protocolo"] == "B" else C.METRICAS_A
    metricas = ";".join(f"{k}={float(C._pegar(r, cam)):.4f}" for k, cam in mapa.items())
    registry_rows.append({"id": exp_id, "data": "2026-10-01", "etapa": "validacao_classificador",
                          "script": "validation/run_protocol.py", "git_commit": "abc1234",
                          "parametros": f"protocolo={parametros['protocolo']};splits={parametros['splits']}",
                          "dataset": "teste", "metricas": metricas, "responsavel": "teste", "notas": ""})


def _escrever_registry(caminho, rows):
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "data", "etapa", "script", "git_commit",
                                           "parametros", "dataset", "metricas", "responsavel", "notas"])
        w.writeheader()
        w.writerows(rows)


SENS_REF = {"bpfi_0.3mm": 1.0, "bpfi_1.0mm": 1.0, "bpfo_0.3mm": 0.0, "bpfo_1.0mm": 1.0}
IDS = {"ref_b": "exp001", "reproducoes_b": [], "ref_a": "exp002", "ref_a_multi": "exp003",
       "semc0_a": "exp004", "semc0_b": "exp005", "norm_a": "exp006", "norm_b": "exp007",
       "permutacao": ["exp008", "exp009", "exp010"], "curva": ["exp011", "exp012", "exp013", "exp014"]}


@pytest.fixture
def ambiente(tmp_path):
    pasta = tmp_path / "validation"
    pasta.mkdir()
    rows: list[dict] = []
    B, A = {**BASE, "protocolo": "B"}, {**BASE, "protocolo": "A"}
    _registrar(pasta, rows, "exp001", B, _rodada_b(SENS_REF))
    _registrar(pasta, rows, "exp002", A, _rodada_a())
    _registrar(pasta, rows, "exp003", {**A, "tarefa": "multiclasse"}, _rodada_a(multiclasse=True))
    _registrar(pasta, rows, "exp004", {**A, "sem_c0": True}, _rodada_a())
    _registrar(pasta, rows, "exp005", {**B, "sem_c0": True}, _rodada_b(SENS_REF))
    norm = {"norm_clipe": True, "features": "mfcc_dsp_media_desvio_normclipe"}
    _registrar(pasta, rows, "exp006", {**A, **norm}, _rodada_a())
    _registrar(pasta, rows, "exp007", {**B, **norm}, _rodada_b(SENS_REF))
    for i, s in zip((8, 9, 10), (0.2, 0.0, 0.4)):
        sens = {f: s for f in C.FALHAS}
        _registrar(pasta, rows, f"exp{i:03d}", {**B, "permutado": True, "semente": i}, _rodada_b(sens))
    curva = {"segundos_treino": 2.0, "amostragem_treino": "por_gravacao"}
    _registrar(pasta, rows, "exp011", {**A, **curva, "semente": 1}, _rodada_a(ab=0.8))
    _registrar(pasta, rows, "exp012", {**A, **curva, "semente": 2}, _rodada_a(ab=0.9))
    _registrar(pasta, rows, "exp013", {**B, **curva, "semente": 1}, _rodada_b(SENS_REF))
    _registrar(pasta, rows, "exp014", {**B, **curva, "semente": 2}, _rodada_b(SENS_REF))
    registry = tmp_path / "registry.csv"
    _escrever_registry(registry, rows)
    return pasta, registry, rows, tmp_path / "saida"


def test_gera_tabelas_e_rastreabilidade(ambiente):
    pasta, registry, _, saida = ambiente
    rod = C.gerar(pasta, registry, saida, IDS, taxas=False, figuras=False)
    for nome in ("tab_protocoloB_folds.tex", "tab_protocoloA_folds.tex", "tab_matriz_confusao_B.tex",
                 "tab_controles.tex", "rastreabilidade.md", "consolidacao.md"):
        assert (saida / nome).exists(), nome
    assert all(c.ok for c in rod.conferencias)
    tex_b = (saida / "tab_protocoloB_folds.tex").read_text(encoding="utf-8")
    assert r"\textbf{0,875}" in tex_b                    # vírgula decimal
    assert "limite superior otimista" in (saida / "tab_protocoloA_folds.tex").read_text(encoding="utf-8").lower()
    assert "exp008–exp010" in (saida / "rastreabilidade.md").read_text(encoding="utf-8")


def test_matriz_soma_os_folds(ambiente):
    pasta, *_ = ambiente
    m = C.matriz_confusao_b(C.carregar(pasta, "exp001"))
    n_folds_por_falha = len(BLOCOS)
    assert m["bpfo_0.3mm"] == {"normal": N_FALHA * n_folds_por_falha, "falha": 0}
    assert m["bpfi_0.3mm"] == {"normal": 0, "falha": N_FALHA * n_folds_por_falha}
    assert m["normal"] == {"normal": N_NORMAL * len(BLOCOS) * 4, "falha": 0}


def test_matriz_recusa_contagem_nao_inteira(ambiente):
    pasta, *_ = ambiente
    d = C.carregar(pasta, "exp001")
    d["folds"][0]["sensibilidade"] = 0.5          # 0,5 × 59 não é inteiro
    with pytest.raises(SystemExit):
        C.matriz_confusao_b(d)


def test_registry_divergente_aborta_sem_gravar(ambiente):
    pasta, registry, rows, saida = ambiente
    rows[0]["metricas"] = rows[0]["metricas"].replace("acc_bal_media=0.8750", "acc_bal_media=0.9000")
    _escrever_registry(registry, rows)
    with pytest.raises(SystemExit, match="não batem"):
        C.gerar(pasta, registry, saida, IDS, taxas=False, figuras=False)
    assert not saida.exists()


def test_rodada_sem_linha_no_registry_aborta(ambiente):
    pasta, registry, rows, saida = ambiente
    _escrever_registry(registry, [r for r in rows if r["id"] != "exp005"])
    with pytest.raises(SystemExit, match="sem linha"):
        C.gerar(pasta, registry, saida, IDS, taxas=False, figuras=False)


@pytest.mark.parametrize("papel, errado", [
    ("ref_b", "exp008"),          # rodada permutada como referência
    ("ref_a", "exp003"),          # multiclasse no lugar do binário
    ("semc0_b", "exp007"),        # normalização no lugar do sem c0
])
def test_rodada_no_papel_errado_aborta(ambiente, papel, errado):
    pasta, registry, _, saida = ambiente
    ids = {**IDS, papel: errado}
    if errado in ids["permutacao"]:
        ids["permutacao"] = [e for e in ids["permutacao"] if e != errado]
    with pytest.raises(SystemExit):
        C.gerar(pasta, registry, saida, ids, taxas=False, figuras=False)


def test_particao_diferente_aborta(ambiente):
    pasta, registry, rows, saida = ambiente
    _registrar(pasta, rows, "exp015", {**BASE, "protocolo": "A", "sem_c0": True, "splits": "outro"},
               _rodada_a())
    _escrever_registry(registry, rows)
    with pytest.raises(SystemExit, match="splits"):
        C.gerar(pasta, registry, saida, {**IDS, "semc0_a": "exp015"}, taxas=False, figuras=False)


def _taxas(pasta, rows, registry, commit_alt="c1", modelo_alt="lda"):
    comum = {**BASE, "protocolo": "B", "features_commit": "c1"}
    _registrar(pasta, rows, "exp020", comum, _rodada_b(SENS_REF))
    _registrar(pasta, rows, "exp021", {**comum, "fs_hz": 25600, "splits": "def",
                                       "features_commit": commit_alt, "modelo": modelo_alt},
               _rodada_b(SENS_REF))
    _escrever_registry(registry, rows)
    return {**IDS, "taxa_ref": "exp020", "taxa_alt": "exp021"}


@pytest.fixture
def sem_particoes(monkeypatch):
    """As partições em segundos são conferidas pelo run_tabela_taxas (e testadas lá)."""
    monkeypatch.setattr(C.run_tabela_taxas, "conferir_particoes", lambda *a, **k: None)


def test_comparacao_de_taxas(ambiente, sem_particoes):
    pasta, registry, rows, saida = ambiente
    ids = _taxas(pasta, rows, registry)
    rod = C.gerar(pasta, registry, saida, ids, figuras=False)
    tex = (saida / "tab_taxas_B.tex").read_text(encoding="utf-8")
    assert "25.600~Hz" in tex and "12.800~Hz" in tex
    assert {"exp020", "exp021"} <= {c.exp_id for c in rod.conferencias}


@pytest.mark.parametrize("kw, motivo", [
    ({"commit_alt": "c2"}, "features_commit"),     # features de outro commit
    ({"modelo_alt": "mlp"}, "modelo"),
])
def test_comparacao_de_taxas_recusa(ambiente, sem_particoes, kw, motivo):
    pasta, registry, rows, saida = ambiente
    ids = _taxas(pasta, rows, registry, **kw)
    with pytest.raises(SystemExit, match=motivo):
        C.gerar(pasta, registry, saida, ids, figuras=False)