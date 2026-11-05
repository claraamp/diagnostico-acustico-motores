"""
Testes da tabela única do Protocolo C (run_tabela_protocolo_c.py).

Usam `metrics.json` e um `registry.csv` mínimos em tmp_path, com duas linhas no
lugar das do HUST; não dependem de data/. Conferem que (1) a tabela sai quando
o registry bate, (2) contagem diferente ou rodada substituída abortam antes de
gravar, (3) a decisão segue a regra de cada etapa: na rotação deixada de fora
com a regra da página do D2, só 10 e 20 Hz decidem, e (4) a detecção não lida e
a extrapolação saem marcadas, com a detecção não lida fora do total.
"""

from __future__ import annotations

import csv
import json

import pytest

from validation import run_tabela_protocolo_c as T

CAMPOS = ("id", "data", "etapa", "script", "git_commit", "parametros", "dataset", "metricas",
          "responsavel", "notas")


def _alarme(k, n, decisao):
    return {"k": k, "n": n, "decisao": decisao}


def _rot(alarmes, papeis=None):
    """por_rotacao com 5 normais e 5 falhas detectadas em cada rotação."""
    pr = {}
    for v, k in alarmes.items():
        pr[str(v)] = {"alarme": _alarme(k, 5, "reprova" if k >= 2 else "inconclusivo"),
                      "deteccao": {"k": 5, "n": 5}}
        if papeis:
            pr[str(v)]["papel"] = papeis[v]
    return pr


C2 = {"alarme": _alarme(0, 8, "inconclusivo"), "deteccao": {"k": 15, "n": 20},
      "por_rotacao": {str(v): {"alarme": {"k": 0, "n": 2}, "deteccao": {"k": 0 if v == 5 else 5, "n": 5}}
                      for v in T.ROTACOES}}
PAPEIS = {5: "extrapolacao", 10: "decide", 20: "decide", 30: "extrapolacao"}
RF = {"alarme": _alarme(15, 20, "reprova"), "deteccao": {"k": 20, "n": 20},
      "por_rotacao": _rot({5: 5, 10: 0, 20: 5, 30: 5}, PAPEIS), "decisao_pagina_d2": "reprova"}

LINHAS = (T.Linha("exp001", "", "C2_D2", "sozinho", "C2", "D2", "reporte"),
          T.Linha("exp002", "rotacao_fora", "RF_D2", "sozinho", "Rotação de fora", "D2", "decide"))


def _registro(exp_id, r, notas=""):
    m = {"alarme_sozinho": f"{r['alarme']['k']}/{r['alarme']['n']}",
         "deteccao_sozinho": f"{r['deteccao']['k']}/{r['deteccao']['n']}"}
    if "decisao_pagina_d2" in r:
        m |= {"decisao_total_sozinho": r["alarme"]["decisao"], "decisao_pagina_d2": r["decisao_pagina_d2"]}
        for v, pr in r["por_rotacao"].items():
            m |= {f"alarme_{v}hz": f"{pr['alarme']['k']}/5", f"deteccao_{v}hz": "5/5",
                  f"decisao_{v}hz": pr["alarme"]["decisao"]}
    else:
        m["decisao_sozinho"] = r["alarme"]["decisao"]
    return {"id": exp_id, "script": T.SCRIPT, "parametros": "protocolo=C;detector=D2",
            "metricas": ";".join(f"{k}={v}" for k, v in m.items()), "notas": notas}


@pytest.fixture
def pasta(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "LINHAS", LINHAS)
    p = tmp_path / "protocolo_c"
    (p / "rotacao_fora").mkdir(parents=True)
    (p / "metrics.json").write_text(json.dumps(
        {"leituras": {"C2_D2": {"detector": "D2", "resultado": {"sozinho": C2}}}}), encoding="utf-8")
    (p / "rotacao_fora" / "metrics.json").write_text(json.dumps(
        {"leituras": {"RF_D2": {"detector": "D2", "resultado": {"sozinho": RF}}}}), encoding="utf-8")
    return p


def _gravar_registry(caminho, linhas):
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS, restval="")
        w.writeheader()
        w.writerows(linhas)


def test_tabela_sai_quando_o_registry_bate(pasta, tmp_path):
    reg = tmp_path / "registry.csv"
    _gravar_registry(reg, [_registro("exp001", C2), _registro("exp002", RF)])
    n = T.gerar(pasta, reg)
    assert n == 3 + 3 + 1 + 4 * 3          # C2: alarme, detecção, decisão; RF: + página do D2 e 4 rotações
    md = (pasta / "tabela_protocolo_c.md").read_text(encoding="utf-8")
    assert "| 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 5/5 · 5/5 · 5/5 | 0/8 | 15/20 | inconclusivo | `exp001` |" in md
    assert "reprova a 20 Hz | `exp002`" in md
    tex = (pasta / "tab_protocoloC.tex").read_text(encoding="utf-8")
    assert r"reprova a 20~Hz & \texttt{exp002}" in tex


def test_contagem_diferente_ou_rodada_substituida_aborta_sem_gravar(pasta, tmp_path):
    reg = tmp_path / "registry.csv"
    errado = _registro("exp001", C2)
    errado["metricas"] = errado["metricas"].replace("alarme_sozinho=0/8", "alarme_sozinho=1/8")
    _gravar_registry(reg, [errado, _registro("exp002", RF, notas="substituída por exp003")])
    with pytest.raises(SystemExit) as e:
        T.gerar(pasta, reg)
    assert "alarme_sozinho registry=1/8" in str(e.value)
    assert "exp002: substituída" in str(e.value)
    assert not (pasta / "tabela_protocolo_c.md").exists()


def test_criterio_da_recalibracao_e_soma_por_rotacao_sao_conferidos(pasta, tmp_path, monkeypatch):
    recal = {**C2, "deteccao": {"k": 0, "n": 20},
             "por_rotacao": {v: {**pr, "deteccao": {"k": 0, "n": 5}} for v, pr in C2["por_rotacao"].items()},
             "deteccao_criterio": {"k": 0, "n": 15, "passa": False},
             "deteccao_fora_do_criterio": {"k": 0, "n": 5}}
    (pasta / "recalibracao").mkdir()
    (pasta / "recalibracao" / "metrics.json").write_text(json.dumps(
        {"leituras": {"REC": {"detector": "D2", "resultado": {"sozinho": recal}}}}), encoding="utf-8")
    monkeypatch.setattr(T, "LINHAS", LINHAS + (T.Linha("exp003", "recalibracao", "REC", "sozinho",
                                                       "Recalibração", "LDA", "decide"),))
    errado = _registro("exp003", recal)
    errado["metricas"] += ";deteccao_10a30hz_sozinho=0/15;passa_10a30hz_sozinho=sim;deteccao_5hz_sozinho=0/5"
    # o C2 com uma rotação que não soma o total: 1/2 a 30 Hz e 0/8 no total
    c2 = json.loads((pasta / "metrics.json").read_text(encoding="utf-8"))
    c2["leituras"]["C2_D2"]["resultado"]["sozinho"]["por_rotacao"]["30"]["alarme"]["k"] = 1
    (pasta / "metrics.json").write_text(json.dumps(c2), encoding="utf-8")
    reg = tmp_path / "registry.csv"
    _gravar_registry(reg, [_registro("exp001", C2), _registro("exp002", RF), errado])
    with pytest.raises(SystemExit) as e:
        T.gerar(pasta, reg)
    assert "exp003: passa_10a30hz_sozinho registry=sim metrics.json=nao" in str(e.value)
    assert "exp001: alarme por rotação soma 1/8, total 0/8" in str(e.value)


def test_rotacao_de_fora_so_le_10_e_20_hz_pela_regra_da_pagina_do_d2():
    # 5 e 30 Hz também têm 5/5, mas são extrapolação; sem a regra da página, contam todas
    assert T.decisao(RF) == "reprova a 20 Hz"
    sem_pagina = {**RF, "por_rotacao": _rot({5: 5, 10: 0, 20: 5, 30: 5}, {v: "reporte" for v in PAPEIS})}
    del sem_pagina["decisao_pagina_d2"]
    assert T.decisao(sem_pagina) == "reprova a 5, 20 e 30 Hz"
    assert T.decisao(C2) == "inconclusivo"


def test_rotacao_que_a_leitura_nao_avalia_sai_como_traco():
    d1 = {"alarme": _alarme(4, 10, "reprova"),
          "por_rotacao": {"20": {"alarme": {"k": 3, "n": 5}}, "30": {"alarme": {"k": 1, "n": 5}}}}
    assert T.por_rotacao(d1, "alarme") == ["—", "—", "3/5", "1/5"]


def test_deteccao_nao_lida_e_extrapolacao_saem_marcadas_e_fora_do_total():
    # pela página do D2: a detecção onde o D2 alarma os normais não conta; 5 e 30 Hz são extrapolação
    rf = {**RF, "por_rotacao": {v: {**pr, "deteccao_lida": pr["alarme"]["decisao"] != "reprova"}
                                for v, pr in RF["por_rotacao"].items()}}
    assert T.por_rotacao(rf, "alarme") == ["[5/5]", "0/5", "5/5", "[5/5]"]
    assert T.por_rotacao(rf, "deteccao") == ["(5/5)", "5/5", "(5/5)", "(5/5)"]
    assert T.deteccao_total(rf) == "5/5"
    # sem o campo: decide o alarme do total; se reprova, toda a detecção sai entre parênteses e o total é traço
    assert T.por_rotacao(C2, "deteccao") == ["0/5", "5/5", "5/5", "5/5"]
    assert T.deteccao_total(C2) == "15/20"
    reprova = {**C2, "alarme": _alarme(8, 8, "reprova")}
    assert T.por_rotacao(reprova, "deteccao") == ["(0/5)", "(5/5)", "(5/5)", "(5/5)"]
    assert T.deteccao_total(reprova) == "—"


def test_recalibracao_decide_uma_vez_com_o_resultado_da_pagina():
    recal = {**C2, "deteccao_criterio": {"k": 0, "n": 15, "passa": False},
             "deteccao_fora_do_criterio": {"k": 0, "n": 5}}
    assert T.decisao(recal, "não funciona") == \
        "não funciona, com o controle sem o c0 (0/15 a 10–30 Hz; 5 Hz à parte: 0/5)"
    assert T.decisao(recal) == "0/15 a 10–30 Hz; 5 Hz à parte: 0/5"
