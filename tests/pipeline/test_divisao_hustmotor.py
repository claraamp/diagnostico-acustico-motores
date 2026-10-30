"""
Testes do 11_divisao_hustmotor.py (carregado por caminho, porque o nome começa com
dígito). Não dependem de data/ além do manifest versionado do HUST.

Garantias: o divisao_rotacao.json versionado é o que o manifest gera; no C2, cada
rotação segue a divisao_c2 do manifest e os papéis não se repetem; na rotação deixada de
fora, a calibração é a normal das outras três rotações, sem a rotação de teste, e só 10
e 20 Hz ficam dentro da faixa calibrada; e os ids são os que o run_protocol_c.py usa.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_RAIZ = Path(__file__).resolve().parents[2]
_CAMINHO = _RAIZ / "scripts" / "pipeline" / "11_divisao_hustmotor.py"
_spec = importlib.util.spec_from_file_location("divisao_hustmotor", _CAMINHO)
D = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = D
_spec.loader.exec_module(D)

MANIFEST = json.loads((_RAIZ / D.MANIFEST).read_text(encoding="utf-8"))
VERSIONADO = (_RAIZ / D.SAIDA).read_text(encoding="utf-8")


def test_arquivo_versionado_e_o_que_o_manifest_gera():
    assert D.texto(D.gerar(MANIFEST, D.MANIFEST.as_posix())) == VERSIONADO


def test_c2_segue_a_regra_do_manifest_sem_repetir_papel():
    div = json.loads(VERSIONADO)
    regra = MANIFEST["_conjunto"]["divisao_c2"]
    seg = {s["id"]: s for s in div["segmentos"]}
    assert len(seg) == 40
    vistos = []
    for v, papeis in div["c2"]["por_rotacao"].items():
        for nome, chave, classe in (("calibracao", "blocos_calibracao", "H"), ("descarte", "blocos_descarte", "H"),
                                    ("medida_normal", "blocos_medida", "H"),
                                    ("medida_falha", "blocos_medida_falha", "BF")):
            assert sorted(seg[i]["aquisicao"] for i in papeis[nome]) == sorted(regra[chave])
            assert all(seg[i]["rotacao"] == int(v) and seg[i]["classe"] == classe for i in papeis[nome])
            vistos += papeis[nome]
    assert len(vistos) == len(set(vistos)) == 40


def test_rotacao_deixada_de_fora_calibra_sem_a_rotacao_de_teste():
    div = json.loads(VERSIONADO)
    seg = {s["id"]: s for s in div["segmentos"]}
    faixa = {}
    for v, d in div["rotacao_deixada_de_fora"]["por_rotacao"].items():
        assert len(d["calibracao"]) == 15 and len(d["teste_normal"]) == 5 and len(d["teste_falha"]) == 5
        assert all(seg[i]["classe"] == "H" and seg[i]["rotacao"] != int(v) for i in d["calibracao"])
        assert all(seg[i]["rotacao"] == int(v) for i in d["teste_normal"] + d["teste_falha"])
        faixa[int(v)] = d["dentro_da_faixa_calibrada"]
    assert faixa == {5: False, 10: True, 20: True, 30: False}


def test_ids_e_amostras_sao_os_do_corte_do_bin():
    div = json.loads(VERSIONADO)
    for s in div["segmentos"]:
        assert s["id"] == D.id_segmento(s["arquivo"], s["aquisicao"])
        assert s["amostra_fim"] - s["amostra_inicio"] == div["amostras_por_segmento"]
        assert s["amostra_inicio"] == (s["aquisicao"] - 1) * div["amostras_por_segmento"]
    from validation import run_protocol_c as C
    assert C.id_segmento(C.SegHust("h_20hz", "H", "normal", 20, 3, np.zeros(1))) == "h_20hz_a3"
