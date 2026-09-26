"""
particao.py — segmentação das gravações e partição dos protocolos A e B.

Implementa o protocolo de validação registrado em 24/09 (Notion, Registro de
Decisões: "Protocolo de validação do classificador"). O problema que ele trata:
o dataset tem UMA gravação contínua por classe, então qualquer divisão feita
dentro dela deixa treino e teste no mesmo registro, e o classificador pode
separar as classes pela identidade da gravação em vez da falha. Foi o que deu
acurácia 1,0 em todas as taxas no estudo de decimação.

Terminologia
------------
`pcm_io.Clip` é a GRAVAÇÃO inteira de uma classe (~60 s). A unidade de
classificação, de `config.SEGMENTO_S` segundos, aqui se chama SEGMENTO, para os
dois conceitos não se confundirem no código.

Estrutura
---------
- Cada gravação é cortada em segmentos consecutivos e sem sobreposição. A sobra
  do fim, que não completa um segmento, é descartada.
- Os segmentos são agrupados em blocos temporais de `SEGMENTOS_POR_BLOCO`. O
  bloco é definido por índice (bloco k = segmentos 10k a 10k+9), então o último
  pode ser menor — hoje, 9.
- Faixa de descarte: um segmento de treino a até `SEGMENTOS_DESCARTE` posições
  de um segmento de teste, NA MESMA GRAVAÇÃO, sai daquele fold. Evita treinar
  com o trecho colado ao teste.

Protocolos
----------
A — blocos temporais (limite otimista). Um fold por bloco k: o bloco k de todas
    as gravações vai para teste, o resto para treino. Vale para binário e
    multiclasse. Não conta para a meta.

B — gravação de falha deixada de fora (resultado principal, binário). Para cada
    gravação de falha f e cada bloco k da gravação normal: teste = gravação f
    inteira + bloco k da normal; treino = as outras falhas + o resto da normal.
    Com 4 falhas e 6 blocos normais, são 24 folds.

A partição é determinística: não há sorteio. Ela é gerada uma vez pelo
`pipeline/04_make_splits.py`, gravada em `data/processed/splits/splits.json`
(versionado) e lida por todo experimento. `verificar` confere as garantias
acima e é usada pelo 04, pelos experimentos e pelos testes.

Este módulo não é executável: é importado.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config

FORMATO = 1   # versão do formato do splits.json


# --------------------------------------------------------------------------- #
# Estruturas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Segmento:
    id: int          # global, 0..N-1, na ordem classe → tempo
    rotulo: str      # classe da gravação de origem
    binario: str     # normal | falha
    indice: int      # posição dentro da gravação (0 = primeiro segmento)
    bloco: int       # bloco temporal dentro da gravação
    inicio: int      # primeira amostra, na taxa de trabalho
    fim: int         # uma depois da última: o segmento é x[inicio:fim]


@dataclass
class Fold:
    protocolo: str               # "A" | "B"
    nome: str                    # ex. "A_bloco0", "B_bpfo_0.3mm_bloco3"
    treino: list[int]
    teste: list[int]
    descartados: list[int]       # faixa de descarte: fora do treino e do teste
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Segmentação
# --------------------------------------------------------------------------- #
def _ordem_classe(rotulo: str) -> tuple[int, str]:
    """Ordem estável: a de config.CLASSES; rótulos desconhecidos vão ao fim."""
    if rotulo in config.CLASSES:
        return (config.CLASSES.index(rotulo), rotulo)
    return (len(config.CLASSES), rotulo)


def segmentar(
    clipes,
    amostras_por_segmento: int = config.AMOSTRAS_POR_SEGMENTO,
    segmentos_por_bloco: int = config.SEGMENTOS_POR_BLOCO,
) -> list[Segmento]:
    """
    Corta cada gravação (`pcm_io.Clip`) em segmentos e atribui os blocos.

    Só usa `rotulo`, `binario` e `n` de cada gravação; não lê o sinal.
    """
    segmentos: list[Segmento] = []
    for c in sorted(clipes, key=lambda c: _ordem_classe(c.rotulo)):
        esperado = config.BINARIO.get(c.rotulo)
        if esperado is not None and esperado != c.binario:
            raise ValueError(
                f"{c.rotulo}: manifest diz '{c.binario}', config.BINARIO diz '{esperado}'"
            )
        n_seg = c.n // amostras_por_segmento
        if n_seg == 0:
            raise ValueError(
                f"{c.rotulo}: {c.n} amostras não completam um segmento de "
                f"{amostras_por_segmento}"
            )
        for k in range(n_seg):
            segmentos.append(Segmento(
                id=len(segmentos),
                rotulo=c.rotulo,
                binario=c.binario,
                indice=k,
                bloco=k // segmentos_por_bloco,
                inicio=k * amostras_por_segmento,
                fim=(k + 1) * amostras_por_segmento,
            ))
    return segmentos


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def _vizinhos_do_teste(segmentos: list[Segmento], candidatos: list[int],
                       teste: list[int], distancia: int) -> set[int]:
    """Candidatos a treino a até `distancia` de um segmento de teste da mesma gravação."""
    if distancia <= 0:
        return set()
    teste_por_gravacao: dict[str, list[int]] = {}
    for i in teste:
        s = segmentos[i]
        teste_por_gravacao.setdefault(s.rotulo, []).append(s.indice)
    fora: set[int] = set()
    for i in candidatos:
        s = segmentos[i]
        indices = teste_por_gravacao.get(s.rotulo)
        if indices and min(abs(s.indice - t) for t in indices) <= distancia:
            fora.add(i)
    return fora


def _montar(protocolo: str, nome: str, segmentos: list[Segmento], teste: list[int],
            descarte: int, info: dict) -> Fold:
    no_teste = set(teste)
    candidatos = [s.id for s in segmentos if s.id not in no_teste]
    fora = _vizinhos_do_teste(segmentos, candidatos, teste, descarte)
    return Fold(
        protocolo=protocolo,
        nome=nome,
        treino=[i for i in candidatos if i not in fora],
        teste=sorted(teste),
        descartados=sorted(fora),
        info=info,
    )


def folds_protocolo_a(segmentos: list[Segmento],
                      descarte: int = config.SEGMENTOS_DESCARTE) -> list[Fold]:
    """Um fold por bloco: o bloco k de todas as gravações vai para teste."""
    folds = []
    for b in sorted({s.bloco for s in segmentos}):
        teste = [s.id for s in segmentos if s.bloco == b]
        folds.append(_montar("A", f"A_bloco{b}", segmentos, teste, descarte,
                             {"bloco": b}))
    return folds


def folds_protocolo_b(segmentos: list[Segmento],
                      descarte: int = config.SEGMENTOS_DESCARTE) -> list[Fold]:
    """Cada gravação de falha deixada de fora, combinada com cada bloco da normal."""
    normais = sorted({s.rotulo for s in segmentos if s.binario == "normal"},
                     key=_ordem_classe)
    falhas = sorted({s.rotulo for s in segmentos if s.binario == "falha"},
                    key=_ordem_classe)
    if len(normais) != 1:
        raise ValueError(f"o Protocolo B supõe uma gravação normal; há {normais}")
    if len(falhas) < 2:
        raise ValueError("o Protocolo B precisa de ao menos 2 gravações de falha")
    normal = normais[0]
    blocos_normal = sorted({s.bloco for s in segmentos if s.rotulo == normal})

    folds = []
    for f in falhas:
        for b in blocos_normal:
            teste = [s.id for s in segmentos
                     if s.rotulo == f or (s.rotulo == normal and s.bloco == b)]
            folds.append(_montar("B", f"B_{f}_bloco{b}", segmentos, teste, descarte,
                                 {"falha_de_fora": f, "bloco_normal": b}))
    return folds


# --------------------------------------------------------------------------- #
# Documento completo (o que vai para o splits.json)
# --------------------------------------------------------------------------- #
def gerar_particoes(
    clipes,
    amostras_por_segmento: int = config.AMOSTRAS_POR_SEGMENTO,
    segmentos_por_bloco: int = config.SEGMENTOS_POR_BLOCO,
    descarte: int = config.SEGMENTOS_DESCARTE,
) -> dict:
    segmentos = segmentar(clipes, amostras_por_segmento, segmentos_por_bloco)
    folds = folds_protocolo_a(segmentos, descarte) + folds_protocolo_b(segmentos, descarte)
    return {
        "formato": FORMATO,
        "parametros": {
            "fs_hz": int(clipes[0].fs),
            "segmento_s": amostras_por_segmento / clipes[0].fs,
            "amostras_por_segmento": amostras_por_segmento,
            "segmentos_por_bloco": segmentos_por_bloco,
            "segmentos_descarte": descarte,
        },
        "gravacoes": {
            c.rotulo: {"binario": c.binario, "n_amostras": c.n,
                       "n_segmentos": c.n // amostras_por_segmento}
            for c in sorted(clipes, key=lambda c: _ordem_classe(c.rotulo))
        },
        "segmentos": [asdict(s) for s in segmentos],
        "folds": [asdict(f) for f in folds],
    }


def segmentos_de(particoes: dict) -> list[Segmento]:
    return [Segmento(**s) for s in particoes["segmentos"]]


def folds_de(particoes: dict, protocolo: str | None = None) -> list[Fold]:
    folds = [Fold(**f) for f in particoes["folds"]]
    return [f for f in folds if protocolo is None or f.protocolo == protocolo]


# --------------------------------------------------------------------------- #
# Verificação — as garantias do protocolo, conferidas uma a uma
# --------------------------------------------------------------------------- #
def verificar(particoes: dict) -> list[str]:
    """
    Devolve a lista de violações; vazia significa partição válida.

    Mesmo contrato de `pcm_io.verificar_integridade`: quem chama decide como
    reportar. Os experimentos devem abortar se a lista não vier vazia.
    """
    erros: list[str] = []
    segmentos = segmentos_de(particoes)
    descarte = int(particoes["parametros"]["segmentos_descarte"])
    todos = set(range(len(segmentos)))

    if [s.id for s in segmentos] != list(range(len(segmentos))):
        erros.append("ids dos segmentos não são 0..N-1 em ordem")
        return erros

    for f in folds_de(particoes):
        tr, te, fo = set(f.treino), set(f.teste), set(f.descartados)
        if len(tr) != len(f.treino) or len(te) != len(f.teste):
            erros.append(f"{f.nome}: segmento repetido dentro de treino ou teste")
        if tr & te:
            erros.append(f"{f.nome}: {len(tr & te)} segmento(s) em treino E teste")
        if fo & (tr | te):
            erros.append(f"{f.nome}: descartado aparece em treino ou teste")
        if tr | te | fo != todos:
            erros.append(f"{f.nome}: {len(todos - (tr | te | fo))} segmento(s) sem destino")
        if not te:
            erros.append(f"{f.nome}: teste vazio")

        # faixa de descarte: nenhum treino colado a um teste da mesma gravação
        if _vizinhos_do_teste(segmentos, f.treino, f.teste, descarte):
            erros.append(f"{f.nome}: treino a até {descarte} segmento(s) de um teste "
                         "na mesma gravação")

        # as duas classes binárias precisam estar no treino
        if {segmentos[i].binario for i in f.treino} != {"normal", "falha"}:
            erros.append(f"{f.nome}: treino sem uma das classes binárias")

        if f.protocolo == "B":
            falha = f.info.get("falha_de_fora")
            da_falha = {s.id for s in segmentos if s.rotulo == falha}
            if not da_falha:
                erros.append(f"{f.nome}: falha deixada de fora '{falha}' não existe")
            if not da_falha <= te:
                erros.append(f"{f.nome}: gravação '{falha}' não está inteira no teste")
            if da_falha & tr:
                erros.append(f"{f.nome}: gravação '{falha}' aparece no treino")
            outros = [segmentos[i] for i in te if segmentos[i].rotulo != falha]
            if len({s.rotulo for s in outros}) != 1 or \
                    any(s.binario != "normal" for s in outros):
                erros.append(f"{f.nome}: o teste deveria ter só a falha de fora e a normal")
            blocos_normal = {segmentos[i].bloco for i in te if segmentos[i].binario == "normal"}
            if len(blocos_normal) != 1:
                erros.append(f"{f.nome}: teste com {len(blocos_normal)} blocos normais")

    # no Protocolo A, cada segmento cai em teste exatamente uma vez
    folds_a = folds_de(particoes, "A")
    if folds_a:
        vezes = [0] * len(segmentos)
        for f in folds_a:
            for i in f.teste:
                vezes[i] += 1
        if any(v != 1 for v in vezes):
            erros.append("Protocolo A: há segmento testado 0 ou mais de 1 vez")

    return erros


def conferir_compatibilidade(particoes: dict, clipes) -> list[str]:
    """
    Confere se o splits.json descreve os dados carregados agora.

    Se a decimação ou os parâmetros de segmentação mudarem, a partição gravada
    deixa de corresponder aos dados, e os índices passam a apontar para trechos
    errados sem erro nenhum. Esta checagem transforma isso em erro.
    """
    erros: list[str] = []
    p = particoes["parametros"]
    esperado = {
        "fs_hz": int(config.FS_TRABALHO),
        "amostras_por_segmento": config.AMOSTRAS_POR_SEGMENTO,
        "segmentos_por_bloco": config.SEGMENTOS_POR_BLOCO,
        "segmentos_descarte": config.SEGMENTOS_DESCARTE,
    }
    for k, v in esperado.items():
        if p.get(k) != v:
            erros.append(f"splits.json tem {k}={p.get(k)}, config.py tem {v}")
    gravacoes = particoes.get("gravacoes", {})
    for c in clipes:
        g = gravacoes.get(c.rotulo)
        if g is None:
            erros.append(f"{c.rotulo}: gravação carregada não existe no splits.json")
        elif g["n_amostras"] != c.n:
            erros.append(f"{c.rotulo}: {c.n} amostras carregadas, splits.json foi "
                         f"gerado com {g['n_amostras']}")
    for r in set(gravacoes) - {c.rotulo for c in clipes}:
        erros.append(f"{r}: está no splits.json mas não foi carregada")
    return erros


# --------------------------------------------------------------------------- #
# Arquivo
# --------------------------------------------------------------------------- #
_LISTA_INTEIROS = re.compile(r"\[\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\]")
_OBJETO_PLANO = re.compile(r"\{[^{}\[\]]*\}")   # objeto sem nada aninhado


def para_texto(particoes: dict) -> str:
    """
    JSON indentado, mas com listas de inteiros e objetos planos numa linha só:
    um segmento por linha, um fold em poucas linhas. Mantém o arquivo legível e
    os diffs do Git informativos.
    """
    texto = json.dumps(particoes, ensure_ascii=False, indent=2)
    texto = _LISTA_INTEIROS.sub(
        lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
        texto,
    )
    texto = _OBJETO_PLANO.sub(lambda m: re.sub(r"\s*\n\s*", " ", m.group(0)), texto)
    return texto + "\n"


def salvar(particoes: dict, caminho: Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(para_texto(particoes), encoding="utf-8")
    return caminho


def carregar(caminho: Path) -> dict:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"não achei {caminho}. Ele é versionado; se não existe, rode "
            "scripts/pipeline/04_make_splits.py."
        )
    particoes = json.loads(caminho.read_text(encoding="utf-8"))
    if particoes.get("formato") != FORMATO:
        raise ValueError(f"{caminho}: formato {particoes.get('formato')}, esperado {FORMATO}")
    return particoes


def hash_arquivo(caminho: Path) -> str:
    """Hash curto do splits.json: vai para o registry e prova qual partição foi usada."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:10]