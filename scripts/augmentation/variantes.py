"""
variantes.py — variantes de aumento de dados por segmento, e o filtro por fold.

Uma VARIANTE é uma versão alterada de um segmento de 1 s da partição. Ela é
descrita por poucos números — deslocamento, taxa de estiramento, SNR do ruído
— e pelo intervalo da gravação que ela lê. Isso resolve as duas perguntas que
importam para não vazar teste para o treino:

1. De onde a variante tira amostras?  `intervalo_lido` responde, e é a mesma
   função que `aplicar` usa para ler o sinal: não há como as duas divergirem.
2. Ela pode entrar no treino deste fold?  `ids_lidos` diz quais segmentos da
   partição o intervalo toca; `mascara_fold` aceita a variante só se TODOS
   forem de treino naquele fold. Uma variante que encosta num segmento de
   teste ou da faixa de descarte fica de fora — a faixa de descarte do
   protocolo continua valendo com o aumento ligado.

Reprodutibilidade
-----------------
O sorteio de cada variante usa uma semente derivada de (config.SEMENTE, id do
segmento, número da cópia). A variante não depende da ordem em que é gerada nem
das outras técnicas ligadas: os três parâmetros são sempre sorteados, e a
técnica desligada só é ignorada. Assim, "só deslocamento" e "tudo ligado" usam
o mesmo deslocamento para a mesma (segmento, cópia), e as ablações comparam
técnica, não sorte.

Este módulo não é executável: é importado pelo 04 e pelo run_protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config
from augmentation import transformacoes

TECNICAS = config.AUMENTO_TECNICAS


@dataclass(frozen=True)
class Variante:
    origem: int            # id do segmento de origem (splits.json)
    copia: int             # 0..n_copias-1
    deslocamento: int      # amostras, relativo ao segmento (0 = sem deslocamento)
    taxa: float            # taxa do estiramento (1.0 = sem estiramento)
    snr_db: float | None   # None = sem ruído
    inicio_lido: int       # primeira amostra lida da gravação
    fim_lido: int          # uma depois da última


def validar_tecnicas(tecnicas) -> tuple[str, ...]:
    tecnicas = tuple(t.strip() for t in tecnicas if t.strip())
    if not tecnicas:
        raise ValueError("aumento pedido sem nenhuma técnica")
    desconhecidas = set(tecnicas) - set(TECNICAS)
    if desconhecidas:
        raise ValueError(f"técnicas desconhecidas: {sorted(desconhecidas)}; "
                         f"as válidas são {TECNICAS}")
    # ordem canônica, para o nome da rodada e o manifesto não dependerem da digitação
    return tuple(t for t in TECNICAS if t in tecnicas)


def intervalo_lido(segmento, deslocamento: int, taxa: float, n_gravacao: int,
                   n_fft: int = config.AUMENTO_PV_NFFT) -> tuple[int, int]:
    """
    Intervalo [a, b) da gravação que a variante lê.

    A janela é centrada no centro do segmento mais o deslocamento. Com
    estiramento, lê `ceil(n·taxa)` amostras, que viram ~n depois de estiradas,
    mais `n_fft` de cada lado para as bordas (do phase vocoder ou da
    reamostragem), que são descartadas no recorte central. Se a janela passar do começo ou do fim da
    gravação, ela é empurrada para dentro.
    """
    n = segmento.fim - segmento.inicio
    comprimento = n if taxa == 1.0 else math.ceil(n * taxa) + 2 * n_fft
    if comprimento > n_gravacao:
        raise ValueError("gravação mais curta que a janela de leitura da variante")
    centro = segmento.inicio + n // 2 + deslocamento
    a = centro - comprimento // 2
    a = min(max(a, 0), n_gravacao - comprimento)
    return a, a + comprimento


def sortear(segmento, copia: int, tecnicas, n_gravacao: int,
            semente: int = config.SEMENTE) -> Variante:
    """Parâmetros da cópia `copia` do segmento, sorteados de forma reprodutível."""
    tecnicas = validar_tecnicas(tecnicas)
    rng = np.random.default_rng([semente, segmento.id, copia])
    d_max = int(round(config.AUMENTO_DESLOC_MAX_S * config.FS_TRABALHO))
    # os três sorteios acontecem sempre, na mesma ordem (ver docstring do módulo)
    d = int(rng.integers(-d_max, d_max + 1))
    r = float(rng.uniform(*config.AUMENTO_ESTIR_TAXAS))
    s = float(rng.uniform(*config.AUMENTO_SNR_DB))

    deslocamento = d if "deslocamento" in tecnicas else 0
    taxa = r if "estiramento" in tecnicas else 1.0
    snr_db = s if "ruido" in tecnicas else None
    a, b = intervalo_lido(segmento, deslocamento, taxa, n_gravacao)
    return Variante(segmento.id, copia, deslocamento, taxa, snr_db, a, b)


def aplicar(x_gravacao: np.ndarray, variante: Variante, n: int,
            modo: str = config.AUMENTO_ESTIR_MODO,
            semente: int = config.SEMENTE) -> np.ndarray:
    """
    Sinal da variante, com `n` amostras: lê o intervalo → estira e recorta o
    centro → soma ruído. O ruído tem semente própria, derivada da mesma
    (segmento, cópia), para ser reproduzível.
    """
    y = np.asarray(x_gravacao[variante.inicio_lido:variante.fim_lido], dtype=np.float64)
    if variante.taxa != 1.0:
        y = transformacoes.recortar_centro(transformacoes.estirar(y, variante.taxa, modo), n)
    if len(y) != n:
        raise AssertionError(f"variante com {len(y)} amostras, esperava {n}")
    if variante.snr_db is not None:
        rng = np.random.default_rng([semente, variante.origem, variante.copia, 1])
        y = transformacoes.adicionar_ruido(y, variante.snr_db, rng)
    return y


# --------------------------------------------------------------------------- #
# Filtro por fold
# --------------------------------------------------------------------------- #
def ids_lidos(segmentos, origem: np.ndarray, inicio: np.ndarray,
              fim: np.ndarray) -> list[tuple[int, ...]]:
    """
    Para cada variante, os ids dos segmentos da partição que o intervalo lido
    toca, na gravação da origem. Amostras da sobra do fim da gravação, que não
    formam segmento e nunca são teste, não contam.
    """
    por_gravacao: dict[str, list] = {}
    for s in segmentos:
        por_gravacao.setdefault(s.rotulo, []).append(s)
    saida = []
    for o, a, b in zip(origem, inicio, fim):
        gravacao = por_gravacao[segmentos[int(o)].rotulo]
        saida.append(tuple(s.id for s in gravacao if s.inicio < b and a < s.fim))
    return saida


def mascara_fold(treino, lidos: list[tuple[int, ...]]) -> np.ndarray:
    """True para as variantes cujos segmentos lidos são todos de treino no fold."""
    treino = set(int(i) for i in treino)
    return np.array([bool(ids) and set(ids) <= treino for ids in lidos], dtype=bool)
