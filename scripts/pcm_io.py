"""
pcm_io.py — leitura e escrita dos vetores PCM e dos manifests do pipeline.

Formato do manifest (escrito pelo `01_convert_mat_to_pcm.py`): dicionário plano,
classe → metadados.

    {
      "normal": {
        "arquivo_origem": "0Nm_Normal.mat",
        "rotulo_binario": "normal",
        "arquivo_pcm": "data/processed/pcm_raw/normal.bin",
        "fs_hz": 51200,
        "n_amostras": 3072000,
        "duracao_s": 60.0,
        "pico_original_pa": 0.876,
        "pico_pcm": 8505
      },
      ...
    }

Os `.bin` são PCM cru, `int16` little-endian, sem cabeçalho, um arquivo por
classe. Eles NÃO são versionados; o manifest É (ver CONVENTIONS.md, seção 1).
Essa assimetria é útil: como a conversão é determinística, o manifest versionado
serve de referência para verificar se uma reconversão reproduziu a anterior —
é o que faz `verificar_integridade`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

import config


@dataclass
class Clip:
    """
    Um registro de áudio de uma classe.
 
    Duas origens possíveis, e o clipe precisa de exatamente uma delas:
 
    - `pcm`: inteiro de 16 bits lido de um .bin. É o caso dos clipes do
      dataset, e o `.x` em float é derivado sob demanda.
    - `x_float`: sinal já em float, quando ele nasce de um processamento e
      nunca passou por int16 — por exemplo a saída da decimação, antes de ser
      gravada. Quantizar só para depois desquantizar introduziria ruído que não
      existe no pipeline real.
    """
 
    rotulo: str                       # classe (normal, bpfi_0.3mm, ...)
    binario: str                      # normal | falha
    fs: float                         # Hz
    pcm: np.ndarray | None = None     # int16, como está no arquivo
    x_float: np.ndarray | None = None # sinal em float, quando nasce assim
    arquivo: Path | None = None       # caminho do .bin, quando veio de um
    origem: str | None = None         # .mat de origem, quando o manifest informa
    meta: dict = field(default_factory=dict)   # entrada completa do manifest
 
    def __post_init__(self) -> None:
        if self.pcm is None and self.x_float is None:
            raise ValueError(f"{self.rotulo}: Clip precisa de `pcm` ou de `x_float`")
        self._cache_x: np.ndarray | None = None
 
    @property
    def x(self) -> np.ndarray:
        """Sinal em float64 em [-1, 1]. Derivado do int16 sob demanda, e em cache."""
        if self.x_float is not None:
            return self.x_float
        if self._cache_x is None:
            self._cache_x = self.pcm.astype(np.float64) / config.INT16_FULL
        return self._cache_x
 
    def como_pcm(self) -> np.ndarray:
        """Representação int16, quantizando se o clipe nasceu em float."""
        return self.pcm if self.pcm is not None else para_pcm(self.x_float)
 
    def derivado(self, x: np.ndarray, fs: float) -> "Clip":
        """Novo clipe com o mesmo rótulo, a partir de um sinal processado."""
        return Clip(rotulo=self.rotulo, binario=self.binario, fs=fs,
                    x_float=x, arquivo=self.arquivo, origem=self.origem)
 
    @property
    def n(self) -> int:
        return int(self.pcm.size if self.pcm is not None else self.x_float.size)
 
    @property
    def duracao_s(self) -> float:
        return self.n / self.fs
 
    @property
    def e_normal(self) -> bool:
        return self.binario == "normal"


# --------------------------------------------------------------------------- #
# Leitura
# --------------------------------------------------------------------------- #
def _entradas(manifest: dict) -> dict:
    """
    Normaliza o manifest para `{rotulo: metadados}`.

    Aceita dois formatos: o dicionário plano do `01`, e o formato antigo com uma
    lista sob a chave `files`, que versões anteriores do `02` gravavam para os
    PCM decimados. Escrita, só o formato plano — ver `gravar_clipes`.
    """
    if "files" in manifest and isinstance(manifest["files"], list):
        saida = {}
        for e in manifest["files"]:
            rotulo = e.get("label") or e.get("rotulo") or Path(e.get("bin", "")).stem
            saida[str(rotulo)] = {
                "arquivo_pcm": e.get("bin"),
                "rotulo_binario": e.get("binary") or e.get("rotulo_binario"),
                "fs_hz": e.get("fs_hz", manifest.get("fs_hz")),
                "n_amostras": e.get("n_amostras"),
                "pico_pcm": e.get("pico_pcm"),
            }
        return saida
    # chaves iniciadas por "_" são metadados do conjunto, não classes
    return {k: v for k, v in manifest.items()
            if isinstance(v, dict) and not k.startswith("_")}


def carregar_manifest(caminho: Path) -> dict:
    """Lê o manifest.json e devolve `{rotulo: metadados}`."""
    with Path(caminho).open(encoding="utf-8") as fh:
        return _entradas(json.load(fh))


def carregar_clipes(
    pcm_dir: Path,
    manifest: Path | None = None,
    segundos: float | None = None,
    fs_esperado: float | None = None,
) -> list[Clip]:
    """
    Carrega todos os clipes descritos por um manifest.

    `pcm_dir`      diretório dos .bin; também onde se procura o manifest
    `manifest`     caminho explícito; por omissão, `pcm_dir/manifest.json`
    `segundos`     se dado, usa só os primeiros N segundos de cada clipe
    `fs_esperado`  se dado, falha se o manifest declarar outra taxa

    O campo `arquivo_pcm` do manifest é relativo à raiz do repositório. Se o
    script for chamado de outro diretório, cai para `pcm_dir/<nome do arquivo>`.
    """
    pcm_dir = Path(pcm_dir)
    caminho_manifest = Path(manifest) if manifest else pcm_dir / "manifest.json"
    if not caminho_manifest.exists():
        raise FileNotFoundError(
            f"não achei {caminho_manifest}. O manifest é versionado; se ele existe "
            "no repositório e os .bin não, rode 01_convert_mat_to_pcm.py."
        )
    entradas = carregar_manifest(caminho_manifest)
    if not entradas:
        raise RuntimeError(f"nenhuma classe lida de {caminho_manifest}")

    clipes: list[Clip] = []
    for rotulo, meta in entradas.items():
        arquivo = Path(str(meta.get("arquivo_pcm", "")))
        if not arquivo.exists():
            arquivo = pcm_dir / arquivo.name
        if not arquivo.exists():
            raise FileNotFoundError(
                f"não achei {arquivo} (classe '{rotulo}'). Os .bin não são "
                "versionados: depois de clonar, rode 01_convert_mat_to_pcm.py."
            )

        fs = float(meta.get("fs_hz", config.FS_ORIGINAL))
        if fs_esperado is not None and fs != fs_esperado:
            raise ValueError(
                f"{rotulo}: manifest declara fs = {fs:.0f} Hz, esperado {fs_esperado:.0f} Hz"
            )

        pcm = np.fromfile(arquivo, dtype=config.PCM_DTYPE)
        if segundos is not None:
            pcm = pcm[: int(segundos * fs)]

        clipes.append(
            Clip(
                rotulo=str(rotulo),
                binario=str(meta.get("rotulo_binario", "falha")),
                fs=fs,
                pcm=pcm,
                arquivo=arquivo,
                origem=meta.get("arquivo_origem"),
                meta=dict(meta),
            )
        )

    taxas = {c.fs for c in clipes}
    if len(taxas) > 1:
        raise ValueError(f"clipes com taxas diferentes no mesmo manifest: {sorted(taxas)}")
    return clipes


def verificar_integridade(clipes: list[Clip], truncado: bool = False) -> list[str]:
    """
    Compara os .bin lidos com o que o manifest versionado afirma.

    Como o manifest é versionado e a conversão é determinística, divergência
    aqui significa que uma reconversão não reproduziu a original — e números
    gerados a partir daí não são comparáveis com os das rodadas anteriores.

    `truncado=True` desliga a checagem de tamanho, para quando se carrega só um
    trecho do sinal (`segundos`), caso em que a diferença é esperada.

    Devolve a lista de divergências; vazia significa tudo certo. Quem chama
    decide como reportar.
    """
    divergencias: list[str] = []
    for c in clipes:
        n_esperado = c.meta.get("n_amostras")
        if not truncado and n_esperado is not None and c.n != n_esperado:
            divergencias.append(
                f"{c.rotulo}: {c.n} amostras no arquivo, {n_esperado} no manifest"
            )
        pico_esperado = c.meta.get("pico_pcm")
        if not truncado and pico_esperado is not None:
            pico = int(np.max(np.abs(c.como_pcm()))) if c.n else 0
            if pico != pico_esperado:
                divergencias.append(
                    f"{c.rotulo}: pico PCM {pico}, manifest diz {pico_esperado}"
                )
    return divergencias


def relatar_integridade(divergencias: list[str]) -> None:
    """Impressão padrão do resultado de `verificar_integridade`."""
    if not divergencias:
        print("  .bin conferem com o manifest versionado (amostras e pico PCM).")
        return
    print("\n  ATENÇÃO — os .bin não batem com o manifest versionado:")
    for d in divergencias:
        print(f"    - {d}")
    print("  A conversão do 01 deveria ser determinística. Entenda a diferença antes")
    print("  de seguir: números gerados a partir daqui não serão comparáveis com os")
    print("  das rodadas anteriores.\n")


# --------------------------------------------------------------------------- #
# Escrita
# --------------------------------------------------------------------------- #
def gravar_clipes(
    clipes: list[Clip],
    destino: Path,
    fs: float,
    extras: dict | None = None,
) -> Path:
    """
    Grava os .bin e o manifest de uma etapa do pipeline.

    Escreve sempre no formato plano do `01` — classe → metadados — para que
    exista um único formato de manifest no repositório. `extras` entra como
    metadados do conjunto (por exemplo, os parâmetros da decimação aplicada).

    Devolve o caminho do manifest escrito.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    for c in clipes:
        caminho = destino / f"{c.rotulo}.bin"
        pcm = c.como_pcm()
        pcm.tofile(caminho)
        manifest[c.rotulo] = {
            "arquivo_origem": c.origem,
            "rotulo_binario": c.binario,
            "arquivo_pcm": str(caminho),
            "fs_hz": int(fs),
            "n_amostras": int(pcm.size),
            "duracao_s": round(pcm.size / fs, 4),
            "pico_pcm": int(np.max(np.abs(pcm))) if pcm.size else 0,
        }

    if extras:
        manifest["_conjunto"] = dict(extras)   # prefixo _ não é classe

    caminho_manifest = destino / "manifest.json"
    caminho_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return caminho_manifest


def para_pcm(x: np.ndarray) -> np.ndarray:
    """Converte float em [-1, 1] de volta para int16, com saturação."""
    return np.clip(np.round(x * config.INT16_FULL), -32768, 32767).astype(config.PCM_DTYPE)