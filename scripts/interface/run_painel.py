#!/usr/bin/env python3
"""
run_painel.py — painel de diagnóstico no computador (leitura serial)

Card "Implementar a interface de diagnóstico no computador (leitura serial)". Mostra,
segmento a segmento, o que a placa decidiu: a classe (normal, falha ou erro), o escore
cru da LDA (nunca em %, decisão de 17/10), o LED que a placa tem de estar mostrando, o
histórico das últimas inferências, quantos dos últimos N segmentos indicaram falha,
pacotes perdidos, reinícios e os `$ERR` da recepção. A leitura e o estado estão em
`interface/pacotes.py`; aqui ficam as fontes das linhas e a janela (Tkinter, da
biblioteca padrão do Python).

Três fontes, uma de cada vez:

- `--porta COM5`: a serial da placa (pyserial), no modo de demonstração (origens C e
  D). Aperte o RESET depois de abrir: o modelo é conferido pelo `$CAB` do boot.
  `--salvar arquivo.log` grava a captura bruta.
- `--seguir PASTA_OU_ARQUIVO`: no modo de recepção, a porta é do
  `matriz_serial_placa.py`, que grava cada linha na captura assim que ela chega; o
  painel acompanha essa captura. Com uma pasta (por exemplo
  `reports/embarcado/teste_serial`, a das execuções de ensaio), ele espera a primeira
  captura `.log` criada ou mudada depois de abrir e troca para uma mais nova quando
  ela aparece. As linhas `#pc envio` da captura dão o nome de cada segmento.
- `--arquivo captura.log`: uma captura salva (teste sem placa). `--ritmo S` espera S
  segundos depois de cada `$SEG` e `$ERR`, para ver a sequência como na placa.

Uso (da raiz do repositório):

    python scripts/interface/run_painel.py --porta COM5                      # demonstração
    python scripts/interface/run_painel.py --seguir reports/embarcado/teste_serial
    python scripts/validation/matriz_serial_placa.py --lista --porta COM5 --ids 3:48,3:3 --pausa 4
    python scripts/interface/run_painel.py --arquivo reports/embarcado/capturas/serial-ruido-15-11-2.log --ritmo 0.3
    python scripts/interface/run_painel.py --arquivo captura.log --sem-janela   # texto, sem janela

`--sem-janela` escreve uma linha por segmento e o resumo no terminal; com `--arquivo`,
sai com 1 se alguma linha foi descartada ou se o modelo da placa não é o do Python.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

from interface.pacotes import MODELO, ORIGENS, Painel, modelo_python

BAUD_PADRAO = 921600


# --------------------------------------------------------------------------- #
# Fontes de linhas                                                            #
# --------------------------------------------------------------------------- #
def linhas_arquivo(caminho: Path, ritmo: float = 0.0, parar: threading.Event | None = None,
                   dormir=time.sleep):
    """As linhas de uma captura salva; com ritmo, uma pausa depois de cada $SEG e $ERR."""
    with open(caminho, "rb") as fh:
        for bruta in fh:
            if parar is not None and parar.is_set():
                return
            linha = bruta.decode("ascii", errors="replace")
            yield linha
            if ritmo > 0 and linha.startswith(("$SEG,", "$ERR,")):
                dormir(ritmo)


def linhas_porta(porta: str, baud: int = BAUD_PADRAO, salvar: Path | None = None,
                 parar: threading.Event | None = None, segundos: float | None = None):
    """As linhas da serial (pyserial). DTR e RTS desligados: não reinicia a placa ao abrir."""
    try:
        import serial
    except ImportError:
        raise SystemExit("--porta precisa do pyserial: pip install pyserial") from None
    destino = open(salvar, "wb") if salvar else None
    fim = None if segundos is None else time.monotonic() + segundos
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = porta, baud, 0.2
    s.dtr = s.rts = False
    try:
        s.open()
        while not (parar is not None and parar.is_set()) and (fim is None or time.monotonic() < fim):
            bruta = s.readline()
            if not bruta:
                continue
            if destino:
                destino.write(bruta)
                destino.flush()
            yield bruta.decode("ascii", errors="replace")
    finally:
        s.close()
        if destino:
            destino.close()


def captura_mais_nova(pasta: Path, desde: float) -> Path | None:
    """A captura .log da pasta mudada mais recentemente, se foi mudada depois de `desde`."""
    melhores = sorted((p.stat().st_mtime, p) for p in Path(pasta).glob("*.log") if p.is_file())
    if melhores and melhores[-1][0] >= desde:
        return melhores[-1][1]
    return None


def linhas_seguir(alvo: Path, parar: threading.Event | None = None, desde: float | None = None,
                  intervalo: float = 0.05, segundos: float | None = None, dormir=time.sleep):
    """
    Acompanha uma captura que outro programa está gravando (tail -f), linha a linha.
    Com uma pasta, segue a captura .log mais nova mudada depois de `desde` (por padrão,
    agora) e troca para outra mais nova quando ela aparece, com uma linha
    "#painel captura <nome>" antes da primeira linha dela.
    """
    alvo = Path(alvo)
    desde = time.time() if desde is None else desde
    fim = None if segundos is None else time.monotonic() + segundos
    atual, fh, resto = None, None, b""

    def acabou():
        return (parar is not None and parar.is_set()) or (fim is not None and time.monotonic() >= fim)

    try:
        while not acabou():
            novo = captura_mais_nova(alvo, desde) if alvo.is_dir() else (alvo if alvo.exists() else None)
            if novo is not None and novo != atual:
                if fh:
                    fh.close()
                atual, fh, resto = novo, open(novo, "rb"), b""
                yield f"#painel captura {novo.name}\n"
            dados = fh.read() if fh else b""
            if not dados:
                dormir(intervalo)
                continue
            resto += dados
            *completas, resto = resto.split(b"\n")
            for bruta in completas:
                yield bruta.decode("ascii", errors="replace") + "\n"
    finally:
        if fh:
            fh.close()


def ler_em_thread(gerador, fila: queue.Queue) -> threading.Thread:
    """Põe as linhas da fonte na fila; um None no fim (ou um ('erro', texto) se ela falhar)."""
    def trabalho():
        try:
            for linha in gerador:
                fila.put(linha)
        except Exception as e:                 # porta que sumiu, arquivo que não abre...
            fila.put(("erro", str(e)))
        finally:
            fila.put(None)
    t = threading.Thread(target=trabalho, daemon=True)
    t.start()
    return t


# --------------------------------------------------------------------------- #
# Texto (sem janela)                                                          #
# --------------------------------------------------------------------------- #
def ms(it, unidade=" ms") -> str:
    """Tempo de processamento do segmento; '—' sem a medida (o emulador não tem DWT)."""
    return "—" if it.ms is None else f"{it.ms:.1f}{unidade}"


def linha_texto(p: Painel) -> str:
    it = p.atual
    if it.tipo == "err":
        return f"{it.texto}{(' ' + it.nome) if it.nome else ''}  LED: apagados"
    nome = f" {it.nome}" if it.nome else ""
    led = {"normal": "verde", "falha": "vermelho", "erro": "os dois"}[it.classe]
    k, n = p.consistencia()
    return (f"n={it.n:>5} {ORIGENS[it.origem]:<9} {it.onde} {it.segmento}{nome}: {it.classe.upper():<6} "
            f"escore da LDA {it.escore} ({ms(it)})  LED: {led}  últimos {n}: {k} falha")


def rodar_texto(linhas, p: Painel, saida=print) -> None:
    avisos_vistos = 0
    for linha in linhas:
        mudou = p.linha(linha)
        for _, aviso in p.avisos[avisos_vistos:]:
            saida(f"  aviso: {aviso}")
        avisos_vistos = len(p.avisos)
        if mudou in ("seg", "err") and p.atual is not None and (p.modelo_ok or mudou == "err"):
            saida(linha_texto(p))
        elif mudou == "cab":
            saida(f"$CAB modelo {p.cab['modelo']} ({p.cab['json_sha12']}): "
                  + ("conferido" if p.modelo_ok else "NÃO é o do Python"))
    saida("resumo: " + p.resumo())


# --------------------------------------------------------------------------- #
# Janela                                                                      #
# --------------------------------------------------------------------------- #
CORES = {"normal": "#1e8e3e", "falha": "#d93025", "erro": "#f29900", "apagado": "#5f6368",
         "fundo": "#202124", "texto": "#e8eaed", "fraco": "#9aa0a6"}
TITULOS = {"normal": "NORMAL", "falha": "FALHA", "erro": "ERRO"}


class Janela:
    """A janela do painel (Tkinter). Lê a fila a cada 50 ms e redesenha a partir do Painel."""

    def __init__(self, raiz, painel: Painel, fila: queue.Queue, fonte: str, parar: threading.Event):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.raiz, self.p, self.fila, self.parar = tk, raiz, painel, fila, parar
        self.fonte, self.fim_da_fonte, self.erro_fonte = fonte, False, ""
        raiz.title("Painel de diagnóstico — Lab III")
        raiz.configure(bg=CORES["fundo"])
        raiz.geometry("1120x780")
        raiz.protocol("WM_DELETE_WINDOW", self.fechar)

        estilo = ttk.Style(raiz)
        estilo.theme_use("clam")
        estilo.configure("Treeview", background="#2b2c2f", fieldbackground="#2b2c2f",
                         foreground=CORES["texto"], rowheight=22, font=("Helvetica", 10))
        estilo.configure("Treeview.Heading", background="#3c4043", foreground=CORES["texto"],
                         font=("Helvetica", 10, "bold"))

        def f(pai=None, **k):
            k.setdefault("bg", CORES["fundo"])
            k.setdefault("fg", CORES["texto"])
            return tk.Label(pai or raiz, **k)
        self.l_modelo = f(font=("Helvetica", 11), anchor="w")
        self.l_modelo.pack(fill="x", padx=12, pady=(10, 0))

        topo = tk.Frame(raiz, bg=CORES["fundo"])
        topo.pack(fill="x", padx=12, pady=8)
        self.caixa = tk.Frame(topo, bg=CORES["apagado"], width=540, height=300)
        self.caixa.pack(side="left")
        self.caixa.pack_propagate(False)
        self.l_classe = tk.Label(self.caixa, text="—", font=("Helvetica", 54, "bold"),
                                 bg=CORES["apagado"], fg="white")
        self.l_classe.pack(pady=(18, 0))
        self.l_escore = tk.Label(self.caixa, text="", font=("Helvetica", 22), bg=CORES["apagado"], fg="white")
        self.l_escore.pack()
        self.l_onde = tk.Label(self.caixa, text="", font=("Helvetica", 12), bg=CORES["apagado"], fg="white",
                               wraplength=520, justify="center")
        self.l_onde.pack(pady=(6, 0))

        lado = tk.Frame(topo, bg=CORES["fundo"])
        lado.pack(side="left", fill="both", expand=True, padx=(16, 0))
        f(lado, text="LED que a placa tem de mostrar", font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.cv = tk.Canvas(lado, width=200, height=80, bg=CORES["fundo"], highlightthickness=0)
        self.cv.pack(anchor="w")
        self.led_v = self.cv.create_oval(10, 10, 70, 70, outline="#3c4043", width=2)
        self.led_r = self.cv.create_oval(100, 10, 160, 70, outline="#3c4043", width=2)
        self.l_consist = f(lado, font=("Helvetica", 16, "bold"), anchor="w")
        self.l_consist.pack(anchor="w", pady=(6, 0))
        self.l_cont = f(lado, font=("Helvetica", 11), anchor="w", justify="left")
        self.l_cont.pack(anchor="w", pady=(6, 0))
        f(lado, text="O escore é o valor cru da LDA (falha se ≥ 0). Não é grau de confiança.",
          font=("Helvetica", 9), fg=CORES["fraco"], anchor="w").pack(anchor="w", pady=(6, 0))

        colunas = ("hora", "n", "origem", "onde", "segmento", "classe", "escore", "ms")
        larguras = (80, 60, 80, 230, 300, 70, 100, 80)
        self.arvore = ttk.Treeview(raiz, columns=colunas, show="headings", height=12)
        for c, w in zip(colunas, larguras):
            self.arvore.heading(c, text={"ms": "tempo (ms)", "onde": "clipe ou grupo"}.get(c, c))
            self.arvore.column(c, width=w, anchor="w")
        for tag, cor in (("normal", CORES["normal"]), ("falha", CORES["falha"]), ("erro", CORES["erro"]),
                         ("err", CORES["fraco"]), ("aviso", CORES["erro"])):
            self.arvore.tag_configure(tag, foreground=cor)
        self.arvore.pack(fill="both", expand=True, padx=12)
        self.l_aviso = f(font=("Helvetica", 10), fg=CORES["erro"], anchor="w")
        self.l_aviso.pack(fill="x", padx=12, pady=(4, 0))
        self.l_fonte = f(font=("Helvetica", 9), fg=CORES["fraco"], anchor="w")
        self.l_fonte.pack(fill="x", padx=12, pady=(0, 8))
        self.n_hist = 0
        self.redesenhar()
        raiz.after(50, self.ler_fila)

    def fechar(self):
        self.parar.set()
        self.raiz.destroy()

    def ler_fila(self):
        mudou = False
        for _ in range(2000):                     # no máximo por volta, para a janela não travar
            try:
                item = self.fila.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self.fim_da_fonte = True
            elif isinstance(item, tuple):
                self.erro_fonte = item[1]
            else:
                if item.startswith("#painel captura "):
                    self.fonte = "seguindo " + item.split(" ", 2)[2].strip()
                self.p.linha(item)
            mudou = True
        if mudou:
            self.redesenhar()
        if not self.parar.is_set():
            self.raiz.after(50, self.ler_fila)

    def _pintar(self, cor):
        for w in (self.caixa, self.l_classe, self.l_escore, self.l_onde):
            w.configure(bg=cor)

    def redesenhar(self):
        p = self.p
        if p.modelo_errado:
            self.l_modelo.configure(text="MODELO DIFERENTE: " + p.modelo_errado, fg=CORES["falha"])
        elif p.modelo_ok:
            self.l_modelo.configure(text=f"Modelo {p.cab['modelo']} (JSON {p.cab['json_sha12']}) conferido "
                                         f"com o lda_final.json do Python", fg=CORES["texto"])
        else:
            self.l_modelo.configure(text="Aguardando o $CAB para conferir o modelo: aperte o RESET da placa "
                                         "(na recepção, ele vem ao entrar no modo)", fg=CORES["erro"])
        it = p.atual
        if it is None or not p.modelo_ok:
            self._pintar(CORES["apagado"])
            self.l_classe.configure(text="—")
            self.l_escore.configure(text="")
            self.l_onde.configure(text="aguardando o primeiro segmento" if p.modelo_ok else "")
        elif it.tipo == "err":
            self._pintar(CORES["apagado"])
            self.l_classe.configure(text="$ERR")
            self.l_escore.configure(text=it.texto.replace("$ERR ", "erro de recepção: "))
            self.l_onde.configure(text=(it.nome or "") + "\nquadro descartado; o PC manda de novo")
        else:
            self._pintar(CORES[it.classe])
            self.l_classe.configure(text=TITULOS[it.classe])
            self.l_escore.configure(text=f"escore da LDA  {it.escore}")
            detalhe = f"{it.onde} · {it.segmento} · origem {it.origem}"
            if it.nome:
                detalhe += f"\n{it.nome}"
            if it.novidade:
                detalhe += f"\nnovidade: {it.novidade}"
            self.l_onde.configure(text=detalhe + f"\nn = {it.n}"
                                  + (f" · {ms(it)} de processamento" if it.ms is not None else ""))
        v, r = p.led
        self.cv.itemconfigure(self.led_v, fill=CORES["normal"] if v else "#2b2c2f")
        self.cv.itemconfigure(self.led_r, fill=CORES["falha"] if r else "#2b2c2f")
        k, n = p.consistencia()
        self.l_consist.configure(text=f"{k} de {n} últimos segmentos indicaram falha" if n else "")
        c = p.contagem
        self.l_cont.configure(text=(
            f"segmentos: {c['segmentos']} ({c['normal']} normal, {c['falha']} falha, {c['erro']} erro)\n"
            f"$ERR da recepção: {c['err']}    pacotes perdidos: {c['perdidos']}    reinícios: {c['reinicios']}\n"
            f"linhas descartadas: {c['descartadas']}    origem desconhecida: {c['origem_desconhecida']}"
            + (f"    antes do $CAB: {c['sem_cab']}" if c["sem_cab"] else "")))
        self.l_aviso.configure(text=("último aviso: " + p.avisos[-1][1]) if p.avisos else "")
        estado = " · fonte terminou" if self.fim_da_fonte else ""
        if self.erro_fonte:
            estado = " · ERRO NA FONTE: " + self.erro_fonte
        self.l_fonte.configure(text=self.fonte + estado)
        self._atualizar_historico()

    def _atualizar_historico(self):
        itens = list(self.p.historico)
        if itens == getattr(self, "_ultimos", None):
            return
        self._ultimos = itens
        self.arvore.delete(*self.arvore.get_children())
        for it in reversed(itens):                 # o mais recente em cima
            hora = time.strftime("%H:%M:%S", time.localtime(it.hora))
            if it.tipo == "seg":
                seg = it.segmento + (f" {it.nome}" if it.nome else "")
                valores = (hora, it.n, ORIGENS[it.origem], it.onde, seg, it.classe, it.escore, ms(it, ""))
                tag = it.classe
            elif it.tipo == "err":
                valores = (hora, "", "serial", it.onde, it.nome, "$ERR", it.texto.split()[1], "")
                tag = "err"
            else:
                valores = (hora, "", "", it.texto, "", "aviso", "", "")
                tag = "aviso"
            self.arvore.insert("", "end", values=valores, tags=(tag,))


def rodar_janela(gerador, p: Painel, fonte: str) -> None:
    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit("o painel precisa do Tkinter (no Linux: sudo apt install python3-tk); "
                         "ou use --sem-janela") from None
    fila, parar = queue.Queue(), threading.Event()
    gerador = gerador(parar)
    ler_em_thread(gerador, fila)
    raiz = tk.Tk()
    Janela(raiz, p, fila, fonte, parar)
    raiz.mainloop()
    parar.set()


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    fonte = ap.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--porta", help="serial da placa (COM5, /dev/ttyUSB0): modo de demonstração")
    fonte.add_argument("--seguir", type=Path, help="captura (ou pasta de capturas) que o "
                                                    "matriz_serial_placa.py está gravando: modo de recepção")
    fonte.add_argument("--arquivo", type=Path, help="captura salva (teste sem a placa)")
    ap.add_argument("--baud", type=int, default=BAUD_PADRAO, help="com --porta (padrão 921600)")
    ap.add_argument("--salvar", type=Path, help="com --porta, grava a captura bruta")
    ap.add_argument("--ritmo", type=float, default=0.0, help="com --arquivo, segundos depois de cada $SEG e $ERR")
    ap.add_argument("--segundos", type=float, help="com --porta ou --seguir, para depois desse tempo")
    ap.add_argument("--historico", type=int, default=50, help="linhas do histórico (padrão 50)")
    ap.add_argument("--ultimos", type=int, default=10, help="N do 'k de N últimos indicaram falha' (padrão 10)")
    ap.add_argument("--modelo", type=Path, default=MODELO, help="lda_final.json para conferir o $CAB")
    ap.add_argument("--sem-janela", action="store_true", help="texto no terminal, sem a janela")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):      # saída redirecionada no Windows (cp1252) não quebra no "→"
        sys.stdout.reconfigure(errors="replace")
    if args.ritmo < 0:
        ap.error("--ritmo não pode ser negativo")
    if args.ritmo and not args.arquivo:
        ap.error("--ritmo só vale com --arquivo")
    if args.salvar and not args.porta:
        ap.error("--salvar só vale com --porta")
    if args.historico < 1 or args.ultimos < 1:
        ap.error("--historico e --ultimos têm de ser pelo menos 1")
    try:
        esperado = modelo_python(args.modelo)
    except (OSError, ValueError, KeyError) as e:
        ap.error(f"não deu para ler o modelo do Python ({args.modelo}): {e}")
    p = Painel(esperado, n_historico=args.historico, n_consistencia=args.ultimos)

    if args.arquivo:
        if not args.arquivo.exists():
            ap.error(f"não achei {args.arquivo}")
        gerador = lambda parar: linhas_arquivo(args.arquivo, args.ritmo, parar)
        desc = f"captura {args.arquivo}"
    elif args.seguir:
        if not args.seguir.exists():
            ap.error(f"não achei {args.seguir} (com uma pasta, ela tem de existir)")
        desde = time.time()
        gerador = lambda parar: linhas_seguir(args.seguir, parar, desde=desde, segundos=args.segundos)
        desc = f"seguindo {args.seguir} (esperando uma captura nova)"
    else:
        gerador = lambda parar: linhas_porta(args.porta, args.baud, args.salvar, parar, args.segundos)
        desc = f"porta {args.porta} a {args.baud} bit/s"

    if args.sem_janela:
        parar = threading.Event()
        try:
            rodar_texto(gerador(parar), p)
        except KeyboardInterrupt:
            parar.set()
            print("resumo: " + p.resumo())
        except OSError as e:
            print(f"não deu para ler a fonte: {e}")
            return 2
        return 1 if (p.contagem["descartadas"] or p.modelo_errado) else 0
    rodar_janela(gerador, p, desc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
