"""
Configurações centralizadas do pipeline de processamento de sinais.

Evita números mágicos dispersos e garante consistência entre as etapas de
decimação, extração de features e o futuro firmware em C.

Princípio: aqui ficam **escolhas**, não **resultados**. Valores que são
consequência de uma escolha (número de coeficientes do FIR, tamanho da FFT)
são calculados a partir dela, nunca copiados como literal — senão, no dia em
que alguém mudar a taxa de trabalho, eles continuam com o valor antigo e nada
acusa o erro.
"""

# ==========================================
# Amostragem e decimação
# ==========================================
FS_ORIGINAL = 51200      # Hz — taxa do dataset Jung et al. (2023)
FS_TRABALHO = 12800      # Hz — ver reports/decimation/

FATOR_DECIMACAO = FS_ORIGINAL // FS_TRABALHO   # 4

assert FS_ORIGINAL % FS_TRABALHO == 0, (
    f"FS_TRABALHO={FS_TRABALHO} não divide FS_ORIGINAL={FS_ORIGINAL}. "
    "A decimação por fator inteiro é requisito de arm_fir_decimate_f32 "
    "(CMSIS-DSP): uma taxa não divisível exigiria interpolador + decimador "
    "em cascata no STM32. Ver o Registro de Decisões no Notion."
)

# ==========================================
# Anti-aliasing
# ==========================================
FIR_ATTENUATION = 60                    # dB na banda de rejeição (escolha)
FIR_CUTOFF = 0.45 * FS_TRABALHO         # Hz — margem de 10% até Nyquist (5760 Hz)

# O número de coeficientes NÃO é fixado aqui: ele é saída do projeto de Kaiser,
# calculado por dsp.design_decimation() a partir da atenuação e da largura de
# transição. Para referência, a 12.800 Hz o projeto resulta em 147 taps.

# ==========================================
# Extração de features (MFCC)
# ==========================================
MFCC_WINDOW_MS = 25      # ms
MFCC_HOP_MS = 10         # ms
MFCC_N_MELS = 20         # número de bandas Mel
MFCC_N_COEFS = 13        # coeficientes mantidos após a DCT
MFCC_FMIN = 20.0         # Hz — piso do banco de Mel
MFCC_FMAX = FS_TRABALHO / 2          # Hz — Nyquist da taxa de trabalho (6400 Hz)
MFCC_JANELA = "hann"     # janela aplicada antes da FFT

# Derivados (não editar: mude os valores acima)
WINDOW_LENGTH = int(FS_TRABALHO * MFCC_WINDOW_MS / 1000)   # 320 amostras
HOP_LENGTH = int(FS_TRABALHO * MFCC_HOP_MS / 1000)         # 128 amostras
N_FFT = 1 << (WINDOW_LENGTH - 1).bit_length()              # 512

# EM ABERTO — ver tarefa "Caracterizar a natureza da assinatura acústica":
# o banco de Mel comprime a região de alta frequência, e a energia discriminante
# das classes incipientes (0,3 mm) se estende até ~6 kHz, contra ~2,3 kHz das
# severas. Conferir se MFCC_N_MELS = 20 resolve adequadamente a faixa de 2–6 kHz,
# ou se convém aumentar o número de bandas ou elevar MFCC_FMIN.

# ==========================================
# Classificação
# ==========================================
SEGMENTO_S = 1.0         # duração da unidade de classificação

CLASSES = ("normal", "bpfi_0.3mm", "bpfi_1.0mm", "bpfo_0.3mm", "bpfo_1.0mm")
BINARIO = {
    "normal": "normal",
    "bpfi_0.3mm": "falha",
    "bpfi_1.0mm": "falha",
    "bpfo_0.3mm": "falha",
    "bpfo_1.0mm": "falha",
}

# =====================================================
# Protocolo de validação (Registro de Decisões, 24/09)
# =====================================================
# Com uma única gravação contínua por classe, qualquer divisão dentro dela deixa
# treino e teste no mesmo registro. O protocolo que contorna isso — blocos
# temporais (A) e gravação de falha deixada de fora (B) — está implementado em
# validation/particao.py. A partição é gerada UMA vez pelo
# pipeline/03_make_splits.py e gravada em data/processed/splits/splits.json;
# todo experimento lê esse arquivo. A meta de 85 % é a acurácia balanceada
# média do Protocolo B.
SEGMENTOS_POR_BLOCO = 10   # segmentos de SEGMENTO_S por bloco temporal
SEGMENTOS_DESCARTE = 1     # faixa de descarte: segmentos de treino vizinhos a um
                           # bloco de teste, na mesma gravação, saem daquele fold
SEMENTE = 20260925         # só para controles aleatórios (permutação de rótulos)
                           # e treino de modelos; a partição é determinística

# Derivado (não editar)
def amostras_por_segmento(fs: float) -> int:
    """Amostras de um segmento de SEGMENTO_S a uma taxa `fs` (a de trabalho ou
    outra, nos estudos que comparam taxas sem mudar a de produção)."""
    return int(round(fs * SEGMENTO_S))


AMOSTRAS_POR_SEGMENTO = amostras_por_segmento(FS_TRABALHO)   # 12800

# A sobra do fim de cada gravação que não completa um segmento é descartada.
# Com a decimação atual, cada gravação tem 767.982 amostras: 59 segmentos
# completos, blocos de 10, 10, 10, 10, 10 e 9.

# ==========================================
# Aumento de dados (só no treino de cada fold)
# ==========================================
# Implementado em augmentation/. Cada variante de um segmento de treino é uma
# janela de 1 s tirada da mesma gravação, possivelmente deslocada, estirada no
# tempo e com ruído somado. As variantes são geradas pelo 04 (--aumento N) e o
# run_protocol só usa, em cada fold, as que leem EXCLUSIVAMENTE amostras de
# segmentos de treino daquele fold — a faixa de descarte continua valendo.
AUMENTO_DESLOC_MAX_S = 0.4          # s — deslocamento máximo da janela, para cada lado
AUMENTO_ESTIR_TAXAS = (0.95, 1.05)  # taxa do estiramento (<1 alonga, >1 comprime)
AUMENTO_ESTIR_MODO = "tempo"        # "tempo" (phase vocoder, preserva o espectro) ou
                                    # "velocidade" (reamostragem, escala as frequências)
                                    # — ver augmentation/transformacoes.py
AUMENTO_SNR_DB = (20.0, 35.0)       # dB — faixa da relação sinal-ruído do ruído branco
AUMENTO_PV_NFFT = 512               # amostras — quadro do phase vocoder (40 ms)
AUMENTO_PV_HOP = 128                # amostras — passo do phase vocoder (10 ms)
AUMENTO_TECNICAS = ("deslocamento", "estiramento", "ruido")

assert AUMENTO_DESLOC_MAX_S < SEGMENTOS_DESCARTE * SEGMENTO_S, (
    "o deslocamento máximo tem que ser menor que a faixa de descarte; senão "
    "quase toda variante de borda do treino seria recusada pelo run_protocol"
)

# ==========================================
# Caminhos dos dados por taxa
# ==========================================
# A taxa de trabalho usa os caminhos de sempre. Outra taxa (estudo que compara
# taxas sem mudar a de produção) tem PCM, partição e features próprios, em
# caminhos separados, para nunca sobrescrever os de FS_TRABALHO.
def dir_pcm_decimado(fs: float) -> str:
    return f"data/processed/pcm_decimated/{int(fs)}"


def arquivo_splits(fs: float) -> str:
    if int(fs) == FS_TRABALHO:
        return "data/processed/splits/splits.json"
    return f"data/processed/splits/splits_{int(fs)}.json"


def dir_features(fs: float) -> str:
    if int(fs) == FS_TRABALHO:
        return "data/processed/features"
    return f"data/processed/features/{int(fs)}"


# ==========================================
# Formato dos dados
# ==========================================
INT16_FULL = 32767.0     # fundo de escala do PCM int16
PCM_DTYPE = "<i2"        # int16 little-endian, sem cabeçalho