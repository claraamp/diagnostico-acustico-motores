"""
Configurações centralizadas do pipeline de processamento de sinais.
Evita números mágicos dispersos e garante consistência entre as etapas
de decimação, extração de features e o futuro firmware em C.
"""

# ==========================================
# Amostragem e Anti-aliasing
# ==========================================
FS_ORIGINAL = 51200      # Hz
FS_TRABALHO = 12800      # Hz (Fator de decimação = 4)

FIR_TAPS = 147           # FIR Kaiser
FIR_CUTOFF = 5760        # Hz
FIR_ATTENUATION = 60     # dB

# ==========================================
# Extração de Features (MFCC)
# ==========================================
MFCC_WINDOW_MS = 25      # ms
MFCC_HOP_MS = 10         # ms
MFCC_N_MELS = 20         # Número de bandas Mel
MFCC_N_COEFS = 13        # Número de coeficientes mantidos

# Derivados em número de amostras (calculados automaticamente)
WINDOW_LENGTH = int(FS_TRABALHO * (MFCC_WINDOW_MS / 1000.0))  # 320 amostras
HOP_LENGTH = int(FS_TRABALHO * (MFCC_HOP_MS / 1000.0))        # 128 amostras
N_FFT = 512              # Próxima potência de 2 superior a 320 para otimizar a FFT