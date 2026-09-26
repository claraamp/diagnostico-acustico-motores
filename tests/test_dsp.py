import sys
from pathlib import Path
import numpy as np

# Adiciona a pasta 'scripts' ao path para o pytest conseguir achá-la
caminho_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(caminho_scripts))

import dsp  # type: ignore

def test_normalizacao_silencio():
    """Garante que um segmento puramente silencioso não cause divisão por zero nem estoure."""
    x_zero = np.zeros(12800)
    x_norm = dsp.normalizar_rms_clipe(x_zero)
    assert np.all(x_norm == 0), "Sinal de zeros deve continuar zero após normalização"

def test_invariancia_ganho_sem_c0():
    """Confere se as features sem c0 são idênticas com ganho 1 e ganho 10."""
    np.random.seed(42)
    # Sinal aleatório simulando áudio (ganho 1)
    x = np.random.randn(12800) 
    
    # Extrai MFCC do sinal normal e do sinal amplificado (ganho 10)
    fs = 12800
    m1 = dsp.mfcc(x, fs)
    m10 = dsp.mfcc(x * 10.0, fs)
    
    # Retira o c0 (primeira coluna) de ambos
    m1_sem_c0 = m1[:, 1:]
    m10_sem_c0 = m10[:, 1:]
    
    # Compara se os coeficientes de c1 a c12 ficaram iguais (margem de erro de precisão float)
    np.testing.assert_allclose(m1_sem_c0, m10_sem_c0, atol=1e-5)