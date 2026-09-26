import numpy as np
import dsp  # type: ignore

def test_normalizacao_silencio():
    """Garante que um segmento puramente silencioso não cause divisão por zero nem estoure."""
    x_zero = np.zeros(12800)
    x_norm = dsp.normalizar_rms_clipe(x_zero)
    assert np.all(x_norm == 0), "Sinal de zeros deve continuar zero após normalização"

def test_invariancia_ganho_sem_c0():
    """Confere se as features sem c0 são idênticas com ganho 1 e ganho 10."""
    np.random.seed(42)
    x = np.random.randn(12800)

    fs = 12800
    m1 = dsp.mfcc(x, fs)
    m10 = dsp.mfcc(x * 10.0, fs)

    m1_sem_c0 = m1[:, 1:]
    m10_sem_c0 = m10[:, 1:]

    np.testing.assert_allclose(m1_sem_c0, m10_sem_c0, atol=1e-5)
