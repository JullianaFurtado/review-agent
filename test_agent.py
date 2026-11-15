import pandas as pd

from agent import (
    CATEGORIAS,
    SENTIMENTOS,
    URGENCIAS,
    _extrair_json,
    _validar,
    classificar,
    classificar_demo,
)


def test_demo_negativo_entrega():
    r = classificar_demo("O pedido atrasou muito e chegou frio", 1)
    assert r.sentimento == "negativo"
    assert r.urgencia in ("media", "alta")


def test_demo_positivo_vira_elogio():
    r = classificar_demo("Excelente, comida deliciosa e entrega rápida!", 5)
    assert r.sentimento == "positivo"
    assert r.categoria == "elogio"
    assert r.urgencia == "baixa"


def test_demo_urgencia_alta():
    r = classificar_demo("Passei mal, a comida estava estragada", 1)
    assert r.urgencia == "alta"


def test_validar_corrige_valores_invalidos():
    r = _validar({"sentimento": "maluco", "categoria": "x", "urgencia": "urgentissima", "resposta_sugerida": ""})
    assert r.sentimento == "neutro" and r.categoria == "outro" and r.urgencia == "baixa"
    assert r.resposta_sugerida


def test_extrair_json_com_cerca_de_codigo():
    assert _extrair_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_dataset_inteiro_produz_valores_validos():
    df = pd.read_csv("data/reviews.csv")
    for texto, nota in zip(df["avaliacao"], df["nota"]):
        r = classificar(texto, nota, "demo")
        assert r.sentimento in SENTIMENTOS
        assert r.categoria in CATEGORIAS
        assert r.urgencia in URGENCIAS
