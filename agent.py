"""Agente de triagem de avaliações de clientes de delivery.

Para cada avaliação, o agente devolve:
- sentimento: positivo | neutro | negativo
- categoria: entrega | comida | embalagem | atendimento | preco | app | elogio | outro
- urgencia: baixa | media | alta
- resposta_sugerida: texto curto e empático, pronto para revisão humana

Dois provedores:
- "gemini": usa a API do Google Gemini (plano gratuito) -> precisa de GEMINI_API_KEY
- "demo": classificador por regras, roda offline e sem chave (bom para testar a interface)
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, asdict

SENTIMENTOS = ("positivo", "neutro", "negativo")
CATEGORIAS = ("entrega", "comida", "embalagem", "atendimento", "preco", "app", "elogio", "outro")
URGENCIAS = ("baixa", "media", "alta")

PROMPT = """Você é um analista de experiência do cliente de um app de delivery.
Analise a avaliação abaixo e responda APENAS com um JSON válido, sem texto extra, com as chaves:
- "sentimento": um de {sentimentos}
- "categoria": um de {categorias} (o principal assunto da avaliação; se o sentimento for positivo, use sempre "elogio")
- "urgencia": um de {urgencias} (alta = risco à saúde/segurança, cobrança indevida ou cliente muito irritado)
- "resposta_sugerida": resposta em português do Brasil, empática, com no máximo 2 frases, sem prometer reembolso.
  NÃO afirme que alguma ação já foi tomada ou está em andamento (nada de "já acionamos", "já estamos resolvendo");
  use apenas intenções, como "vamos encaminhar" ou "vamos verificar".

Avaliação (nota {nota}/5): "{texto}"
"""

PROMPT_LOTE = """Você é um analista de experiência do cliente de um app de delivery.
Analise TODAS as avaliações numeradas abaixo e responda APENAS com um JSON array válido, sem texto extra,
com um objeto por avaliação, na mesma ordem. Cada objeto tem as chaves:
- "id": o número da avaliação
- "sentimento": um de {sentimentos}
- "categoria": um de {categorias} (o principal assunto da avaliação; se o sentimento for positivo, use sempre "elogio")
- "urgencia": um de {urgencias} (alta = risco à saúde/segurança, cobrança indevida ou cliente muito irritado)
- "resposta_sugerida": resposta em português do Brasil, empática, com no máximo 2 frases, sem prometer reembolso,
  mencionando o caso específico da avaliação.
  NÃO afirme que alguma ação já foi tomada ou está em andamento (nada de "já acionamos", "já estamos resolvendo");
  use apenas intenções, como "vamos encaminhar" ou "vamos verificar".

Avaliações:
{avaliacoes}
"""


@dataclass
class Resultado:
    sentimento: str
    categoria: str
    urgencia: str
    resposta_sugerida: str
    origem: str = "demo"  # "gemini" quando o LLM respondeu; "demo" quando foram as regras
    erro: str = ""  # motivo da falha do Gemini, se houve queda para o modo demo
    modelo: str = ""  # modelo do Gemini que respondeu

    def to_dict(self) -> dict:
        return asdict(self)


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------- modo demo
# A ordem importa: a primeira categoria que casar vence.
_REGRAS_CATEGORIA = {
    "preco": ["caro", "preco", "taxa", "cobrar", "cobranca", "cobrado", "valor"],
    "app": ["app", "aplicativo", "travou", "bug", "erro", "pagamento", "cupom"],
    "embalagem": ["embalagem", "vazou", "derram", "amassad", "lacre", "aberto", "aberta"],
    "entrega": ["atras", "demor", "entregador", "entrega", "nao chegou", "esperei"],
    "atendimento": ["atendimento", "suporte", "reembolso", "ninguem respond", "grosseir", "atendente"],
    "comida": ["frio", "fria", "gosto", "cru", "salgad", "insipid", "sabor", "queimad", "estragad", "azedo",
               "delicios", "cabelo", "nojo", "alergi", "camarao"],
}
_NEGATIVAS = ["pessim", "horrivel", "pior", "frio", "fria", "atras", "demor", "vazou", "derram", "caro",
              "nao chegou", "ninguem", "erro", "travou", "cru", "estragad", "grosseir", "decepc", "nunca mais", "absurd"]
_POSITIVAS = ["otim", "excelente", "perfeito", "perfeita", "delicios", "rapido", "recomendo", "amei", "adorei", "maravilh", "nota 10"]
_ALTA = ["estragad", "passei mal", "intoxic", "cabelo", "inseto", "cobrado duas", "cobranca indevida", "nunca mais", "absurd", "alergi", "perigoso"]

_RESPOSTAS = {
    "entrega": "Sentimos muito pelo atraso na sua entrega. Já repassamos seu relato ao time responsável para evitar que isso se repita.",
    "comida": "Lamentamos que o pedido não tenha atendido sua expectativa. Vamos compartilhar seu feedback com o restaurante parceiro.",
    "embalagem": "Sentimos muito pelo problema com a embalagem. Vamos alertar o restaurante para reforçar o cuidado no envio.",
    "atendimento": "Pedimos desculpas pela experiência com o atendimento. Seu caso será encaminhado para revisão pela nossa equipe.",
    "preco": "Obrigado por nos contar sobre os valores cobrados. Vamos analisar o seu caso com atenção.",
    "app": "Sentimos muito pela falha no aplicativo. Nosso time técnico já foi avisado para investigar o problema.",
    "elogio": "Muito obrigado pelo carinho! Ficamos felizes que tenha gostado e vamos repassar o elogio ao restaurante e ao entregador.",
    "outro": "Obrigado pelo seu feedback. Ele nos ajuda a melhorar a cada pedido.",
}


def classificar_demo(texto: str, nota: int | None = None) -> Resultado:
    t = _normalizar(texto)
    neg = sum(p in t for p in _NEGATIVAS)
    pos = sum(p in t for p in _POSITIVAS)

    if nota is not None and nota >= 4 and neg <= pos:
        sentimento = "positivo"
    elif nota is not None and nota <= 2:
        sentimento = "negativo"
    elif neg > pos:
        sentimento = "negativo"
    elif pos > neg:
        sentimento = "positivo"
    else:
        sentimento = "neutro"

    categoria = "outro"
    for cat, palavras in _REGRAS_CATEGORIA.items():
        if any(p in t for p in palavras):
            categoria = cat
            break
    if sentimento == "positivo":
        categoria = "elogio"

    urgencia = "baixa"
    if sentimento == "negativo":
        urgencia = "media"
    if any(p in t for p in _ALTA):
        urgencia = "alta"

    return Resultado(sentimento, categoria, urgencia, _RESPOSTAS[categoria])


# ------------------------------------------------------------- modo gemini
def _extrair_json(texto: str) -> dict:
    texto = texto.strip()
    texto = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
    return json.loads(texto)


def _validar(dados: dict) -> Resultado:
    """Garante que a saída do LLM respeita os valores permitidos."""
    sentimento = str(dados.get("sentimento", "neutro")).lower()
    categoria = str(dados.get("categoria", "outro")).lower()
    urgencia = str(dados.get("urgencia", "baixa")).lower().replace("é", "e")
    return Resultado(
        sentimento if sentimento in SENTIMENTOS else "neutro",
        categoria if categoria in CATEGORIAS else "outro",
        urgencia if urgencia in URGENCIAS else "baixa",
        str(dados.get("resposta_sugerida", "")).strip() or _RESPOSTAS["outro"],
    )


MODELO_PADRAO = "gemini-3.8-flash"  # o Google aposenta modelos antigos: pode ser trocado na barra lateral do app


def classificar_gemini(texto: str, nota: int | None = None, api_key: str | None = None, modelo: str | None = None) -> Resultado:
    from google import genai  # import tardio: só é necessário neste modo

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Defina GEMINI_API_KEY (veja o README) ou use o modo demo.")
    modelo = modelo or os.getenv("GEMINI_MODEL", MODELO_PADRAO)
    from google.genai import types

    # timeout de 60 s por chamada (também vira o prazo no servidor): evita ficar "girando" para sempre
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
    prompt = PROMPT.format(
        sentimentos=", ".join(SENTIMENTOS),
        categorias=", ".join(CATEGORIAS),
        urgencias=", ".join(URGENCIAS),
        nota=nota if nota is not None else "n/d",
        texto=texto.replace('"', "'"),
    )
    resp = client.models.generate_content(
        model=modelo,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    resultado = _validar(_extrair_json(resp.text))
    resultado.origem = "gemini"
    return resultado


def _resultados_do_lote(dados, itens: list[tuple[str, int | None]]) -> list[Resultado]:
    """Casa a resposta do LLM (lista de objetos com 'id') com as avaliações enviadas.
    Se uma avaliação faltar ou vier inválida, ela cai no modo demo com o motivo registrado."""
    if not isinstance(dados, list):
        raise ValueError("resposta do modelo não é uma lista JSON")
    por_id = {}
    for d in dados:
        if isinstance(d, dict):
            try:
                por_id[int(d.get("id"))] = d
            except (TypeError, ValueError):
                continue
    resultados = []
    for i, (texto, nota) in enumerate(itens, start=1):
        if i in por_id:
            r = _validar(por_id[i])
            r.origem = "gemini"
        else:
            r = classificar_demo(texto, nota)
            r.erro = f"ValueError: o modelo não devolveu a avaliação {i} do lote"
        resultados.append(r)
    return resultados


def classificar_lote(
    itens: list[tuple[str, int | None]], provedor: str = "demo", api_key: str | None = None, modelo: str | None = None
) -> list[Resultado]:
    """Classifica várias avaliações em UMA chamada à API (economiza cota e tempo)."""
    if provedor != "gemini":
        return [classificar_demo(t, n) for t, n in itens]
    try:
        from google import genai
        from google.genai import types

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Defina GEMINI_API_KEY (veja o README) ou use o modo demo.")
        modelo = modelo or os.getenv("GEMINI_MODEL", MODELO_PADRAO)
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90000))
        avaliacoes = "\n".join(
            f'[{i}] (nota {n if n is not None else "n/d"}/5) "{t.replace(chr(34), chr(39))}"'
            for i, (t, n) in enumerate(itens, start=1)
        )
        prompt = PROMPT_LOTE.format(
            sentimentos=", ".join(SENTIMENTOS),
            categorias=", ".join(CATEGORIAS),
            urgencias=", ".join(URGENCIAS),
            avaliacoes=avaliacoes,
        )
        resp = client.models.generate_content(
            model=modelo,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        resultados = _resultados_do_lote(_extrair_json(resp.text), itens)
        for r in resultados:
            if r.origem == "gemini":
                r.modelo = modelo
        return resultados
    except RuntimeError:
        raise
    except Exception as e:  # falha de rede/cota/JSON: não derruba a triagem, mas é registrada
        erro = f"{type(e).__name__}: {str(e)[:1200]}"
        resultados = [classificar_demo(t, n) for t, n in itens]
        for r in resultados:
            r.erro = erro
        return resultados


_EVITAR = ("image", "tts", "live", "audio", "embed", "robotics", "computer", "veo", "imagen", "native")
_ERROS_DE_MODELO = ("429", "404", "RESOURCE_EXHAUSTED", "NOT_FOUND", "UNAVAILABLE", "503", "504", "DEADLINE")


def _filtrar_candidatos(nomes: list[str], preferido: str, maximo: int = 6) -> list[str]:
    """Modelos 'flash' de texto, os 'lite' primeiro, com o preferido no início da fila."""
    bons = [n for n in nomes if "flash" in n and not any(x in n for x in _EVITAR)]
    bons.sort(key=lambda n: ("lite" not in n, n))
    fila = [preferido] + [n for n in bons if n != preferido]
    return fila[:maximo]


def candidatos(api_key: str, preferido: str) -> list[str]:
    try:
        nomes = listar_modelos(api_key)
    except Exception:
        nomes = []
    return _filtrar_candidatos(nomes, preferido)


def classificar_lote_auto(itens: list[tuple[str, int | None]], api_key: str, modelos: list[str]) -> list[Resultado]:
    """Tenta cada modelo da fila até um responder; só troca de modelo em erros de cota,
    modelo inexistente ou indisponibilidade. Outros erros são devolvidos como estão."""
    resultados: list[Resultado] = []
    for m in modelos:
        resultados = classificar_lote(itens, "gemini", api_key, m)
        erro = next((r.erro for r in resultados if r.erro), "")
        if not erro or not any(c in erro for c in _ERROS_DE_MODELO):
            return resultados
    return resultados


def listar_modelos(api_key: str) -> list[str]:
    """Modelos que a chave consegue usar para gerar texto (nomes prontos para colar no campo Modelo)."""
    from google import genai

    client = genai.Client(api_key=api_key)
    nomes = []
    for m in client.models.list():
        acoes = getattr(m, "supported_actions", None) or []
        if not acoes or "generateContent" in acoes:
            nomes.append(str(m.name).removeprefix("models/"))
    return sorted(nomes)


def classificar(texto: str, nota: int | None = None, provedor: str = "demo", api_key: str | None = None, modelo: str | None = None) -> Resultado:
    if provedor == "gemini":
        try:
            return classificar_gemini(texto, nota, api_key, modelo)
        except RuntimeError:
            raise
        except Exception as e:  # falha de rede/cota/JSON: não derruba a triagem, mas é registrada
            resultado = classificar_demo(texto, nota)
            resultado.erro = f"{type(e).__name__}: {str(e)[:1200]}"
            return resultado
    return classificar_demo(texto, nota)
