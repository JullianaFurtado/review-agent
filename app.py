"""Dashboard Streamlit do agente de triagem de avaliações.

Rodar: streamlit run app.py
"""

import os
import time

import pandas as pd
import streamlit as st

from agent import MODELO_PADRAO, candidatos, classificar, classificar_lote, classificar_lote_auto, listar_modelos

MINUTOS_MANUAIS_POR_AVALIACAO = 2  # premissa ajustável na barra lateral
TAMANHO_LOTE = 10  # avaliações por chamada à API: economiza a cota gratuita (30 avaliações = 3 chamadas)

st.set_page_config(page_title="Triagem de avaliações com IA", page_icon="📝", layout="wide")
st.title("📝 Agente de triagem de avaliações")
st.caption("Classifica sentimento, categoria e urgência e sugere uma resposta para cada avaliação de cliente.")

# ------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("Configuração")
    provedor_label = st.radio("Modo", ["Demo (sem chave)", "Gemini (LLM)"])
    provedor = "gemini" if provedor_label.startswith("Gemini") else "demo"
    api_key = None
    modelo = MODELO_PADRAO
    if provedor == "gemini":
        api_key = st.text_input("GEMINI_API_KEY", value=os.getenv("GEMINI_API_KEY", ""), type="password")
        st.caption("Crie uma chave gratuita em aistudio.google.com/apikey")
        modelo = st.text_input("Modelo do Gemini", value=os.getenv("GEMINI_MODEL", MODELO_PADRAO))
        st.caption("Se der erro de modelo ou 504, clique em 'Listar modelos' e teste outro.")
        c_a, c_b = st.columns(2)
        if c_a.button("Testar conexão"):
            if not api_key:
                st.error("Cole a chave primeiro.")
            else:
                inicio = time.time()
                with st.spinner("Testando (até 60 s)..."):
                    r = classificar("O pedido atrasou e chegou frio.", 2, "gemini", api_key, modelo)
                seg = time.time() - inicio
                if r.origem == "gemini":
                    st.success(f"Funcionou em {seg:.1f} s")
                else:
                    st.error(f"Falhou após {seg:.1f} s")
                    st.code(r.erro)
        if c_b.button("Listar modelos"):
            if not api_key:
                st.error("Cole a chave primeiro.")
            else:
                try:
                    st.code("\n".join(listar_modelos(api_key)))
                except Exception as e:
                    st.code(f"{type(e).__name__}: {str(e)[:300]}")
    minutos = st.number_input("Minutos por avaliação (triagem manual)", 1, 15, MINUTOS_MANUAIS_POR_AVALIACAO)
    arquivo = st.file_uploader("Usar meu CSV (colunas: id, nota, avaliacao)", type="csv")
    limite = st.slider("Quantidade de avaliações a processar", 5, 100, 30)

df = pd.read_csv(arquivo) if arquivo else pd.read_csv("data/reviews.csv")
df = df.head(limite)


def processar(textos: tuple, notas: tuple, provedor: str, api_key: str | None, modelo: str) -> list[dict]:
    """Classifica em lotes (uma chamada à API por lote), com barra de progresso.
    Após 2 lotes seguidos com falha, desiste da API e termina no modo demo
    (o aviso amarelo mostra o motivo)."""
    itens = list(zip(textos, notas))
    lotes = [itens[i : i + TAMANHO_LOTE] for i in range(0, len(itens), TAMANHO_LOTE)]
    barra = st.progress(0.0, text="Analisando...")
    resultados, falhas_seguidas, desistiu = [], 0, False
    fila = candidatos(api_key, modelo) if provedor == "gemini" and api_key else [modelo]
    for k, lote in enumerate(lotes, start=1):
        if provedor == "gemini" and not desistiu:
            # tenta os modelos da fila; o que funcionar passa a ser o primeiro nos próximos lotes
            rs = classificar_lote_auto(lote, api_key, fila)
            usado = next((r.modelo for r in rs if r.modelo), "")
            if usado:
                fila = [usado] + [m for m in fila if m != usado]
            falhas_seguidas = falhas_seguidas + 1 if any(r.erro for r in rs) else 0
            desistiu = falhas_seguidas >= 2
        else:
            rs = classificar_lote(lote, "demo", api_key, modelo)
        resultados.extend(r.to_dict() for r in rs)
        barra.progress(k / len(lotes), text=f"Analisando... lote {k}/{len(lotes)}")
    barra.empty()
    return resultados


if st.button("▶ Processar avaliações", type="primary"):
    if provedor == "gemini" and not api_key:
        st.error("Informe a GEMINI_API_KEY ou use o modo Demo.")
        st.stop()
    with st.spinner("Analisando..."):
        resultados = processar(tuple(df["avaliacao"]), tuple(df["nota"]), provedor, api_key, modelo)
    st.session_state["resultado"] = pd.concat([df.reset_index(drop=True), pd.DataFrame(resultados)], axis=1)
    st.session_state["provedor_usado"] = provedor

if "resultado" not in st.session_state:
    st.info("Clique em **Processar avaliações** para ver o dashboard.")
    st.dataframe(df, width="stretch", hide_index=True)
    st.stop()

res = st.session_state["resultado"]

# ------------------------------------------------- aviso de origem dos resultados
if st.session_state.get("provedor_usado") == "gemini":
    n_gemini = int((res["origem"] == "gemini").sum())
    if n_gemini == len(res):
        usados = ", ".join(sorted(set(res["modelo"])))
        st.success(f"Todas as {len(res)} avaliações foram classificadas pelo Gemini (modelo: {usados}).")
    else:
        falhas = res[res["origem"] != "gemini"]
        st.warning(
            f"O Gemini classificou {n_gemini} de {len(res)} avaliações. "
            f"As outras {len(falhas)} caíram no modo demo (regras simples)."
        )
        erros = falhas["erro"].replace("", pd.NA).dropna()
        if not erros.empty:
            st.code(erros.iloc[0])
else:
    st.info("Resultados do modo demo (regras simples, sem IA).")

# --------------------------------------------------------------- indicadores
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avaliações analisadas", len(res))
c2.metric("% negativas", f"{(res['sentimento'] == 'negativo').mean():.0%}")
c3.metric("Urgência alta", int((res["urgencia"] == "alta").sum()))
horas = len(res) * minutos / 60
c4.metric("Tempo de triagem economizado", f"{horas:.1f} h", help="Estimativa: avaliações × minutos manuais por avaliação")

# ------------------------------------------------------------------ gráficos
g1, g2 = st.columns(2)
with g1:
    st.subheader("Sentimento")
    st.bar_chart(res["sentimento"].value_counts())
with g2:
    st.subheader("Categorias de problema")
    st.bar_chart(res[res["sentimento"] != "positivo"]["categoria"].value_counts())

# ------------------------------------------------------------------- fila
st.subheader("Fila de atendimento (urgentes primeiro)")
ordem = {"alta": 0, "media": 1, "baixa": 2}
fila = res.sort_values("urgencia", key=lambda s: s.map(ordem))
filtro = st.multiselect("Filtrar urgência", ["alta", "media", "baixa"], default=["alta", "media"])
st.dataframe(
    fila[fila["urgencia"].isin(filtro)][["id", "nota", "avaliacao", "sentimento", "categoria", "urgencia", "resposta_sugerida"]],
    width="stretch",
    hide_index=True,
)
st.download_button("⬇ Baixar resultado (CSV)", res.to_csv(index=False).encode("utf-8"), "avaliacoes_classificadas.csv", "text/csv")
