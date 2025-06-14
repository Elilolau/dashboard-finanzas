import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CSS para Fira Sans en toda la app ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Fira Sans', sans-serif !important;
    }
    .stTable, .stSelectbox, .stRadio, .stButton, .stMarkdown, .stCaption, .stDataFrame {
        font-family: 'Fira Sans', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Paleta de colores corporativa ---
COLORES = [
    "#BF1B18",  # 50%
    "#FBAF3A",  # 20%
    "#FF7502",  # 20%
    "#1E87C1",  # 5%
    "#50D2FF"   # 5%
]

# --- 3. Cargar datos ---
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/capitalia-datos-publicos/empresas.csv"
    return pd.read_csv(url, sep=',', encoding="utf-8")

df = load_data()



# --- 4. Variables y nombres ---
numericos = [
    "ingresos", "utilidad_neta", "total_de_activos", "total_pasivos",
    "patrimonio_total", "flujo_de_caja_libre", "total_capex", "ebitda",
    "ROE", "ROA", "deuda_financiera", "margen_ebitda", "deuda_/_activos",
    "crecimiento_ingresos"
]
nombres = {
    "Ingresos": "ingresos",
    "Utilidad Neta": "utilidad_neta",
    "Total de Activos": "total_de_activos",
    "Total Pasivos": "total_pasivos",
    "Patrimonio Total": "patrimonio_total",
    "Flujo de Caja Libre": "flujo_de_caja_libre",
    "Total Capex": "total_capex",
    "EBITDA": "ebitda",
    "ROE": "ROE",
    "ROA": "ROA",
    "Deuda Financiera": "deuda_financiera",
    "Margen EBITDA": "margen_ebitda",
    "Deuda / Activos": "deuda_/_activos",
    "Crecimiento Ingresos": "crecimiento_ingresos"
}
DIVISOR = 1_000  # Para mostrar valores en Miles de millones
variables_porcentaje = {"ROE", "ROA", "margen_ebitda", "deuda_/_activos", "crecimiento_ingresos"}

# --- LOGO ---
st.markdown(
    "<div style='text-align:center'><img src='https://github.com/Elilolau/dashboard-finanzas/main/logo_capitalia.png' width='220'/></div>",
    unsafe_allow_html=True
)

# --- 5. Definir Tabs ---
tab1, tab_sumas, tab2, tab3 = st.tabs(["Conteo", "Sumas", "Relación entre Variables", "Correlaciones"])

# --------- TAB 1: CONTEO -----------
with tab1:
    st.markdown("<h1 style='text-align:center; font-family: Fira Sans, sans-serif;'>Distribución de Empresas en SuperSociedades</h1>", unsafe_allow_html=True)
    ultimo_anio = df["anio"].max()
    df_anio = df[df["anio"] == ultimo_anio]
    analisis_opcion = st.radio(
        "¿Qué empresas quieres analizar?",
        options=["Top 1,000 empresas por ingresos", "Todas las empresas"]
    )

    if analisis_opcion == "Top 1,000 empresas por ingresos":
        df_filtradas = df_anio[df_anio["ingresos"] > 0].sort_values("ingresos", ascending=False).head(1000)
    else:
        df_filtradas = df_anio[df_anio["ingresos"] > 0].copy()

    indicador_opcion = st.selectbox(
        "Selecciona la variable que quieras analizar:",
        list(nombres.keys()),
    )
    col_num = nombres[indicador_opcion]

    departamentos = ["Todos los departamentos"] + sorted(df_filtradas["departamento"].dropna().unique())
    departamento_seleccionado = st.selectbox("Selecciona el Departamento:", departamentos)

    if departamento_seleccionado != "Todos los departamentos":
        df_filtradas = df_filtradas[df_filtradas["departamento"] == departamento_seleccionado]

    industrias = ["Todas las industrias"] + sorted(df_filtradas["industria"].dropna().unique())
    industria_seleccionada = st.selectbox("Selecciona la Industria:", industrias)

    if industria_seleccionada != "Todas las industrias":
        df_filtradas = df_filtradas[df_filtradas["industria"] == industria_seleccionada]

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Histograma logarítmico y tabla de rangos ---
    valores = df_filtradas[col_num]
    valores = valores[valores > 0]

    if valores.empty:
        st.warning("No hay empresas en esta combinación de filtros.")
    else:
        valores_log = np.log10(valores)

        if len(valores) > 50:
            log_min = valores_log.min()
            log_max = valores_log.max()
            num_bins = 10
            bin_edges = np.linspace(log_min, log_max, num_bins + 1)
            bins = 10 ** bin_edges
        else:
            log_min = np.floor(valores_log.min())
            log_max = np.ceil(valores_log.max())
            num_bins = int(log_max - log_min)
            bin_edges = np.linspace(log_min, log_max, num_bins + 1)
            bins = 10 ** bin_edges

        segmentos = pd.cut(
            valores, 
            bins=bins, 
            labels=range(1, len(bins)), 
            include_lowest=True, 
            right=False
        )
        df_segmentos_plot = pd.DataFrame({
            "Segmento": segmentos.astype("Int64")
        }).dropna()

        # --- Títulos y subtítulos
        filtros = []
        if departamento_seleccionado != "Todos los departamentos":
            filtros.append(f"Departamento: {departamento_seleccionado}")
        if industria_seleccionada != "Todas las industrias":
            filtros.append(f"Industria: {industria_seleccionada}")
        filtros.append("Top 1,000 empresas por ingresos" if analisis_opcion == "Top 1,000 empresas por ingresos" else "Todas las empresas")
        st.markdown(
            f"<h2 style='text-align:center; font-family: Fira Sans, sans-serif;'>Histograma de {indicador_opcion} ({ultimo_anio})</h2>", 
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='text-align:center; color:#666; font-size:18px; font-family: Fira Sans, sans-serif;'>" +
            "<br>".join(filtros) +
            "</div>", unsafe_allow_html=True
        )

        fig = px.histogram(
            df_segmentos_plot,
            x="Segmento",
            labels={'Segmento': 'Segmento', 'count': 'Número de Empresas'},
            category_orders={"Segmento": list(range(1, len(bins)))},
            color_discrete_sequence=[COLORES[0]]
        )
        fig.update_layout(
            title_text="",
            margin=dict(t=40, b=60),
            font=dict(family="Fira Sans, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(
            title="Segmento",
            tickmode='linear',
            tick0=1,
            dtick=1
        )
        fig.update_yaxes(
            title="Número de Empresas"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Tabla de segmentos sincronizada con los bins ---
        conteo_segmentos = []
        for i in range(len(bins) - 1):
            if col_num in variables_porcentaje:
                min_val = bins[i] * 100
                max_val = bins[i + 1] * 100
                rango = f"{min_val:,.2f}% – {max_val:,.2f}%"
                columna_rango = "Rango (%)"
            else:
                min_val = bins[i] / DIVISOR
                max_val = bins[i + 1] / DIVISOR
                rango = f"{min_val:,.0f} – {max_val:,.0f}"
                columna_rango = "Rango (Miles de millones de COP)"
            mask = (valores >= bins[i]) & (valores < bins[i + 1])
            if i == len(bins) - 2:
                mask = (valores >= bins[i]) & (valores <= bins[i + 1])
            empresas_en_bin = mask.sum()
            conteo_segmentos.append({
                "Segmento": i+1,
                columna_rango: rango,
                "Empresas": int(empresas_en_bin)
            })

        if col_num in variables_porcentaje:
            df_segmentos = pd.DataFrame(conteo_segmentos, columns=["Segmento", "Rango (%)", "Empresas"])
        else:
            df_segmentos = pd.DataFrame(conteo_segmentos, columns=["Segmento", "Rango (Miles de millones de COP)", "Empresas"])

        st.markdown("<h3 style='font-family: Fira Sans, sans-serif;'>Número de empresas por rango</h3>", unsafe_allow_html=True)
        st.dataframe(df_segmentos, hide_index=True)

        st.caption(
                f"Empresas filtradas por: "
                f"{'Top 1,000 por ingresos' if analisis_opcion == 'Top 1,000 empresas por ingresos' else 'Todas las empresas'}, "
                f"{'Departamento: ' + departamento_seleccionado if departamento_seleccionado != 'Todos los departamentos' else 'Todos los departamentos'}, "
                f"{'Industria: ' + industria_seleccionada if industria_seleccionada != 'Todas las industrias' else 'Todas las industrias'}. "
                f"Si hay más de 50 empresas en la muestra, los rangos corresponden a 10 segmentos logarítmicos iguales entre el mínimo y máximo."
            )

        if analisis_opcion == "Top 1,000 empresas por ingresos":
            top1000_2023 = df_filtradas.sort_values("ingresos", ascending=False).head(1000)
            empresas_top = top1000_2023["nit"].unique()
            df_2022 = df[(df["anio"] == (ultimo_anio - 1)) & (df["nit"].isin(empresas_top))].copy()
            resumen = pd.merge(
                top1000_2023,
                df_2022,
                on="nit",
                suffixes=('_2023', '_2022'),
                how='left'
            )

            if "razon_social_2023" in resumen.columns:
                resumen.rename(columns={"razon_social_2023": "razon_social"}, inplace=True)
            elif "razon_social_x" in resumen.columns:
                resumen.rename(columns={"razon_social_x": "razon_social"}, inplace=True)

            if "razon_social_2022" in resumen.columns:
                resumen.drop(columns=["razon_social_2022"], inplace=True)
            elif "razon_social_y" in resumen.columns:
                resumen.drop(columns=["razon_social_y"], inplace=True)

            if "razon_social" not in resumen.columns:
                resumen["razon_social"] = np.nan
            def crecimiento(a, b):
                try:
                    if pd.isna(a) or pd.isna(b) or b == 0:
                        return np.nan
                    return ((a - b) / b) * 100
                except Exception:
                    return np.nan

            resumen["Crec. Ingresos (%)"] = [
                crecimiento(row.get("ingresos_2023"), row.get("ingresos_2022"))
                for _, row in resumen.iterrows()
            ]
            resumen["Crec. Activos (%)"] = [
                crecimiento(row.get("total_de_activos_2023"), row.get("total_de_activos_2022"))
                for _, row in resumen.iterrows()
            ]
            resumen["Crec. Util. Neta (%)"] = [
                crecimiento(row.get("utilidad_neta_2023"), row.get("utilidad_neta_2022"))
                for _, row in resumen.iterrows()
            ]
            required_cols = [
                "razon_social",
                "ingresos_2023", "ingresos_2022", "Crec. Ingresos (%)",
                "total_de_activos_2023", "total_de_activos_2022", "Crec. Activos (%)",
                "utilidad_neta_2023", "utilidad_neta_2022", "Crec. Util. Neta (%)",
                "total_pasivos_2023", "total_pasivos_2022",
                "ROA_2023", "ROE_2023"
            ]

            for col in required_cols:
                if col not in resumen.columns:
                    resumen[col] = np.nan

            tabla_top = resumen[required_cols].copy()
            tabla_top.columns = [
                "Razón Social",
                "Ingresos 2023", "Ingresos 2022", "Crec. Ingresos (%)",
                "Activos 2023", "Activos 2022", "Crec. Activos (%)",
                "Utilidad Neta 2023", "Utilidad Neta 2022", "Crec. Util. Neta (%)",
                "Endeudamiento 2023", "Endeudamiento 2022",
                "ROA 2023", "ROE 2023"
            ]
            def miles_millones(x):
                try:
                    return f"{x/1e3:,.0f}"
                except:
                    return ""
            def porcentaje(x):
                try:
                    return f"{x:,.2f}%"
                except:
                    return ""
            for col in ["Ingresos 2023", "Ingresos 2022", "Activos 2023", "Activos 2022", "Utilidad Neta 2023", "Utilidad Neta 2022", "Endeudamiento 2023", "Endeudamiento 2022"]:
                tabla_top[col] = tabla_top[col].apply(miles_millones)
            for col in ["Crec. Ingresos (%)", "Crec. Activos (%)", "Crec. Util. Neta (%)", "ROA 2023", "ROE 2023"]:
                tabla_top[col] = tabla_top[col].apply(porcentaje)
            st.markdown("<br><h3 style='font-family: Fira Sans, sans-serif;'>Detalle Financiero Empresas Seleccionadas</h3>", unsafe_allow_html=True)
            st.dataframe(tabla_top, hide_index=True)

# --------- TAB 2: SUMAS -----------
with tab_sumas:
    st.markdown("<h1 style='text-align:center; font-family: Fira Sans, sans-serif;'>Distribución de Empresas en SuperSociedades</h1>", unsafe_allow_html=True)
    ultimo_anio = df["anio"].max()
    df_anio = df[df["anio"] == ultimo_anio]

    analisis_opcion = st.radio(
        "¿Qué empresas quieres analizar?",
        options=["Top 1,000 empresas por ingresos", "Todas las empresas"],
        key="radio_tab_sumas"
    )

    if analisis_opcion == "Top 1,000 empresas por ingresos":
        df_filtradas = df_anio[df_anio["ingresos"] > 0].sort_values("ingresos", ascending=False).head(1000)
    else:
        df_filtradas = df_anio[df_anio["ingresos"] > 0].copy()

    indicador_opcion = st.selectbox(
        "Selecciona la variable que quieras analizar:",
        list(nombres.keys()),
        key="variable_tab_sumas"
    )
    col_num = nombres[indicador_opcion]

    # Agrupar por industria y calcular suma o promedio según el tipo de variable
    if col_num in variables_porcentaje:
        # Eliminar outliers fuera del rango [-100%, 100%]
        df_pct = df_filtradas[(df_filtradas[col_num] <= 1) & (df_filtradas[col_num] >= -1)].copy()
        # Promedio para variables de porcentaje
        df_group = (
            df_pct.groupby("industria", as_index=False)[col_num]
            .mean()
            .rename(columns={col_num: "Valor"})
        )
        df_group["Valor"] = df_group["Valor"] * 100  # Para mostrar en %
        valor_label = "Promedio (%)"
        formato_valor = lambda x: f"{x:,.2f}%"
        subtitulo = f"Promedio de {indicador_opcion} por industria"
    else:
        # Suma para variables monetarias
        df_group = (
            df_filtradas.groupby("industria", as_index=False)[col_num]
            .sum()
            .rename(columns={col_num: "Valor"})
        )
        df_group["Valor"] = df_group["Valor"] / DIVISOR  # Para mostrar en miles de millones
        valor_label = "Suma (Miles de millones de COP)"
        formato_valor = lambda x: f"{x:,.0f}"
        subtitulo = f"Suma de {indicador_opcion} por industria"


    df_group = df_group.sort_values("Valor", ascending=False)
    df_group = df_group[df_group["industria"].notnull()]

    # --------- TÍTULOS ---------
    # Título principal
    st.markdown(
        f"<h2 style='text-align:center; font-family: Fira Sans, sans-serif;'>{indicador_opcion} ({ultimo_anio})</h2>",
        unsafe_allow_html=True
    )

    # Subtítulo con formato e info de filtros
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:18px; font-family: Fira Sans, sans-serif;'>" +
        f"{subtitulo}<br>" +
        ("Top 1,000 empresas por ingresos" if analisis_opcion == "Top 1,000 empresas por ingresos" else "Todas las empresas") +
        "</div>", unsafe_allow_html=True
    )

    # Gráfica de barras horizontales
    import plotly.express as px
    fig = px.bar(
        df_group,
        x="Valor",
        y="industria",
        orientation="h",
        color_discrete_sequence=[COLORES[0]],
        labels={"industria": "Industria", "Valor": valor_label},
        title="",
    )
    fig.update_layout(
        margin=dict(t=40, b=40),
        font=dict(family="Fira Sans, sans-serif"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis={'categoryorder':'total ascending'},
        xaxis_title=valor_label,
        yaxis_title="Industria"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar tabla de valores
    df_group["Valor"] = df_group["Valor"].apply(formato_valor)
    st.markdown(f"<h3 style='font-family: Fira Sans, sans-serif;'>{valor_label} por industria</h3>", unsafe_allow_html=True)
    st.dataframe(df_group.rename(columns={"industria": "Industria", "Valor": valor_label}), hide_index=True)

    
# --------- TAB 2: SCATTERPLOT ------------
with tab2:
    st.markdown("<h2 style='text-align:center; font-family: Fira Sans, sans-serif;'>Relación entre dos Indicadores Financieros</h2>", unsafe_allow_html=True)
    ultimo_anio = df["anio"].max()
    df_anio = df[df["anio"] == ultimo_anio]
    analisis_opcion2 = st.radio(
        "¿Qué empresas quieres analizar?",
        options=["Top 1,000 empresas por ingresos", "Todas las empresas"],
        key="radio_tab2"
    )
    if analisis_opcion2 == "Top 1,000 empresas por ingresos":
        df_base = df_anio[df_anio["ingresos"] > 0].sort_values("ingresos", ascending=False).head(1000).copy()
    else:
        df_base = df_anio[df_anio["ingresos"] > 0].copy()

    departamentos2 = ["Todos los departamentos"] + sorted(df_base["departamento"].dropna().unique())
    departamento_sel2 = st.selectbox("Selecciona el Departamento:", departamentos2, key="dep_tab2")
    if departamento_sel2 != "Todos los departamentos":
        df_base = df_base[df_base["departamento"] == departamento_sel2]
    industrias2 = ["Todas las industrias"] + sorted(df_base["industria"].dropna().unique())
    industria_sel2 = st.selectbox("Selecciona la Industria:", industrias2, key="ind_tab2")
    if industria_sel2 != "Todas las industrias":
        df_base = df_base[df_base["industria"] == industria_sel2]
    st.markdown("<br>", unsafe_allow_html=True)
    variables_disp = list(nombres.keys())
    var_x = st.selectbox("Selecciona variable para el eje X:", variables_disp, index=0, key="x_tab2")
    var_y = st.selectbox("Selecciona variable para el eje Y:", variables_disp, index=1, key="y_tab2")
    col_x = nombres[var_x]
    col_y = nombres[var_y]
    st.markdown("**Escala de los ejes**")
    col1, col2 = st.columns(2)
    log_x = col1.checkbox("Escala logarítmica en X", value=False, key="logx_tab2")
    log_y = col2.checkbox("Escala logarítmica en Y", value=False, key="logy_tab2")
    df_plot = df_base[[col_x, col_y, "razon_social", "industria", "anio"]].copy()
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_x, col_y])
    if df_plot.shape[0] > 50:
        q1_x, q99_x = df_plot[col_x].quantile([0.01, 0.99])
        q1_y, q99_y = df_plot[col_y].quantile([0.01, 0.99])
        mask = (df_plot[col_x] >= q1_x) & (df_plot[col_x] <= q99_x) & (df_plot[col_y] >= q1_y) & (df_plot[col_y] <= q99_y)
        df_plot = df_plot[mask]
    if log_x:
        df_plot = df_plot[df_plot[col_x] > 0]
    if log_y:
        df_plot = df_plot[df_plot[col_y] > 0]
    if not df_plot.empty:
        corr_pearson = df_plot[[col_x, col_y]].corr(method="pearson").iloc[0,1]
    else:
        corr_pearson = np.nan
    filtros_aplicados = []
    filtros_aplicados.append("Top 1,000 por ingresos" if analisis_opcion2 == "Top 1,000 empresas por ingresos" else "Todas las empresas")
    if departamento_sel2 != "Todos los departamentos":
        filtros_aplicados.append(f"Departamento: {departamento_sel2}")
    if industria_sel2 != "Todas las industrias":
        filtros_aplicados.append(f"Industria: {industria_sel2}")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:18px; font-family: Fira Sans, sans-serif;'>" +
        "<br>".join(filtros_aplicados) +
        "</div>", unsafe_allow_html=True
    )
    if not df_plot.empty:
        fig2 = px.scatter(
            df_plot,
            x=col_x, y=col_y,
            color="industria",
            color_discrete_sequence=COLORES + px.colors.qualitative.Pastel + px.colors.qualitative.Set1,
            hover_data={"razon_social": True, "industria": True, "anio": True, col_x: ':.2f', col_y: ':.2f'},
            labels={col_x: var_x, col_y: var_y, "industria": "Industria"}, title=""
        )
        fig2.update_traces(marker=dict(size=9, opacity=0.65, line=dict(width=0.3, color='black')))
        fig2.update_layout(
            margin=dict(t=40, b=60),
            font=dict(family="Fira Sans, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(title="Industria", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        if log_x: fig2.update_xaxes(type="log")
        if log_y: fig2.update_yaxes(type="log")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"<div style='text-align:center; color:#222; font-size:18px; font-family: Fira Sans, sans-serif;'>Correlación de Pearson: <b>{corr_pearson:.2f}</b></div>", unsafe_allow_html=True)
        st.caption("Pasa el mouse por los puntos para ver el detalle de cada empresa. Si la relación visual no es clara, prueba escalas logarítmicas o filtra por industria.")
    else:
        st.warning("No hay datos disponibles para esta combinación de variables y filtros.")


# --------- TAB 3: CORRELACIONES ------------
with tab3:
    import plotly.express as px

    with st.expander("Ver mapa de calor de correlaciones entre indicadores financieros (toda la muestra)", expanded=True):
        st.markdown("### Mapa de calor de correlaciones entre indicadores")
        df_num = df[numericos].copy().replace([np.inf, -np.inf], np.nan).dropna()
        corr_matrix = df_num.corr(method="pearson")  # O "spearman" si prefieres

    metodo = st.radio("Tipo de correlación", options=["Pearson", "Spearman"], horizontal=True)
    corr_matrix = df_num.corr(method=metodo.lower())


    fig_corr = px.imshow(
        corr_matrix,
        x=[k for k in nombres.keys() if nombres[k] in numericos],
        y=[k for k in nombres.keys() if nombres[k] in numericos],
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_corr.update_layout(
        height=600,
        margin=dict(t=50, b=50, l=50, r=50),
        font=dict(family="Fira Sans, sans-serif"),
        coloraxis_colorbar=dict(title="Correlación"),
        xaxis_title="Variable",
        yaxis_title="Variable"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("El color indica la fuerza y signo de la correlación de Pearson. Haz hover sobre cada celda para ver el valor exacto. Rojo: correlación positiva fuerte. Azul: negativa fuerte. Blanco: sin relación.")

    import itertools

    # --- Crea el ranking de correlaciones ---
    # Solo parte superior de la matriz, sin duplicados ni diagonal
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Variable 1", "Variable 2", "Correlación"]

    # Ranking: top 5 positivas y top 5 negativas (absolutas)
    top_positivas = corr_pairs.sort_values("Correlación", ascending=False).head(5)
    top_negativas = corr_pairs.sort_values("Correlación").head(5)

    # --- 1. Función para limpiar nombres de variables ---
    def limpiar_nombre(var):
        # Transforma snake_case a Capital Case bonito
        return var.replace("_", " ").replace("  ", " ").title()

    # Aplica limpieza de nombres
    top_positivas_fmt = top_positivas.copy()
    top_positivas_fmt["Variable 1"] = top_positivas_fmt["Variable 1"].apply(limpiar_nombre)
    top_positivas_fmt["Variable 2"] = top_positivas_fmt["Variable 2"].apply(limpiar_nombre)

    top_negativas_fmt = top_negativas.copy()
    top_negativas_fmt["Variable 1"] = top_negativas_fmt["Variable 1"].apply(limpiar_nombre)
    top_negativas_fmt["Variable 2"] = top_negativas_fmt["Variable 2"].apply(limpiar_nombre)

    # --- 2. Título con estilo ---
    st.markdown("""
    <div style="font-size: 24px; font-weight: 700; font-family: Fira Sans, sans-serif; margin-bottom:0.5em;">
    📊 Ranking de correlaciones más altas <span style='font-size:16px; font-weight:400;'>(muestra completa)</span>
    </div>
    """, unsafe_allow_html=True)

    # --- 3. Tablas con títulos destacados ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='font-size:18px;font-weight:600; color:#BF1B18; font-family:Fira Sans, sans-serif;'>Top 5 correlaciones positivas</div>", unsafe_allow_html=True)
        st.dataframe(top_positivas_fmt[["Variable 1", "Variable 2", "Correlación"]].rename(
            columns={"Variable 1": "Variable 1", "Variable 2": "Variable 2", "Correlación": "Correlación"}),
            hide_index=True, height=240)
    with col2:
        st.markdown("<div style='font-size:18px;font-weight:600; color:#1E87C1; font-family:Fira Sans, sans-serif;'>Top 5 correlaciones negativas</div>", unsafe_allow_html=True)
        st.dataframe(top_negativas_fmt[["Variable 1", "Variable 2", "Correlación"]].rename(
            columns={"Variable 1": "Variable 1", "Variable 2": "Variable 2", "Correlación": "Correlación"}),
            hide_index=True, height=240)

    # --- 4. Resumen con interpretación elegante ---
    mayor = top_positivas_fmt.iloc[0]
    menor = top_negativas_fmt.iloc[0]

    st.markdown(f"""
    <div style="border-radius:10px; background-color:#F2F6FA; border-left:5px solid #BF1B18; padding:1em 1.5em; font-size:16px; font-family: Fira Sans, sans-serif; margin-top:1em;">
    <b>La correlación más alta</b> es entre <b style="color:#BF1B18;">{mayor['Variable 1']}</b> y <b style="color:#BF1B18;">{mayor['Variable 2']}</b> <b>({mayor['Correlación']:.2f})</b>,
    mientras que <b>la más negativa</b> es entre <b style="color:#1E87C1;">{menor['Variable 1']}</b> y <b style="color:#1E87C1;">{menor['Variable 2']}</b> <b>({menor['Correlación']:.2f})</b>.
    </div>
    """, unsafe_allow_html=True)
