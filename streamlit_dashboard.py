import streamlit as st
import pandas as pd
import altair as alt

import plotly.express as px
import plotly.graph_objects as go

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import seaborn as sns

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import shap

config = {'displayModeBar': False}

# helper functions

def plot_strip(df, x, y, hue=None, reg_line=False):
    # plot our strip plot
    fig = plt.figure(figsize=(30, 10), facecolor='#0e1117').tight_layout()
    if hue:
        sns.stripplot(data=df, x=x, y=y, hue=hue, palette="flare", native_scale=True, dodge=True)
        plt.legend(title=hue, facecolor='#1f1f1f', edgecolor='#1f1f1f')
    else:
        sns.stripplot(data=df, x=x, y=y, color='#c2dafc', native_scale=True, dodge=True)

    # plot a regression line
    if reg_line:
        sns.regplot(
            data=df, x=x, y=y, ci=99,
            scatter=False, truncate=False, order=1, color='.75'
        )
    return fig

def plot_gauge(val, ref_val, grade, title):
    cmap = LinearSegmentedColormap.from_list('rg',["firebrick", "goldenrod", "seagreen"], N=50) 
    color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    
    delta_suff = ""
    gauge_pref = ""
    val = np.round(val)
    ref_val = np.round(ref_val)
    if ref_val - val > 25:
        gauge_pref = "<"
        delta_suff = "<"
    if ref_val - val < -25:
        gauge_pref = ">"
        delta_suff = ">"
        
    val = np.clip(val, ref_val - 25, ref_val + 25)
    color = color_list[int(val - ref_val + 25 - 1)]
        
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0.25, 1]},
        value = val,
        mode = "gauge+number+delta",
        delta = {'reference': ref_val, 'prefix': delta_suff, 'suffix': "%"},
        number = { 'suffix': "%", 'prefix': gauge_pref},
        gauge = {'shape': 'bullet',
                'axis': {'range': [-25 + ref_val, 25 + ref_val]},
                 'bar': {'color': color, 'thickness': 1},
                 'borderwidth': 0,
                 'steps' : [
                     {'range': [-25 + ref_val, 25 + ref_val], 'color': "#2f2f2f",},],
                 'threshold' : {'line': {'color': "whitesmoke", 'width': 8}, 'thickness': 1, 'value': val}}
        ))
    fig.update_layout(margin_b=0)
    fig.update_layout(margin_l=0)
    fig.update_layout(margin_r=20)
    fig.update_layout(margin_t=40)
    fig.update_layout(height=120)
    fig.update_layout(
        title_text=f"{title} vs median V{grade} climber"
    )
    return fig

def plot_prediction(val, ref_val, title, delta = False):
    cmap = LinearSegmentedColormap.from_list('rg',["firebrick", "goldenrod", "seagreen"], N=11) 
    color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    
    delta_suff = ""
    gauge_pref = "V"
    if val > 13:
        gauge_pref = "<V"
        delta_suff = "<"
    if val < 3:
        gauge_pref = ">V"
        delta_suff = ">"
        
    val = np.round(np.clip(val, 3, 13))
    ref_val = np.round(ref_val)
    color = color_list[int(val - 3)]
        
    if delta:
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0.25, 1]},
            value = val,
            mode = "number+delta",
            delta = {'reference': ref_val, 'prefix': delta_suff, 'position': 'right'},
            number = {'prefix': gauge_pref},
            ))
    else:
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0.25, 1]},
            value = val,
            mode = "number",
            number = {'prefix': gauge_pref},
            ))
    fig.update_layout(margin_b=0)
    fig.update_layout(margin_l=0)
    fig.update_layout(margin_r=20)
    fig.update_layout(margin_t=40)
    fig.update_layout(height=120)
    fig.update_layout(
        title_text=title,
        title={
            'x':0.4,
            'xanchor': 'center',
        }
    )
    return fig


# streamlit stuff start

st.set_page_config(
    page_title="Bouldering Analytics Dashboard",
    page_icon="🧗",
    layout="wide",
    initial_sidebar_state="expanded")

# removes some whitespace
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

alt.themes.enable("dark")

categorical_cols = [
    "sex",
    "V Grade"
]

bouldering_clean = pd.read_csv('clean_dataset.csv')
residual_data = pd.read_csv('residual_data.csv')
residual_data["residual error"] = residual_data["actual y"] - residual_data["predicted y"]
bouldering_jitter = bouldering_clean.copy()
np.random.seed(42)
for col in categorical_cols:
    bouldering_jitter[col] = bouldering_jitter[col] + np.random.rand(len(bouldering_jitter[col])) * 0.25 - 0.125

# load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

new_climber_dict = {}

sex_dict_map = {
    "Male": 0.0,
    "Female": 1.0,
    "Other": 0.5
}

sex_dict_map_display = {
    "Male": 0.0,
    "Female": 1.0,
}

st.markdown(
    """
    <style>
    [data-testid="stElementToolbar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title('🧗 Bouldering Grade Prediction Dashboard')
    new_climber_dict["sex"] = sex_dict_map[st.selectbox("Sex", ('Male', 'Female', 'Other'))]
    new_climber_dict["age"] = st.number_input("Age", value=22)
    new_climber_dict["height"] = st.number_input("Height in inches (6ft = 72in)", value=70)
    new_climber_dict["weight"] = st.number_input("Weight (lbs)", value=165)
    new_climber_dict["experience"] = st.number_input("Number of years climbing",value=9)
    new_climber_dict["training experience"] = st.number_input("Number of years training for climbing",value=1)
    new_climber_dict["days"] = st.number_input("Number of days climbing outdoors per month", value=12)
    new_climber_dict["finger strength"] = 1 + (st.number_input("Max additional weight for a 10s hang on a 20mm edge (lbs)", value=55) / new_climber_dict["weight"])
    new_climber_dict["weighted pull"] = 1 + (st.number_input("Max additional weight for a pullup on a bar (lbs)", value=105) / new_climber_dict["weight"])
    new_climber_dict["ape index"] = st.number_input("Ape index (someone with a height of 6ft and a span of 6'2\" would enter 2)", value=2)
    new_climber_dict["v grade"] = st.number_input("Current grade or goal grade (V Grade)", value=8, min_value=3, max_value=13)

# set up our prediction
val_array = np.array(
    [
        new_climber_dict["height"],
        new_climber_dict["weight"],
        new_climber_dict["age"],
        new_climber_dict["days"],
        new_climber_dict["experience"],
        new_climber_dict["training experience"],
        new_climber_dict["finger strength"],
        new_climber_dict["weighted pull"],
        new_climber_dict["ape index"]
    ]
).reshape(1, -1)

similarity_array = np.array(
    [
        new_climber_dict["age"],
        new_climber_dict["days"],
        new_climber_dict["sex"],
        new_climber_dict["height"],
        new_climber_dict["weight"],
        new_climber_dict["experience"],
        new_climber_dict["training experience"],
        new_climber_dict["finger strength"],
        new_climber_dict["weighted pull"],
        new_climber_dict["ape index"]
    ]
).reshape(1, -1)

similarity_feature_list = [
    "age",
    "days outdoors",
    "sex",
    "height",
    "weight",
    "years climbing",
    "years training",
    "Weighted hang ratio",
    "Weighted pull ratio",
    "ape index"
]

feature_names = [
    "height",
    "weight",
    "age",
    "days outside",
    "climbing experience",
    "training experience",
    "finger strength",
    "weighted pull",
    "ape index"
]

column_display_order = [
    "V Grade", 
    "similarity",
    "age", 
    "sex", 
    "height", 
    "weight", 
    "ape index", 
    "Weighted hang ratio",
    "Weighted pull ratio",
    "years climbing",
    "years training",
    "days outdoors"
]

st.markdown('#### Your metrics')
col1, col2, col3, col4 = st.columns((1, 1, 3.5, 3.5), gap='small')

p_grade = model.predict(val_array)[0]
v_grade = new_climber_dict["v grade"]

# load
with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

scaled = model[0].transform(val_array)
shap_vals = explainer.shap_values(scaled).tolist()[0]
shap_vals, feature_names = zip(*sorted(zip(shap_vals, feature_names)))

scaler = StandardScaler()
bouldering_clean_no_na = bouldering_clean.dropna().drop(columns=["span", "# pullups", "# pushups"])
scaled_features = scaler.fit_transform(bouldering_clean_no_na[similarity_feature_list].values)
scaled_climber = scaler.transform(similarity_array)

if 'similarity' in bouldering_clean_no_na.columns:
    bouldering_clean_no_na.pop('similarity')

from sklearn.metrics.pairwise import cosine_similarity
similarity = (cosine_similarity(scaled_climber, scaled_features).reshape(-1) + 1) / 2 * 100
bouldering_clean_no_na.insert(0, 'similarity', similarity)
bouldering_similarity_display = bouldering_clean_no_na.sort_values(by=['similarity'], ascending=False)
bouldering_similarity_display["sex"] = bouldering_similarity_display["sex"].replace([0, 1], sex_dict_map_display)
bouldering_similarity_display["Weighted hang ratio"] *= 100
bouldering_similarity_display["Weighted pull ratio"] *= 100

with col1:
    metric = plot_prediction(v_grade, p_grade, "Entered Grade")
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(metric, use_container_width=True, sharing="streamlit", theme="streamlit", config = config)

with col2:
    #st.metric("Predicted V Grade", f"{p_grade:.0f}")
    metric = plot_prediction(p_grade, v_grade, "Predicted Grade", delta=True)
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(metric, use_container_width=True, sharing="streamlit", theme="streamlit", config = config)

with col3:
    f_strength = new_climber_dict["finger strength"] * 100
    v_grade = new_climber_dict["v grade"]
    hang_median = np.nanmedian(bouldering_clean[bouldering_clean["V Grade"] == v_grade]["Weighted hang ratio"]) * 100
    gauge = plot_gauge(f_strength, hang_median, v_grade, "Weighted Hang (% BW)")
    
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(gauge, use_container_width=True, sharing="streamlit", theme="streamlit", config = config)

with col4:
    p_strength = new_climber_dict["weighted pull"] * 100
    v_grade = new_climber_dict["v grade"]
    pull_median = np.nanmedian(bouldering_clean[bouldering_clean["V Grade"] == v_grade]["Weighted pull ratio"]) * 100
    gauge = plot_gauge(p_strength, pull_median, v_grade, "Weighted Pull (% BW)")
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(gauge, use_container_width=True, sharing="streamlit", theme="streamlit", config = config)

col1, col2 = st.columns((1, 3.5), gap='small')
with col1:
    st.markdown("#### Model metrics")
    col11, col12, col13 = st.columns((1, 1, 1), gap='small')
    with col11:
        st.metric("Test RMSE", 1.4460)
    with col12:
        st.metric("Test MAE", 1.1967)
    with col13:
        st.metric("Test R^2", 0.6117)
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "h",
        measure = ["relative" for x in range(len(feature_names))],
        x = shap_vals,
        text = [f"{'+' if i > 0 else ''}{i:0.1f}" for i in shap_vals],
        y = feature_names,
        connector = {"line":{"color":"#8f8f8f"}},
        base = explainer.expected_value
    ))

    fig.update_layout(
            waterfallgap = 0.1,
    )
    fig.update_layout(margin_b=0)
    fig.update_layout(margin_l=0)
    fig.update_layout(margin_r=0)
    fig.update_layout(margin_t=0)
    fig.update_layout(height=300)
    st.markdown("#### Model variable influence")
    st.plotly_chart(fig, use_container_width=True, config = config)

    st.markdown("#### Model Residuals")
    fig = px.scatter(
        residual_data, x='predicted y', y='residual error',
        marginal_y='violin',
        color='actual y'
    )
    fig.update_layout(margin_b=0)
    fig.update_layout(margin_l=0)
    fig.update_layout(margin_r=0)
    fig.update_layout(margin_t=0)
    fig.update_yaxes(range=[min(residual_data["residual error"]) - 0.1, max(residual_data["residual error"]) + 0.1])
    st.plotly_chart(fig, use_container_width=True, config = config)

with col2:
    st.markdown("#### Data exploration")
    col11, col12, col13, col14 = st.columns((1, 1, 1, 1), gap='medium')
    axis_list = list(bouldering_jitter.columns)
    color_list = [None] + list(bouldering_jitter.columns)
    with col11:
        x_axis = st.selectbox("X axis", axis_list, index = 0)
        axis_list.remove(x_axis)
    with col12:
        y_axis = st.selectbox("Y axis", axis_list, index = 11)
    with col13:
        color = st.selectbox("Color", color_list, index = 2)
    with col14:
        st.write('<div style="height: 35px;">Plot Regression Line</div>', unsafe_allow_html=True)
        reg_line = st.toggle("Plot Regression Line", label_visibility="collapsed", value=True)
    if reg_line:
        plot_trend="ols"
    else:
        plot_trend=None

    fig = px.scatter(bouldering_jitter, x=x_axis, y=y_axis, color=color, trendline=plot_trend)
    fig.update_layout(margin_b=0)
    fig.update_layout(margin_l=0)
    fig.update_layout(margin_r=0)
    fig.update_layout(margin_t=0)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, config = config)

    # similarity chart

    st.markdown(f"#### Your similar climbers")
    st.dataframe(
        bouldering_similarity_display.head(15), 
        column_order=column_display_order,
        column_config={
            "similarity": st.column_config.ProgressColumn(
                "similarity score",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "V Grade": st.column_config.NumberColumn(
                "grade",
                format="V%d",
            ),
            "days outdoors": st.column_config.NumberColumn(
                "outdoor time",
                format="%d days / month",
            ),
            "height": st.column_config.NumberColumn(
                format="%d in.",
            ),
            "weight": st.column_config.NumberColumn(
                format="%d lbs.",
            ),
            "days outdoors": st.column_config.NumberColumn(
                "outdoor time",
                format="%d days / month",
            ),
            "Weighted hang ratio": st.column_config.NumberColumn(
                "Weighted hang",
                format="%d%% BW",
            ),
            "Weighted pull ratio": st.column_config.NumberColumn(
                "Weighted pull",
                format="%d%% BW",
            )
        },
        hide_index=True,
        use_container_width=True,
        height=315)