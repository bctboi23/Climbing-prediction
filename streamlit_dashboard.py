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

bouldering_clean = pd.read_csv('clean_dataset.csv')
bouldering_jitter = bouldering_clean.copy()
np.random.seed(42)
bouldering_jitter["V Grade"] = bouldering_jitter["V Grade"] + np.random.rand(len(bouldering_jitter["V Grade"])) * 0.25 - 0.125

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


with st.sidebar:
    st.title('🧗 Bouldering Grade Prediction Dashboard')
    new_climber_dict["sex"] = sex_dict_map[st.selectbox("Sex", ('Male', 'Female', 'Other'))]
    new_climber_dict["age"] = st.number_input("Age", value=22)
    new_climber_dict["height"] = st.number_input("Height in inches (6ft = 72in)", value=70)
    new_climber_dict["weight"] = st.number_input("Weight (lbs)", value=165)
    new_climber_dict["experience"] = st.number_input("Number of years climbing",value=8)
    new_climber_dict["training experience"] = st.number_input("Number of years training for climbing",value=3)
    new_climber_dict["days"] = st.number_input("Number of days spent climbing outdoors per year", value=75)
    new_climber_dict["finger strength"] = 1 + (st.number_input("Max additional weight for a 10s hang on a 20mm edge (lbs)", value=65) / new_climber_dict["weight"])
    new_climber_dict["weighted pull"] = 1 + (st.number_input("Max additional weight for a pullup on a bar (lbs)", value=105) / new_climber_dict["weight"])
    new_climber_dict["ape index"] = st.number_input("Ape index (someone with a height of 6ft and a span of 6'2\" would enter 2)", value=2)
    new_climber_dict["v grade"] = st.number_input("Current grade or goal grade (V Grade)", value=8, min_value=3, max_value=13)

# set up our prediction
val_array = np.array(
    [
        new_climber_dict["age"],
        np.log(new_climber_dict["days"]), # feature transformation
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
        np.log(new_climber_dict["days"]), # feature transformation
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
    "days outdoors yearly",
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
    "age",
    "days outside",
    "climbing experience",
    "training experience",
    "finger strength",
    "weighted pull",
    "ape index"
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
bouldering_clean_no_na["days outdoors yearly"] = np.log(bouldering_clean_no_na["days outdoors yearly"])
scaled_features = scaler.fit_transform(bouldering_clean_no_na[similarity_feature_list].values)
scaled_climber = scaler.transform(similarity_array)
if 'similarity' in bouldering_clean_no_na.columns:
    bouldering_clean_no_na.pop('similarity')

from sklearn.metrics.pairwise import cosine_similarity
similarity = (cosine_similarity(scaled_climber, scaled_features).reshape(-1) + 1) / 2 * 100
bouldering_clean_no_na.insert(0, 'similarity', similarity)
bouldering_similarity_display = bouldering_clean_no_na.sort_values(by=['similarity'], ascending=False)
bouldering_similarity_display["days outdoors yearly"] = np.round(np.exp(bouldering_clean_no_na["days outdoors yearly"])) # retranslate for display purposes
bouldering_similarity_display["sex"] = bouldering_similarity_display["sex"].replace([0, 1], sex_dict_map_display)

with col1:
    metric = plot_prediction(v_grade, p_grade, "Entered Grade")
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(metric, use_container_width=True, sharing="streamlit", theme="streamlit")

with col2:
    #st.metric("Predicted V Grade", f"{p_grade:.0f}")
    metric = plot_prediction(p_grade, v_grade, "Predicted Grade", delta=True)
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(metric, use_container_width=True, sharing="streamlit", theme="streamlit")

with col3:
    f_strength = new_climber_dict["finger strength"] * 100
    v_grade = new_climber_dict["v grade"]
    hang_median = np.nanmedian(bouldering_clean[bouldering_clean["V Grade"] == v_grade]["Weighted hang ratio"]) * 100
    gauge = plot_gauge(f_strength, hang_median, v_grade, "Weighted Hang (% BW)")
    
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(gauge, use_container_width=True, sharing="streamlit", theme="streamlit")

with col4:
    p_strength = new_climber_dict["weighted pull"] * 100
    v_grade = new_climber_dict["v grade"]
    pull_median = np.nanmedian(bouldering_clean[bouldering_clean["V Grade"] == v_grade]["Weighted pull ratio"]) * 100
    gauge = plot_gauge(p_strength, pull_median, v_grade, "Weighted Pull (% BW)")
    
    #st.plotly_chart(gauge_finger, sharing="streamlit", theme="streamlit", **{'config': {'displayModeBar': False}})
    st.plotly_chart(gauge, use_container_width=True, sharing="streamlit", theme="streamlit")

col1, col2 = st.columns((1, 3.5), gap='small')
with col1:
    st.markdown("#### Model stats")
    col11, col12, col13 = st.columns((1, 1, 1), gap='small')
    with col11:
        st.metric("Root Mean Squared Error", 1.4459)
    with col12:
        st.metric("Mean Absolute Error", 1.1627)
    with col13:
        st.metric("Adjusted R^2", 0.6214)
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
    fig.update_layout(height=370)
    st.markdown("#### Model variable influence")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    col11, col12 = st.columns((1, 1), gap='small')
    with col11:
        st.markdown("#### Grade vs finger strength")
        fig = px.scatter(bouldering_jitter, x="V Grade", y="Weighted hang ratio", color="years climbing", trendline="ols")
        fig.update_layout(margin_b=0)
        fig.update_layout(margin_l=0)
        fig.update_layout(margin_r=0)
        fig.update_layout(margin_t=0)
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
    with col12:
        st.markdown("#### Grade vs pull strength")
        fig = px.scatter(bouldering_jitter, x="V Grade", y="Weighted pull ratio", color="sex", trendline="ols")
        fig.update_layout(margin_b=0)
        fig.update_layout(margin_l=0)
        fig.update_layout(margin_r=0)
        fig.update_layout(margin_t=0)
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns((1, 1), gap='small')
with col1:
    st.markdown(f"#### Most similar climbers")
    st.dataframe(
        bouldering_similarity_display.head(20), 
        column_config={
            "similarity": st.column_config.ProgressColumn(
                "similarity",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "V Grade": st.column_config.NumberColumn(
                "grade",
                format="V%d",
            ),
            "days outdoors yearly": st.column_config.NumberColumn(
                "outdoor time",
                format="%d days / yr",
            ),
        },
        hide_index=True,
        use_container_width=True)
with col2:
    st.markdown(f"#### Most similar V{v_grade} climbers")
    st.dataframe(
        bouldering_similarity_display[bouldering_similarity_display["V Grade"] == v_grade].head(20), 
        column_config={
            "similarity": st.column_config.ProgressColumn(
                "similarity",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "V Grade": st.column_config.NumberColumn(
                "grade",
                format="V%d",
            ),
            "days outdoors yearly": st.column_config.NumberColumn(
                "outdoor time",
                format="%d days / yr",
            )
        },
        hide_index=True,
        use_container_width=True)