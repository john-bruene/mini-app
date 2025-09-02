import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output

# -----------------------------
# 1) Demo-Datensatz (synthetisch)
# -----------------------------
RND = np.random.RandomState(42)
n = 12000

df = pd.DataFrame({
    "IV1_Age": RND.randint(20, 85, n),
    "IV2_AnnualIncome": np.clip(RND.lognormal(mean=10.7, sigma=0.7, size=n), 4_000, 2_100_000),
    "IV3_HomeOwnership": RND.randint(0, 4, n),
    "IV4_EmploymentLen": np.clip(RND.normal(5, 4, n).round().astype(int), 0, 38),
    "IV5_LoanIntent": RND.randint(0, 6, n),
    "IV6_LoanGrade": RND.randint(0, 7, n),
    "IV7_LoanAmount": np.clip(RND.normal(10_000, 6_500, n), 500, 35_000),
    "IV8_InterestRate": np.clip(RND.normal(11.0, 3.2, n), 5.2, 23.5),
    "IV9_PercentIncome": np.clip(RND.beta(2, 8, n), 0, 0.9),
    "IV10_HistDefault": RND.binomial(1, 0.18, n),
    "IV11_CreditHistLen": np.clip(RND.normal(6, 4, n).round().astype(int), 0, 30)
})

# "wahre" Default-Wahrsch.
z = (
    0.015 * (df["IV8_InterestRate"] - 10) +
    0.002 * (df["IV7_LoanAmount"] / 1000) +
    0.9   * df["IV10_HistDefault"] +
    0.8   * (df["IV9_PercentIncome"] > 0.25).astype(int) -
    0.3   * (df["IV2_AnnualIncome"] > 120_000).astype(int) -
    0.01  * (df["IV11_CreditHistLen"] > 10).astype(int)
)
p = 1 / (1 + np.exp(-z))
df["DV_Default"] = (RND.uniform(size=n) < p).astype(int)

FEATURES = df.columns[:-1]
TARGET = "DV_Default"

# -----------------------------
# 2) Vorberechnungen für EDA
# -----------------------------
corr = df[FEATURES].corr()

scaler_all = StandardScaler().fit(df[FEATURES])
X_scaled_all = scaler_all.transform(df[FEATURES])

pca = PCA(n_components=3, random_state=0).fit(X_scaled_all)
X_pca = pca.transform(X_scaled_all)
pca2_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Default": df[TARGET]})
pca3_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "PC3": X_pca[:, 2], "Default": df[TARGET]})

# EDA-Figuren (fixe Höhen, kompakte Ränder)
fig_corr = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                     labels=dict(color="Corr."))
fig_corr.update_layout(margin=dict(l=6, r=6, t=24, b=6), height=340, coloraxis_colorbar=dict(len=0.8))

fig_pca2 = px.scatter(
    pca2_df, x="PC1", y="PC2",
    color=pca2_df["Default"].map({1: "Defaulted", 0: "Not defaulted"}),
    color_discrete_map={"Defaulted": "#e74c3c", "Not defaulted": "#2ecc71"},
    render_mode="webgl"
)
fig_pca2.update_layout(margin=dict(l=6, r=6, t=24, b=6), height=340)
# Achsen bewusst begrenzen (keine "durch die Decke"-Plots)
fig_pca2.update_xaxes(range=[-6, 6])
fig_pca2.update_yaxes(range=[-6, 6])

fig_pca3 = px.scatter_3d(
    pca3_df, x="PC1", y="PC2", z="PC3",
    color=pca3_df["Default"].map({1: "Defaulted", 0: "Not defaulted"}),
    color_discrete_map={"Defaulted": "#e74c3c", "Not defaulted": "#2ecc71"},
    opacity=0.7
)
fig_pca3.update_layout(margin=dict(l=6, r=6, t=24, b=6), height=340)

# -----------------------------
# 3) Dash App Layout (Fokus: Layout)
# -----------------------------
app = dash.Dash(__name__)
server = app.server

container = {"maxWidth": "1280px", "margin": "0 auto", "padding": "24px",
             "fontFamily": "system-ui,-apple-system,Segoe UI,Roboto,Arial"}

card = {"background": "white", "borderRadius": "12px", "boxShadow": "0 6px 18px rgba(0,0,0,.08)", "padding": "16px"}

app.layout = html.Div(style=container, children=[

    html.H2("A Machine learning approach to credit risk assessment",
            style={"textAlign": "center", "marginBottom": "18px"}),

    # Kopf: kurzer Text + Controls in 2 Spalten
    html.Div(style={"display": "grid", "gridTemplateColumns": "1.2fr 0.8fr", "gap": "16px"}, children=[
        html.Div(style=card, children=[
            html.P(
                "Minimal demo focusing on layout. Adjust the network structure on the right and inspect "
                "the data via correlation and PCA. This is synthetic data for a credit risk illustration."
            )
        ]),
        html.Div(style=card, children=[
            html.Label("Hidden layers"), 
            dcc.Dropdown(
                id="layers",
                options=[{"label": f"{k} layer{'s' if k>1 else ''}", "value": k} for k in [1, 2, 3, 4]],
                value=2, clearable=False, style={"marginBottom": "10px"}),

            html.Label("Neurons per layer"),
            dcc.Dropdown(
                id="neurons",
                options=[{"label": f"{k}", "value": k} for k in [8, 16, 32, 64, 128]],
                value=32, clearable=False, style={"marginBottom": "10px"}),

            html.Label("Max iterations"),
            dcc.Slider(id="maxiter", min=100, max=600, step=50, value=250,
                       tooltip={"placement": "bottom"}, marks=None, updatemode="drag"),

            html.Button("Train model", id="train", n_clicks=0, style={"marginTop": "12px"}),
            html.Div(id="acc-text", style={"fontWeight": 700, "marginTop": "10px"})
        ])
    ]),

    # EDA: 3 Spalten, gleiche Höhen
    html.H3("Data exploration", style={"margin": "22px 0 10px"}),
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "16px"}, children=[
        html.Div(style=card, children=[html.H4("Correlation matrix", style={"marginTop": 0}), dcc.Graph(figure=fig_corr, config={"displayModeBar": False})]),
        html.Div(style=card, children=[html.H4("PCA (2D)", style={"marginTop": 0}), dcc.Graph(figure=fig_pca2, config={"displayModeBar": False})]),
        html.Div(style=card, children=[html.H4("PCA (3D)", style={"marginTop": 0}), dcc.Graph(figure=fig_pca3, config={"displayModeBar": False})]),
    ]),

    # Ergebnisse: 2 Spalten (gleich hoch)
    html.H3("Results", style={"margin": "22px 0 10px"}),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
        html.Div(style=card, children=[html.H4("Confusion matrix", style={"marginTop": 0}), dcc.Graph(id="cm-heatmap", config={"displayModeBar": False}, style={"height": 360})]),
        html.Div(style=card, children=[html.H4("Training loss", style={"marginTop": 0}), dcc.Graph(id="loss-curve", config={"displayModeBar": False}, style={"height": 360})]),
    ])
])

# -----------------------------
# 4) Training + Plots Update
# -----------------------------
@app.callback(
    [Output("acc-text", "children"),
     Output("cm-heatmap", "figure"),
     Output("loss-curve", "figure")],
    [Input("train", "n_clicks"),
     Input("layers", "value"),
     Input("neurons", "value"),
     Input("maxiter", "value")]
)
def train_and_report(n_clicks, n_layers, n_neurons, max_iter):
    # Split + Scale
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=123, stratify=df[TARGET]
    )
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train); Xte = scaler.transform(X_test)

    hidden = tuple([int(n_neurons)] * int(n_layers))

    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=int(max_iter),
        random_state=1,
        early_stopping=True,
        n_iter_no_change=8,
        verbose=False
    )
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)

    # Confusion Matrix (kompakt)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Act. ND", "Act. D"], columns=["Pred. ND", "Pred. D"])
    cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens")
    cm_fig.update_layout(margin=dict(l=6, r=6, t=30, b=6), height=360)

    # Loss Curve mit y-Grenze (keine "Decke")
    loss = getattr(clf, "loss_curve_", [])
    loss_fig = go.Figure(go.Scatter(y=loss, mode="lines"))
    loss_fig.update_layout(
        margin=dict(l=6, r=6, t=30, b=6),
        height=360,
        xaxis_title="Iteration",
        yaxis_title="Loss"
    )
    # konservative Range, nur setzen wenn Daten vorhanden
    if len(loss) > 0:
        ymin = max(0, min(loss) - 0.05)
        ymax = max(loss) + 0.05
        loss_fig.update_yaxes(range=[ymin, ymax])

    acc_txt = f"Test accuracy: {acc*100:.2f}%  |  Hidden: {n_layers} × {n_neurons}"
    return acc_txt, cm_fig, loss_fig


if __name__ == "__main__":
    app.run_server(debug=True)
