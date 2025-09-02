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
import dash.dash_table as dt

# -----------------------------
# 1) Fake-"Loan"-Datensatz bauen (11 IVs + DV), reproducible
# -----------------------------
RND = np.random.RandomState(42)
n = 32000

df = pd.DataFrame({
    "IV1_Age": RND.randint(20, 85, n),
    "IV2_AnnualIncome": np.clip(RND.lognormal(mean=10.7, sigma=0.7, size=n), 4_000, 2_100_000),
    "IV3_HomeOwnership": RND.randint(0, 4, n),             # 0 rent, 1 own, 2 mortgage, 3 other
    "IV4_EmploymentLen": np.clip(RND.normal(5, 4, n).round().astype(int), 0, 38),
    "IV5_LoanIntent": RND.randint(0, 6, n),                # encoded categories
    "IV6_LoanGrade": RND.randint(0, 7, n),                 # 0..6
    "IV7_LoanAmount": np.clip(RND.normal(10_000, 6_500, n), 500, 35_000),
    "IV8_InterestRate": np.clip(RND.normal(11.0, 3.2, n), 5.2, 23.5),
    "IV9_PercentIncome": np.clip(RND.beta(2, 8, n), 0, 0.9),
    "IV10_HistDefault": RND.binomial(1, 0.18, n),
    "IV11_CreditHistLen": np.clip(RND.normal(6, 4, n).round().astype(int), 0, 30)
})

# "Wahre" Default-Wahrscheinlichkeit: sigmoid( lineare Kombi )
z = (
    0.015 * (df["IV8_InterestRate"] - 10) +
    0.002 * (df["IV7_LoanAmount"] / 1000) +
    0.9   * df["IV10_HistDefault"] +
    0.8   * (df["IV9_PercentIncome"] > 0.25).astype(int) +
   -0.3   * (df["IV2_AnnualIncome"] > 120_000).astype(int) +
   -0.01  * (df["IV11_CreditHistLen"] > 10).astype(int)
)
p = 1/(1+np.exp(-z))
df["DV_Default"] = (RND.uniform(size=n) < p).astype(int)

FEATURES = df.columns[:-1]
TARGET = "DV_Default"

# -----------------------------
# 2) Vorberechnungen: Summary, Corr, PCA (für initiale Visuals)
# -----------------------------
summary = pd.DataFrame({
    "Ind. variable": FEATURES,
    "Mean": df[FEATURES].mean().round(2),
    "Std.": df[FEATURES].std().round(2),
    "Min": df[FEATURES].min().round(2),
    "Max": df[FEATURES].max().round(2),
}).reset_index(drop=True)

corr = df[FEATURES].corr()

# Standardisierung für PCA/NN
scaler = StandardScaler().fit(df[FEATURES])
X_scaled_all = scaler.transform(df[FEATURES])

pca_all = PCA(n_components=3, random_state=0).fit(X_scaled_all)
X_pca_all = pca_all.transform(X_scaled_all)
pca2_df = pd.DataFrame({"PC1": X_pca_all[:,0], "PC2": X_pca_all[:,1], "Default": df[TARGET]})
pca3_df = pd.DataFrame({"PC1": X_pca_all[:,0], "PC2": X_pca_all[:,1], "PC3": X_pca_all[:,2], "Default": df[TARGET]})

# -----------------------------
# 3) Dash App
# -----------------------------
app = dash.Dash(__name__)
server = app.server

heading_style = {"fontWeight": "700", "fontSize": "22px", "margin": "0 0 10px 0"}
subhead_style = {"fontWeight": "600", "fontSize": "18px", "marginTop": "20px"}
para_style = {"lineHeight": "1.6", "textAlign": "justify"}

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "24px", "maxWidth": "1280px", "margin": "0 auto"},
    children=[

    html.H2("A Machine learning approach to credit risk assessment",
            style={"textAlign": "center", "marginBottom": "24px"}),

    html.Div(style={"display": "grid", "gridTemplateColumns": "1.2fr 0.8fr", "gap": "24px"}, children=[

        # -------- Left Column: Beschreibung, Summary, Corr, PCA ----------
        html.Div(children=[
            html.P(
                "This application uses a dense neural network to solve a credit-risk classification problem: "
                "predicting whether a loan will default. You can alter the network structure and see the results.",
                style=para_style),

            html.H3("Information used", style=subhead_style),
            html.Ul([
                html.Li("Age (IV 1)"),
                html.Li("Annual Income (IV 2), in dollars"),
                html.Li("Home ownership (IV 3)"),
                html.Li("Employment length (IV 4), in years"),
                html.Li("Loan intent (IV 5)"),
                html.Li("Loan grade (IV 6)"),
                html.Li("Loan amount (IV 7), in dollars"),
                html.Li("Interest rate (IV 8), percent"),
                html.Li("Percent income (IV 9)"),
                html.Li("Historical default (IV 10)"),
                html.Li("Credit history length (IV 11)"),
                html.Li("Loan status (DV): defaulted vs not defaulted"),
            ]),

            html.P("The dataset is split into train (80%) and test (20%).", style=para_style),

            html.H3("Understanding the data", style=subhead_style),
            dt.DataTable(
                columns=[{"name": c, "id": c} for c in summary.columns],
                data=summary.to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={"padding": "6px", "fontSize": 13},
                style_header={"fontWeight": "700"}
            ),

            html.H3("Plot the matrix of the correlations between the variables:", style=subhead_style),
            dcc.Graph(
                id="corr-heatmap",
                figure=px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                 labels=dict(color="Corr.")).update_layout(margin=dict(l=0,r=0,t=0,b=0))
            ),

            html.P(
                "Perform a two-dimensional principal component analysis where red markers are defaulted loans "
                "and green markers are non-defaulted.", style=para_style),
            dcc.Graph(
                id="pca-2d",
                figure=px.scatter(pca2_df, x="PC1", y="PC2", color=pca2_df["Default"].map({1: "Defaulted", 0: "Not defaulted"}),
                                  color_discrete_map={"Defaulted": "#e74c3c", "Not defaulted": "#2ecc71"},
                                  render_mode="webgl").update_layout(margin=dict(l=0,r=0,t=0,b=0))
            ),

            html.P(
                "A three-dimensional PCA scatter shows the same split in 3D.", style=para_style),
            dcc.Graph(
                id="pca-3d",
                figure=px.scatter_3d(pca3_df, x="PC1", y="PC2", z="PC3",
                                     color=pca3_df["Default"].map({1: "Defaulted", 0: "Not defaulted"}),
                                     color_discrete_map={"Defaulted": "#e74c3c", "Not defaulted": "#2ecc71"},
                                     opacity=0.7)
                    .update_layout(margin=dict(l=0,r=0,t=0,b=0))
            ),
        ]),

        # -------- Right Column: Controls, Model, Results ----------
        html.Div(children=[
            html.H3("Neural network construction", style=heading_style),
            html.P("Select network structure and train the model on the training set. Accuracy and confusion matrix are computed on the test set."),

            html.Label("Number of hidden layers"),
            dcc.Dropdown(
                id="layers",
                options=[{"label": f"{k} hidden layer{'s' if k>1 else ''}", "value": k} for k in [1,2,3,4]],
                value=2, clearable=False, style={"marginBottom": "10px"}),

            html.Label("Neurons per hidden layer"),
            dcc.Dropdown(
                id="neurons",
                options=[{"label": f"{k} neurons", "value": k} for k in [8,16,32,64,128]],
                value=32, clearable=False, style={"marginBottom": "10px"}),

            html.Label("Max iterations"),
            dcc.Slider(id="maxiter", min=100, max=600, step=50, value=250, tooltip={"placement": "bottom"}),

            html.Button("Train model", id="train", n_clicks=0, style={"marginTop": "12px"}),

            html.Hr(),

            html.Div(id="acc-text", style={"fontWeight": "700", "fontSize": "18px", "margin": "8px 0"}),

            dcc.Graph(id="cm-heatmap"),
            dcc.Graph(id="loss-curve"),
        ])
    ])
])

# -----------------------------
# 4) Callback: train & update plots
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
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=123, stratify=df[TARGET]
    )
    scaler_loc = StandardScaler().fit(X_train)
    Xtr = scaler_loc.transform(X_train); Xte = scaler_loc.transform(X_test)

    # Hidden layer sizes tuple
    hidden = tuple([n_neurons]*n_layers)

    # MLP
    clf = MLPClassifier(hidden_layer_sizes=hidden,
                        activation="relu", solver="adam",
                        max_iter=int(max_iter), random_state=1,
                        early_stopping=True, n_iter_no_change=10,
                        verbose=False)
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["Act. ND","Act. D"], columns=["Pred. ND","Pred. D"])
    cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens")
    cm_fig.update_layout(title="Confusion matrix (test set)", margin=dict(l=0,r=0,t=40,b=0))

    # Loss curve
    loss = getattr(clf, "loss_curve_", [])
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(y=loss, mode="lines"))
    loss_fig.update_layout(title="Training loss over iterations", xaxis_title="Iteration", yaxis_title="Loss",
                           margin=dict(l=0,r=0,t=40,b=0))

    acc_txt = f"Test accuracy: {acc*100:.2f}%  |  Hidden layers: {n_layers} × {n_neurons}"
    return acc_txt, cm_fig, loss_fig


if __name__ == "__main__":
    app.run_server(debug=True)
