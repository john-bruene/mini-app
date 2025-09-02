import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Beispiel-Datensatz: Iris (Blumendaten)
df = px.data.iris()

# Dash App starten
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interaktive Demo App", style={"textAlign": "center"}),

    html.P("Wähle die Achsen aus:", style={"textAlign": "center"}),

    html.Div([
        html.Label("X-Achse:"),
        dcc.Dropdown(
            id="x-axis",
            options=[{"label": col, "value": col} for col in df.columns if df[col].dtype != "object"],
            value="sepal_width"
        ),
        html.Label("Y-Achse:"),
        dcc.Dropdown(
            id="y-axis",
            options=[{"label": col, "value": col} for col in df.columns if df[col].dtype != "object"],
            value="sepal_length"
        )
    ], style={"width": "50%", "margin": "auto"}),

    dcc.Graph(id="scatter-plot", style={"marginTop": "2em"})
])

@app.callback(
    Output("scatter-plot", "figure"),
    Input("x-axis", "value"),
    Input("y-axis", "value")
)
def update_graph(x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, color="species", title=f"{y_col} vs {x_col}")
    fig.update_layout(template="plotly_white")
    return fig

# Für Render/Heroku/Netlify
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
