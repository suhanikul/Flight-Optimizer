import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output
import numpy as np

# Load the dataset
file_path = "data/goibibo_flights_data.csv"
df = pd.read_csv(file_path)

# Data Cleaning
df['price'] = df['price'].replace({',': ''}, regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.drop(columns=[col for col in ['Unnamed: 11', 'Unnamed: 12'] if col in df.columns])
df['route'] = df['from'] + ' to ' + df['to']

# Route-based Analysis
route_data = df.groupby('route').agg(
    flight_count=('route', 'count'),
    avg_price=('price', 'mean')
).reset_index()

# Network Graph Creation
graph = nx.DiGraph()
for _, row in df.iterrows():
    graph.add_edge(row['from'], row['to'], weight=row['price'])

# Spring Layout
pos = nx.spring_layout(graph, seed=42)  # Use spring layout to position nodes

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Flight Route Analysis", style={'textAlign': 'center'}),
    
    html.H3("Data Overview"),
    dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.head(10).to_dict('records'),
        style_table={'overflowX': 'auto'}
    ),
    
    html.H3("Flight Frequency"),
    dcc.Graph(
        figure={
            'data': [
                go.Bar(
                    x=route_data['route'],
                    y=route_data['flight_count'],
                    marker={'color': route_data['flight_count'], 'colorscale': 'viridis'}
                )
            ],
            'layout': go.Layout(title='Number of Flights per Route', xaxis=dict(title='Route', tickangle=45))
        }
    ),
    
    html.H3("Flight Network Visualization"),
    dcc.Graph(
        id='network-graph',
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=[pos[node][0] for node in graph.nodes],
                    y=[pos[node][1] for node in graph.nodes],
                    mode='markers+text',
                    text=list(graph.nodes),
                    marker=dict(size=10, color='blue', opacity=0.8),
                    textposition="top center"
                ),
                # Create edges as lines
                go.Scatter(
                    x=[pos[u][0] for u, v in graph.edges] + [pos[v][0] for u, v in graph.edges],
                    y=[pos[u][1] for u, v in graph.edges] + [pos[v][1] for u, v in graph.edges],
                    mode='lines',
                    line=dict(width=0.5, color='gray'),
                    opacity=0.7
                )
            ],
            layout=go.Layout(
                title='Airport Connectivity',
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False)
            )
        )
    ),
    
    html.H3("Price Distribution"),
    dcc.Graph(
        figure={
            'data': [go.Histogram(x=df['price'], nbinsx=50, marker=dict(color='blue'))],
            'layout': go.Layout(
                title='Price Distribution',
                xaxis=dict(title='Price'),
                yaxis=dict(title='Number of Flights')
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
