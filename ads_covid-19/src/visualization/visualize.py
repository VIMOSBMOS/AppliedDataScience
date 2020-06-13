import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go

import os
# Get current working directory
print(os.getcwd())
df_input_large = pd.read_csv('data/processed/COVID_final_set.csv',sep=';')

fig = go.Figure()
app = dash.Dash()
app.layout = html.Div([ # In Markdown one can add text for the dashboard
dcc.Markdown('''
# Applied Data Science on COVID-19 data
Goals of the project:
- teach data science by applying a cross industry standard process
- automated data gathering, data transformations, filtering
- apply machine learning to approximate the doubling Timeline
- static deployment of the dashboard


'''),

dcc.Markdown('''
## Select different countries for visualization
'''),

dcc.Dropdown(# This is the selection area of the countries
id = 'country_drop_down',
options = [{'label' : each, 'value' : each} for each in df_input_large['country'].unique()],# check if countries are not repeated
value = ['US', 'Germany', 'Italy'],# pre-selected
multi = True # Many options can be chosen
),

dcc.Markdown('''
## Features
'''),

dcc.Dropdown(
id = 'doubling_time',
options = [
{'label' : 'Timeline Confirmed', 'value' : 'confirmed'},
{'label' : 'Timeline Confirmed Filtered', 'value' : 'confirmed_filtered'},
{'label' : 'Timeline Doubling Rate', 'value' : 'confirmed_DR'},
{'label' : 'Timeline Doubling Rate Filtered', 'value' : 'confirmed_filtered_DR'},
],
value = 'confirmed',
multi = False
),
dcc.Graph(figure = fig, id = 'main_window_slope')
])

@app.callback(
Output('main_window_slope', 'figure'), # property -> figure
[Input('country_drop_down', 'value'), # property -> value
Input('doubling_time', 'value')])

def update_figure(country_list, show_doubling):
    if 'confirmed_DR' in show_doubling:
        my_yaxis = {'type' : "log",
        'title' : 'Approximated doubling rate over 3 days (larger numbers are better)'
        }
    else:
        my_yaxis = {'type' : "log",
        'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
        }

    traces = [] # creating a list
    for each in country_list:
        df_plot = df_input_large[df_input_large['country'] == each]

        if show_doubling == 'confirmed_filtered_DR':
            # aggregate per date and country
            df_plot = df_plot[['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR', 'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
        else: # confimred data -> therefore add all the cases
            df_plot = df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()

        traces.append(dict(x = df_plot.date,
                           y = df_plot[show_doubling],
                           mode = 'markers+lines',
                           opacity = 0.9,
                           name = each,
                          )
                       )
    return {
    'data' : traces,
    'layout' : dict(
    width = 1280,
    height = 720,

    xaxis={'title':'Timeline',
            'tickangle':-45,
            'nticks':20,
            'tickfont':dict(size=14,color="#7f7f7f"),
          },
    yaxis = my_yaxis
    )
}


if __name__ == '__main__':
    # set debug to true to ensure we don't have to keep refreshing the server every time we make some changes
    app.run_server(debug=True, use_reloader=False)
