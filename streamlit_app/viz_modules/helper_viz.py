import plotly.offline as py
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.tools as tls
import plotly.figure_factory as ff
import pickle
import pandas as pd
import numpy as np
import os
import boto3
import pickle
s3 = boto3.resource('s3')

colors = ['#C8B6E2','#b59dd8', '#a081cd','#847eba', '#7A86B6','#6c81ad','#6c81ad', '#495C83']
colors1 = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
class_list=['The Eiffel Tower','airplane','apple', 'backpack','banana','bicycle','bird','book']
def get_desired_data(query_strategy,loop_number):
    data=pickle.loads(s3.Bucket("s3-bucket-link-test-delete").Object(f'ALL/{query_strategy}/metrics/test/metrics_{loop_number}.pkl').get()['Body'].read())
    return data



def get_plot_confusion_matrix(data,loop_number):
    cf=data['confusion_matrix']
    fig = ff.create_annotated_heatmap(data['confusion_matrix'],
                                  showscale=True,opacity=1,
                                  colorscale=colors,
                                  #x = ["0 (pred)","1 (pred)","2 (pred)","3 (pred)","4 (pred)","5 (pred)","7 (pred)","7 (pred)"],
                                  x = class_list,
                                  y = class_list)
    fig.layout.update({'title': f'Confusion Matrix Quick Draw for loop {loop_number}'},xaxis_title="Pred",yaxis_title="True")
    fig['layout']['xaxis']['side'] = 'bottom'
    return fig
    
    


def accuracy_metrics(data,loop_number):
    accuracy  =  data['accuracy']
    precision =  data['precision']
    recall    =  data['recall']
    f1_score  =  data ['f1_score']

    show_metrics = pd.DataFrame(data=[[accuracy , precision, recall, f1_score]])

    show_metrics = show_metrics.T
    trace2 = go.Bar(x = (show_metrics[0].values), 
                       y = ['Accuracy', 'Precision', 'Recall', 'F1_score'], text = np.round_(show_metrics[0].values,4),
                        textposition = 'auto',
                       orientation = 'h', opacity = 1,marker=dict(
                color=colors,
                line=dict(color='#000000',width=0.3)))
    fig = go.Figure(data = trace2)
    
    fig['layout'].update(
        title=f"Accuracy Metrics for loop {loop_number}")
    return fig

def acc_per_class(data,loop_number):
    per_class_list=[]
    for i,v in data['acc_per_class'].items():
        per_class_list.append(v)
    trace = go.Bar(x = per_class_list, 
                       y = class_list,
                        textposition = 'auto',
                       orientation = 'h', opacity = 1,marker=dict(
                color=colors,
                line=dict(color='#000000',width=0.3)))
    fig = go.Figure(data = trace)

    fig['layout'].update(
        title=f"Accuracy Per Class for loop {loop_number}")
    return fig
    
    
def f1_score_per_class(data,loop_number):
    per_class_list=[]
    for i,v in data['f1_score_per_class'].items():
        per_class_list.append(v)
    trace = go.Bar(x = per_class_list, 
                       y = class_list, text = '',
                        textposition = 'auto',
                       orientation = 'h', opacity = 1,marker=dict(
                color=colors,
                line=dict(color='#000000',width=0.3)))
    fig = go.Figure(data = trace)

    fig['layout'].update(
        title=f"F1 Score Per Class for loop {loop_number}")
    return fig

def precision_per_class(data,loop_number):
    per_class_list=[]
    for i,v in data['precision_per_class'].items():
        per_class_list.append(v)
    trace = go.Bar(x = per_class_list, 
                       y = class_list, text = '',
                        textposition = 'auto',
                       orientation = 'h', opacity = 1,marker=dict(
                color=colors,
                line=dict(color='#000000',width=0.3)))
    fig = go.Figure(data = trace)

    fig['layout'].update(
        title=f"Precison Per Class for loop {loop_number}")
    return fig




def comparison_function():
    #analysis_dir='../ALL'
    num_loops=10
    strats = {

        'Confidence':1,
        'random': 2,
        'Margin':3,
        'Entropy':4
    }

    qs_name=['Confidence','random','Margin','Entropy']
    accuracies = {new_list: [] for new_list in strats }

    for k in qs_name:
        for i in range(num_loops):
            data=pickle.loads(s3.Bucket("s3-bucket-link-test-delete").Object(f'ALL/{k}/metrics/test/metrics_{str(i)}.pkl').get()['Body'].read())
            accuracies[k].append(data["accuracy"])
                # with open(os.path.join('ALL',k) + "/metrics/test/metrics_" + str(i) + ".pkl", "rb") as f:
                #     data = pickle.load(f)
            
   
    df=pd.DataFrame(accuracies)
    fig = px.line(df,markers=True,
                   labels={
                         "value": "Acc",
                         "index": "Loops",
                         "variable": "Querying Strategies"
                     })
    fig['layout'].update(
            title="Accuracy comparison of Different QS")
    return fig
    
    
    
