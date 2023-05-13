# Install libs that are not preinstalled
from turtle import title
import streamlit as st
from PIL import Image
import pandas as pd
import json
import PySIP
import requests
import numpy as np
from metalog import metalog
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import io
import time
import copy as cp
import altair as alt
import base64
from microprediction import MicroReader
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from microprediction import MicroWriter
warnings.filterwarnings('ignore')
sipmath_name = "T-Mobile Stock Value"
stock_stream_name = 'quick_yarx_tmo.json'
corr = 0.51
var_id = 2
main_title = f'One Hour Ahead Stochastic {sipmath_name} Predictions'
st.set_page_config(page_title=f"microprediction: {main_title}", page_icon=None,
                   layout="wide", initial_sidebar_state="auto", menu_items=None)


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()          


@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" height="90"; />
        </a>'''
    return html_code

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.sub-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.header-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {1}rem;
                }}
                .reportview-container .main .block-container {{
                    padding-top: {1}rem;
                }}
            </style>
            ''',unsafe_allow_html=True)

# path = os.path.dirname(__file__)
path = "."

Mircopredictions_img = get_img_with_href(
    path+'/images/micropredictions.png', 'https://micropredictions.com')
# image = Image.open('PM_logo_transparent.png')
images_container = st.container()
images_cols = images_container.columns([5, 9])

images_cols[1].header(main_title)
images_cols[1].markdown('''
    <p class="sub-font">If you can measure it, consider it predicted in real time.</p>''', unsafe_allow_html=True)
# images_cols[3].markdown(HDR_Generator, unsafe_allow_html=True)
images_cols[0].markdown(Mircopredictions_img, unsafe_allow_html=True)

graphs_container, text_container = st.container().columns([5, 9])
graphs_container_main = st.empty().container()

def remove_outliers(data):
    # Calculate the first and third quartiles (Q1 and Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75) 
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Define the bounds for outliers
    lower_bound = Q1 - 3.5 * IQR
    upper_bound = Q3 + 3.5 * IQR
    # Remove outliers and retain data within the bounds
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

def micropredictions_stock(stream_name = stock_stream_name):
    CYMBALO_COYOTE="e0a0c29acbf143899df20a20ceaf3556"
    mw = MicroWriter(write_key=CYMBALO_COYOTE)
    samples = mw.get_own_predictions(name=stream_name,delay=mw.DELAYS[-1], strip=True, consolidate=True)
    data = pd.DataFrame(remove_outliers(samples), columns=[sipmath_name])
    # step = 1 / data.shape[0]
    # data.index = (data.index + 1)*step
    return data

def micropredictions_S_P():
    HEBDOMAD_LEECH='8c386f8221c950008bad5221e9d4ada6'
    mw = MicroWriter(write_key=HEBDOMAD_LEECH)
    stream_name = 'rdps_spy.json'
    samples = mw.get_own_predictions(name=stream_name,delay=mw.DELAYS[-1], strip=True, consolidate=True)
    data = pd.DataFrame(remove_outliers(samples), columns=["S&P"])
    # step = 1 / data.shape[0]
    # data.index = (data.index + 1)*step
    return data

def plot(m, big_plots=None, csv=None, term=None, name=None, key=None):
    # st.write(m)
    # print("is_quantile is ",is_quantile," csv is ",csv)
    # if is_quantile or not csv:
    # key = 'quantile'
    # else:
    # key = 'csv'
    # if res_data
    # Collecting data to set limits of axes
    print(f"running plot for {name} in {key}")
    if 'res_data' not in st.session_state['mfitted'][key][name]:
        # st.write("notthere")
        res_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound'])
                                                   + ' Terms', len(m['M'].iloc[:, 0])),
                                 'pdfValues': m['M'].iloc[:, 0],
                                 'quantileValues': m['M'].iloc[:, 1],
                                 'cumValue': m['M']['y']
                                 })
        if m['M'].shape[-1] > 3:
            for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
                if m['Validation']['valid'][i] == 'yes':
                    # st.write(i)
                    temp_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound'] + i - 1)
                                                                + ' Terms', len(m['M'].iloc[:, 0])),
                                              'pdfValues': m['M'].iloc[:, i * 2 - 2],
                                              'quantileValues': m['M'].iloc[:, i * 2 - 1],
                                              'cumValue': m['M']['y']})
                    res_data = pd.concat(
                        [res_data, temp_data], ignore_index=True)
        res_data['frames'] = res_data['term']
        res_data['groups'] = res_data['term']
        st.session_state['mfitted'][key][name]['res_data'] = res_data
    else:
        res_data = st.session_state['mfitted'][key][name]['res_data']

    # Collecting data into dictionary
    InitialResults = {}
    InitialResults[str(m['params']['term_lower_bound']) + ' Terms'] = pd.DataFrame({
        'pdfValues': m['M'].iloc[:, 0],
        'quantileValues': m['M'].iloc[:, 1],
        'cumValue': m['M']['y']
    })
    if m['M'].shape[-1] > 3:
        for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
            InitialResults[str(m['params']['term_lower_bound'] + i - 1) + ' Terms'] = pd.DataFrame({
                'pdfValues': m['M'].iloc[:, i * 2 - 2],
                'quantileValues': m['M'].iloc[:, i * 2 - 1],
                'cumValue': m['M']['y']
            })

    # ggplot style
    plt.style.use('ggplot')
    max_valid_term = m['Validation'][m['Validation']
                                     ['valid'] == 'yes']['term'].max()

    # st.write(m['M'])
    # st.write(InitialResults)

    results_len = len(InitialResults)

    if big_plots:
        fig, ax = plt.subplots()
        # if st.session_state['mfitted'][key][name]['fit']
        # fig, ax = plt.subplots(1, 2, figsize=(4, 2), sharex='col')
        # i = 2
        if term is None:
            pass
        else:
            terms_for_loop = [term]
            # for i in range(2,term+1):
            for i in terms_for_loop:
                # print(m['Validation']['valid'])
                # print("i is",i)
                if m['Validation']['valid'][i] == 'yes':
                    j = 0
                    # Plotting PDF
                    # ax[j].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['pdfValues'],
                    #            linewidth=2, c='darkblue')
                    # ax[j].patch.set_facecolor('white')
                    # ax[j].axes.yaxis.set_ticks([])
                    plt.rcParams["figure.figsize"] = [7.00, 3.50]
                    plt.rcParams["figure.autolayout"] = True
                    im = plt.imread(path+'/images/SIPmath Standard Certified.png') # insert local path of the image.
                    ax.plot(InitialResults[str(i) + ' Terms']['quantileValues'], 
                                InitialResults[str(i) + ' Terms']['pdfValues'],
                                linewidth=2, 
                                c='darkblue')
                    ax.patch.set_facecolor('white')
                    ax.axes.yaxis.set_ticks([])
                    ax.set(title=sipmath_name, xlabel='Basis Points')
                    newax = fig.add_axes([0.5,0.5,0.5,0.5], anchor=(0.59, 0.15), zorder=1)
                    newax.imshow(im)
                    newax.axis('off')
                    
            plt.tight_layout(rect=[0, 0, 0.75, 1])
            # graphs_container.subheader('ER Wait Time in 1 Hour')
            graphs_container.pyplot(plt)
            if st.session_state['mfitted'][key][name]['plot']['big plot'] is None:
                temp_img = io.BytesIO()
                plt.savefig(temp_img, format='png',
                            transparent=True, unsafe_allow_html=True)
                st.session_state['mfitted'][key][name]['plot']['big plot'] = temp_img
    if csv:
        if st.session_state['mfitted'][key][name]['plot']['csv'] is None:
            fig, ax = plt.subplots(3, 5, figsize=(10, 5), sharex='col')
            for i in range(2, 4 + 1):
                for j in range(0, 5):
                    current_term = (2 + (i - 2)*5 + j)
                    print(f"{current_term}")
                    # Check to make sure it is valid before plotting.
                    if results_len + 2 > current_term and m['Validation']['valid'][current_term] == 'yes':
                        print(f"plotting {current_term}")
                        # Plotting PDF
                        ax[i-2, j].plot(InitialResults[str(current_term) + ' Terms']['quantileValues'], InitialResults[str(current_term) + ' Terms']['pdfValues'],
                                        linewidth=2, c='darkblue')

                    else:  # if not valid plot nothing
                        # Plotting blank PDF chart
                        # ax[i-2, 0].plot()
                        # Plotting blank CDF chart
                        ax[i-2, j].plot()
                    # Axes setup
                    # if norm:
                    # ax[i-2, j].axis([min(res_data['quantileValues']), max(res_data['quantileValues']),
                        # round(min(m["dataValues"]['probs']),1), round(max(m["dataValues"]['probs']),1)])
                    ax[i-2, j].patch.set_facecolor('white')
                    ax[i-2, j].axes.xaxis.set_ticks([])
                    ax[i-2, j].axes.yaxis.set_ticks([])
                    if current_term < 11:
                        ax[i-2, j].set(title=str(current_term) +
                                       ' Terms', ylabel='PDF')
                        # ax[i-2, j].patch.set()
                    else:
                        ax[i-2, j].set(title=str(current_term) +
                                       ' Terms', ylabel='PDF', xlabel='Quantiles')

                        # ax[i-2, j].patch.set(title=str(current_term) + ' Terms', ylabel='PDF', xlabel='Quantiles')

                    # if current_term != 5*3:
                        # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF')
                    # else:
                        # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF', xlabel='Quantiles')

            plt.tight_layout(rect=[0, 0, 0.75, 1])
            temp_img = io.BytesIO()
            plt.savefig(temp_img, format='png',
                        transparent=True, unsafe_allow_html=True)
            st.session_state['mfitted'][key][name]['plot']['csv'] = temp_img

            graphs_container_main.image(
                st.session_state['mfitted'][key][name]['plot']['csv'], use_column_width=True)
            print("looped")
        else:
            graphs_container_main.image(
                st.session_state['mfitted'][key][name]['plot']['csv'], use_column_width=True)

    # graphs_container.pyplot(plt)
    # return plt


def convert_to_JSON(input_df,
                    filename,
                    author,
                    dependence,
                    boundedness,
                    bounds,
                    term_saved,
                    probs,
                    quantile_corr_matrix,
                    seeds ):

    PySIP.Json(SIPdata=input_df,
               file_name=filename,
               author=author,
               dependence=dependence,
               boundedness=boundedness,
               bounds=bounds,
               term_saved=term_saved,
               probs=probs,
               quantile_corr_matrix=quantile_corr_matrix,
               seeds=seeds
               )

    with open(filename) as f:
        graphs_container.download_button(
            label=f"Download {filename}",
            data=f,
            file_name=filename
        )
    return True


def preprocess_charts(x,
                      probs,
                      boundedness,
                      bounds,
                      big_plots,
                      terms,
                      csv,
                      name,
                      user_term):
    # Create metalog
    # st.write(boundedness,
    # bounds)
    if 'mfitted' not in st.session_state:
        st.session_state['mfitted'] = {'csv': {}, 'quantile': {}}
    if probs is np.nan:
        key = 'csv'
    else:
        key = 'quantile'
    # update_boundedness(False)
    if (name not in st.session_state['mfitted'][key] or st.session_state['mfitted'][key][name]['fit'] is None) or (name in st.session_state['mfitted'][key] and not user_term is None and st.session_state['mfitted'][key][name]['fit']['Validation']['term'].max() < user_term):
        print(f"running metalog fit for {name} in {key}")
        mfitted = metalog.fit(x, bounds=bounds, boundedness=boundedness,
                              fit_method='OLS', term_limit=terms, probs=probs)
        # max_valid_term = int(mfitted['Validation'][(mfitted['Validation']['valid'] == 'yes') & (mfitted['Validation']['term'] <= user_term)]['term'].max())
        st.session_state['mfitted'][key][name] = {'fit': mfitted, 'plot': {'csv': None, 'big plot': None}, 'options': {
            'boundedness': boundedness, 'terms': user_term, 'bounds': bounds}}
    print("user term is", user_term)
    
    plot(st.session_state['mfitted'][key][name]['fit'],
         big_plots, csv, user_term, name=name, key=key)

def get_micropredictions():
    HAMOOSE_CHEETAH = '612a4363e8ba2100de3d12e077d0b13e'
    NAME = 'noaa_wind_speed_46073.json'
    mr = MicroReader()
    predictions = mr.get_predictions(name=NAME,write_key=HAMOOSE_CHEETAH,delay=mr.DELAYS[-1])
    print(predictions)
    return predictions

def get_micropredictions_SP():
    HAMOOSE_CHEETAH = '612a4363e8ba2100de3d12e077d0b13e'
    NAME = 'noaa_wind_speed_46073.json'
    mr = MicroReader()
    predictions = mr.get_predictions(name=NAME,write_key=HAMOOSE_CHEETAH,delay=mr.DELAYS[-1])
    print(predictions)
    return predictions

def get_stock_value():
    #	https://api.microprediction.org/live/hospital-er-wait-minutes-piedmont_mountside_ellijay.json
    stock_value_lagged_url = "https://api.microprediction.org/lagged/quick_yarx_googl.json"
    r = requests.get(stock_value_lagged_url, timeout=30)
    if r.ok:
        stock_value_json = r.json()
        stock_value_df = pd.DataFrame(stock_value_json, columns=['timeStamp','stock_value'])
        print(stock_value_df)
        return stock_value_df[['stock_value']]
        # return stock_value_df.loc[((stock_value_df['timeStamp'] >= (stock_value_df['timeStamp'][0] - 86400)) & (stock_value_df['stock_value'] > 0)), ['stock_value']]
        # return stock_value_df.loc[((stock_value_df['timeStamp'] >= (stock_value_df['timeStamp'][0] - 1*86400))), ['stock_value']]
    else:
        return pd.DataFrame()


def sent_to_pastebin(filename, file):
    payload = {"api_dev_key": '7lc7IMiM_x5aMUFFudCiCo35t4o0Sxx6',
               "api_paste_private": '1',
               "api_option": 'paste',
               "api_paste_name": filename,
               "api_paste_expire_date": '10M',
               "api_paste_code": file,
               "api_paste_format": 'json'}
    url = 'https://pastebin.com/api/api_post.php'
    r = requests.post(url, data=payload)
    return r


def convert_to_number(value):

    if isinstance(value, dict):
        value = {k: float(v) if isinstance(v, str) and (v.isnumeric() or (
            data_type_str == 'csv' and v != 'PM_Index')) else v for k, v in value.items()}
    return value


def update_max_term(variable_index, variable_name):
    if 'quantiles_data' in st.session_state and variable_name in st.session_state['quantiles_data'] and 'number_of_quantiles' in st.session_state['quantiles_data'][variable_name]:
        st.session_state['quantiles_data'][variable_name][
            'number_of_quantiles'] = st.session_state[f"Quantile{variable_index}"]
        st.session_state['quantiles_data'][variable_name].pop(
            'col_terms', None)
        st.session_state['quantiles_data'][variable_name].pop('q_data', None)


def update_terms(selected_column, data_type='csv', variable_index=0):
    if data_type == 'csv':
        value = st.session_state["Column_Terms"]
        if 'mfitted' in st.session_state:
            if selected_column not in st.session_state['mfitted'][data_type]:
                print("selected_column", selected_column)
                # st.session_state['mfitted'][data_type][selected_column]['options']['terms'] = value
            elif st.session_state['mfitted'][data_type][selected_column]['options']['terms'] != value:
                st.session_state['mfitted'][data_type][selected_column]['options']['terms'] = value
    else:
        if 'quantiles_data' in st.session_state and selected_column in st.session_state['quantiles_data'] and 'col_terms' in st.session_state['quantiles_data'][selected_column]:
            st.session_state['quantiles_data'][selected_column]['col_terms'] = st.session_state[
                f"Column_Terms {selected_column} {variable_index}"]


def update_values(variable_name, session_key, row):
    if 'quantiles_data' in st.session_state and variable_name in st.session_state['quantiles_data'] and 'q_data' in st.session_state['quantiles_data'][variable_name]:
        # st.session_state['quantiles_data'][quantile_name[quantile_index]]['q_data']
        df = st.session_state['quantiles_data'][variable_name]['q_data'].reset_index(
        )
        col = 0 if session_key[0] == 'y' else 1
        print("value in df is", df.iloc[row, col])
        df.iloc[row, col] = st.session_state[session_key]
        print("value in df is", df.iloc[row, col])
        print("df is", df.set_index(""))
        st.session_state['quantiles_data'][variable_name]['q_data'] = df.set_index(
            "")
        if 'mfitted' in st.session_state and variable_name in st.session_state['mfitted'][data_type_str]:
            st.session_state['mfitted'][data_type_str].pop(variable_name, None)


def update_correlations(session_key):
    if 'quantiles_data' in st.session_state and 'correlations' in st.session_state['quantiles_data']:
        # st.session_state['quantiles_data'][quantile_name[quantile_index]]['q_data']
        df = st.session_state['quantiles_data']['correlations']
        col, row = session_key.split(" vs ")
        print("col,row", col, row)
        print("value in df is", df.loc[row, col])
        df.loc[row, col] = st.session_state[session_key]
        print("value in df is", df.loc[row, col])
        print("df is", df)
        st.session_state['quantiles_data']['correlations'] = df


def update_input_name():
    print("Big Graph Column is", st.session_state["Big Graph Column"])
    st.session_state["quantile_counter"] = st.session_state['quantiles_data'][st.session_state["Big Graph Column"]]['pos']


def update_variable_count():
    if "Number of Quantiles Variables" in st.session_state:
        st.session_state['quantiles_variable_count'] = st.session_state["Number of Quantiles Variables"]


def update_counter(value):
    if data_type_str == 'quantile':
        if 'quantile_counter' in st.session_state and 'Number of Quantiles Variables' in st.session_state:
            if (value == -1 and st.session_state['quantile_counter'] == 1) or (value == 1 and st.session_state['quantile_counter'] == st.session_state['Number of Quantiles Variables']):
                return None
            st.session_state['quantile_counter'] += value
        else:
            st.session_state['quantile_counter'] = 1
    else:
        if 'csv_counter' in st.session_state and 'column_index' in st.session_state:
            if (value == -1 and st.session_state['csv_counter'] == 1) or (value == 1 and st.session_state['csv_counter'] == len(st.session_state["column_index"])):
                return None
            st.session_state['csv_counter'] += value
            print("keys", list(st.session_state["column_index"].keys())[
                  st.session_state['csv_counter'] - 1])
            st.session_state["Big Graph Column"] = list(st.session_state["column_index"].keys())[
                st.session_state['csv_counter'] - 1]
        else:
            st.session_state['csv_counter'] = 1


def update_name(num):
    new_name = st.session_state[f"Quantile Name {num}"]
    if not new_name in st.session_state['quantiles_data']:
        for variable in st.session_state['quantiles_data']:
            if st.session_state['quantiles_data'][variable]['pos'] == num:
                st.session_state['quantiles_data'][new_name] = st.session_state['quantiles_data'].pop(
                    variable)
                if 'mfitted' in st.session_state and variable in st.session_state['mfitted'][data_type_str]:
                    st.session_state['mfitted'][data_type_str][new_name] = st.session_state['mfitted'][data_type_str].pop(
                        variable, None)
                break
    pass


def update_boundedness(refresh=False, data_type='csv', max=1, min=0, quantile_count=None, variable_name=None):
    if quantile_count is None:
        boundedness = st.session_state["Column_boundedness"]
    else:
        boundedness = st.session_state[f"Column_boundedness {variable_name} {quantile_count+1}"]

    selected_column = st.session_state["Big Graph Column"] if data_type == 'csv' else variable_name
    print("boundedness from session is ", boundedness)
    print("selected_column from session is ", selected_column)
    if "Column_upper" not in st.session_state:
        print("Column_upper in session")
        upper = max
    else:
        upper = st.session_state["Column_upper"]
    if "Column_lower" not in st.session_state:
        print("Column_lower in session")
        lower = min
    else:
        lower = st.session_state["Column_lower"]

    # convert to float and list
    if boundedness == "'b' - bounded on both sides":
        bounds = [lower, upper]
    elif boundedness.find("lower") != -1:
        bounds = [lower]
    elif boundedness.find("upper") != -1:
        bounds = [upper]
    else:
        bounds = [0, 1]
    boundedness = boundedness.strip().split(" - ")[0].replace("'", "")

    if 'quantiles_data' in st.session_state and selected_column in st.session_state['quantiles_data'] and 'boundedness' in st.session_state['quantiles_data'][selected_column]:
        st.session_state['quantiles_data'][selected_column]['boundedness'] = boundedness
        st.session_state['quantiles_data'][selected_column]['bounds'] = bounds

    if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type]:
        if selected_column not in st.session_state['mfitted'][data_type] or (st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] != boundedness or st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] != bounds):
            # print("selected_column",selected_column)
            if any([x[0] != x[1] for x in zip(st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'], boundedness)]):
                print("saved", st.session_state['mfitted'][data_type][selected_column]
                      ['options']['boundedness'], "current boundedness", boundedness)
                st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] = boundedness
                refresh = True
            if any([float(x[0]) != float(x[1]) for x in zip(st.session_state['mfitted'][data_type][selected_column]['options']['bounds'], bounds)]):
                print("saved", st.session_state['mfitted'][data_type]
                      [selected_column]['options']['bounds'], "current bounds", bounds)
                st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] = bounds
                print("saved after saving", st.session_state['mfitted'][data_type]
                      [selected_column]['options']['bounds'], "current bounds", bounds)
                refresh = True
        # elif :
            # st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] = boundedness
            # st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] = boundedness
            # st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] = bounds
            # st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] = bounds
            # TODO: recalculate when bounds change
    if refresh and 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type]:
        st.session_state['mfitted'][data_type][selected_column]['fit'] = None
        # st.session_state['mfitted']['quantile'][selected_column]['fit'] = None
        st.session_state['mfitted'][data_type][selected_column]['plot'] = {
            data_type: None, 'big plot': None}
        # st.session_state['mfitted']['quantile'][selected_column]['plot'] = {data_type:None,'big plot':None}


def update_seeds(data_type='csv',  entity=None,  varId=None,  seed3=None,  seed4=None, variable_name=None):
    selected_column = st.session_state["Big Graph Column"] if data_type == 'csv' else variable_name

    if data_type != 'csv' and 'quantiles_data' in st.session_state and selected_column in st.session_state['quantiles_data'] and 'seeds' in st.session_state['quantiles_data'][selected_column]:
        for item in ['entity', 'seed3', 'seed4']:
            if st.session_state[f"{item} {selected_column}"].isnumeric():
                st.session_state['quantiles_data'][selected_column]['seeds'][
                    'arguments'][item] = st.session_state[f"{item} {selected_column}"]
            else:
                st.warning(f'{item} must be a number.')
                # st.stop()
        if st.session_state[f"varId {selected_column}"].strip():
            st.session_state['quantiles_data'][selected_column]['seeds'][
                'arguments']['varId'] = st.session_state[f"varId {selected_column}"]
        else:
            st.warning(f'varId must have a value.')
            # st.stop()

        st.session_state['quantiles_data'][selected_column]['seeds']['arguments'] = {'counter': 'PM_Index',
                                                                                     'entity': st.session_state[f"entity {selected_column}"],
                                                                                     'varId': st.session_state[f"varId {selected_column}"],
                                                                                     'seed3': st.session_state[f"seed3 {selected_column}"],
                                                                                     'seed4': st.session_state[f"seed4 {selected_column}"]}

    if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type] and 'seeds' in st.session_state['mfitted'][data_type][selected_column]['options']:
        if not entity is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['entity'] = entity
        elif not varId is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['varId'] = varId
        elif not seed3 is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['seed3'] = seed3
        elif not seed4 is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['seed4'] = seed4
        elif selected_column in st.session_state['mfitted'][data_type]:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments'] = {'counter': 'PM_Index',
                                                                                                        'entity': st.session_state[f"entity {selected_column}"],
                                                                                                        'varId': st.session_state[f"varId {selected_column}"],
                                                                                                        'seed3': st.session_state[f"seed3 {selected_column}"],
                                                                                                        'seed4': st.session_state[f"seed4 {selected_column}"]}


def make_csv_graph(series,
                   probs,
                   boundedness,
                   bounds,
                   big_plots,
                   user_terms,
                   graphs):
    if big_plots:
        graphs_container.markdown(
            f"<div id='linkto_head'></div>", unsafe_allow_html=True)
        # graphs_container.header(series.name)
        print(probs)
    preprocess_charts(series.to_list(),
                      probs,
                      boundedness,
                      bounds,
                      big_plots,
                      16 if probs is np.nan else user_terms,
                      graphs,
                      series.name,
                      user_terms)

    return None
# @st.cache

# col_name = 'Stock_Value'
# micro_data = get_micropredictions()
# micro_data_df = pd.DataFrame([ p for p in micro_data if p > 0.01 ], columns=[col_name])
SP_data = micropredictions_S_P()
stock_data = micropredictions_stock()
SP_data_stats = SP_data.describe()
stock_data_stats = stock_data.describe()
micro_data_df = pd.concat([SP_data_stats.loc[['25%', '50%','75%']], 
                           stock_data_stats.loc[['25%', '50%','75%']]], 
                           axis=1)*10
micro_data_df.index = [0.25, 0.5, 0.75]
# micro_data_df = get_nyc_data()
# print(micro_data_df.dtypes)
# print(micro_data_df.dtypes)
micro_data_df.columns = micro_data_df.columns.str.replace(' |&', '_')
name = micro_data_df.columns[-1]
seeds = [
            {
                "name": "hdr1",
                "function": "HDR_2_0",
                "arguments": {
                    "counter": "PM_Index",
                    "entity": 1,
                    "varId": 0,
                    "seed3": 0,
                    "seed4": 0
                }
            },
            {
                "name": "hdr2",
                "function": "HDR_2_0",
                "arguments": {
                    "counter": "PM_Index",
                    "entity": 1,
                    "varId": var_id,
                    "seed3": 0,
                    "seed4": 0
                }
            }
        ]
# table_container.subheader(f"Preview for {name}")
# table_container.write(micro_data_df[:10].to_html(
#     index=False), unsafe_allow_html=True)
probs=micro_data_df.index
boundedness='u'
# bounds=[micro_data_df.iloc[:,0].min()]
bounds=[micro_data_df.iloc[:,0].min()-0.01, 1.25*micro_data_df.iloc[:,0].max()]
# bounds=[0, 1]
big_plots=True
user_terms=3
graphs=False
dependence = 'dependent'
file_name = f'{name}.SIPmath'
micro_data_df[[name]].apply(make_csv_graph,
                probs=probs,
                boundedness=boundedness,
                bounds=bounds,
                big_plots=big_plots,
                user_terms=user_terms,
                graphs=graphs)
corrs_data = [[1,None],[corr,1]]            
correlation_df = pd.DataFrame(corrs_data,columns=micro_data_df.columns,index=micro_data_df.columns)
print('correlation_df is ', correlation_df)
text_container.markdown('''
    <p class="big-font"></p>''', unsafe_allow_html=True)
text_container.markdown('''
    <p class="big-font"></p>''', unsafe_allow_html=True)
text_container.markdown('''
<p class="big-font">Microprediction’s algorithms deliver forecasts as stochastic (SIP) libraries in the open <a href="https://www.probabilitymanagement.org/30-standard">SIPmath™ 3.0 Standard</a>, so they may be used in Monte Carlo or other calculations in R, Python or Excel using <a href="https://www.probabilitymanagement.org/chancecalc">ChanceCalc™</a>.</p>
''', unsafe_allow_html=True)
text_container.markdown('''
<p class="big-font">For background on stochastic information packets (SIPs) see the <a href="https://en.wikipedia.org/wiki/Probability_management">Wikipedia entry on probability management</a>. The <a href="https://micropredictions.com">microprediction site</a> can be viewed as a source of real-time SIPs.</p>
''', unsafe_allow_html=True)

convert_to_JSON(micro_data_df,
                file_name,
                name,
                dependence,
                boundedness,
                bounds,
                user_terms,
                probs,
                correlation_df,
                seeds)
     
copy_button = Button(label=f"Copy {file_name} to clipboard.")
with open(file_name, 'rb') as f:
    copy_button.js_on_event("button_click", CustomJS(args=dict(data=str(json.load(f))), code="""
        navigator.clipboard.writeText(data);
        """))

no_event = streamlit_bokeh_events( 
    copy_button,
    events="GET_TEXT",
    key="get_text",
    refresh_on_update=True,
    override_height=50,
    debounce_time=0)
# if text_container.button(f'Display {file_name}'):
#     text_container.text("Mouse over the text then click on the clipboard icon to copy to your clipboard.")
#     
#         text_container.json(json.load(f))