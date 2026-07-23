import streamlit as st


def setup_page_config():
    st.set_page_config(
        page_title="hado",
        page_icon=r'images/icon.png',
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.logo(r'images/icon_long.png', size="large", icon_image=r'images/icon.png')


def get_custom_css():
    return f"""
    <style>
        h1, h2, h3, h4, h5, h6 {{
            color: #262626 !important;
        }}

        a {{
            color: #ab7ce0 !important;        
            text-decoration: none !important; 
            font-weight: 600 !important;     
        }}

        a:hover {{
            color: rgba(107,191,216,0.733) !important;        
        }}

        [data-testid="stHeaderActionElements"] {{
            display: none;
        }}

        .stDataFrame {{
            /* width: 100%; */ 
        }}

        .stPlotlyChart {{
            min-height: 500px;
            border: 1px solid #ddd; 
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

         .stSidebar {{
            background-color: #D8DEE6; 
            border-right: 1px solid #e0e0e0;
        }} 

        div.stButton button {{
            background-color: #b3b3b3;
            color: white;
            border: none;
            border-radius: 16px;
            padding: 0.5em 1em;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }}

        div.stDownloadButton button {{
            background-color: #b3b3b3;
            color: white;
            border: none;
            border-radius: 16px;
            padding: 0.5em 1em;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }}

        div.stButton button:hover {{
            background-color: #8c8c8c;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        div.stDownloadButton button:hover {{
            background-color: #8c8c8c;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        div.stButton button:active  {{
            color: white !important;
        }}

        input[type="text"], .stSelectbox div[data-baseweb="select"] > div {{
                border-radius: 5px !important;
        }}

        div[data-testid="stMetric"] {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        div[data-testid="stMetricValue"] {{
            font-size: 1.75rem;
            text-align: center;
        }}

        details summary {{
            background-color: #DDEAF5;
            border-radius: .4rem;
            cursor: pointer;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.0);
        }}

        div[data-testid="stExpander"] details summary p {{
            color: black !important;
        }}

        details summary:hover {{
            background-color: #D8DEE6;
            box-shadow: 0 4px 8px rgba(0,0,0,0.0);
        }}

        div[data-testid="stExpander"] details summary:hover p {{
            color: black !important; 
            font-weight: bold !important;
        }}
        details summary:hover::before,
        details summary:hover::after {{
            color: black !important; 
        }}

        details summary:hover svg {{
            fill: black !important; 
        }}
        
        .stTabs [role="tab"] p {{
        color: black; 
        font-weight: 500;
        }}
        
        .stTabs [data-baseweb="tab-highlight"] {{
        background-color: #ab7ce0;
        }}

        .stTabs [role="tab"]:hover p {{
        color: darkgray;
        }}

        .stTabs [role="tab"][aria-selected="true"] p {{
        color: #ab7ce0;
        font-weight: 600;
        }}

    </style>
    """
