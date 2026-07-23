import streamlit as st
from hado.app.config import setup_page_config, get_custom_css
from hado.app.utils import initialize_session_state
setup_page_config()

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    pg = st.navigation([
                        st.Page("hado/app/pages/0_Home.py", title="Home",
                                icon=":material/home:", url_path="/home"),

                        st.Page("hado/app/pages/1_Hado_Automation.py", title="Design",
                                icon=":material/draw:", url_path="/hado"),

                        st.Page("hado/app/pages/2_Sticky_End_Designer.py", title="Sticky Ends",
                                icon=":material/genetics:", url_path="/sticky-ends"),

                        st.Page("hado/app/pages/3_FAQ.py", title="FAQs",
                                icon=":material/help:", url_path="/FAQs"),
                        ])
    pg.run()


if __name__ == "__main__":
    initialize_session_state()
    main()
