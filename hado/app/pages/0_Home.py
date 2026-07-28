import streamlit as st

from hado.app.utils import apply_page_width

if __name__ == '__main__':
    apply_page_width()
    PROJECT_NAME = 'hado'  # HADO = HollowfrAme DNA Origami
    st.title(f"{PROJECT_NAME}: Design automation tool for DNA origami nanostructures with hollow cross-sections")
    mango = "https://github.com/CMU-Integrated-Design-Innovation-Group/Mango"

    st.write("## About")
    st.markdown(f"This is a web-based application in support of the <u>H</u>ollowfr<u>A</u>me <u>D</u>NA "
                f"<u>O</u>rigami (*hado*) automation package. This will let you create custom input mesh files by "
                f"defining simple vertices-and-edges (thus is not limited to closed-surface meshes like other "
                f"automated DNA origami algorithms). You can also start designing using one of the starting meshes, "
                f"the Python API, or a generative design package for DNA origami ([mango]({mango})). This UI was "
                f"built using [streamlit](https://streamlit.io/)", unsafe_allow_html=True)

    st.markdown(f" - **GitHub**: [{PROJECT_NAME}](https://github.com/ajvetturini/hado/)")
    st.markdown(" - **Documentation & Python API**: [Read the Docs](https://hado.readthedocs.io/en/latest/)")
    st.markdown(" - **FAQs**: [Common questions](./FAQs)")
    st.markdown(" - **Our Research Groups / Contact Information**: "
                "[Microsystems and MechanoBiology Lab](https://www.andrew.cmu.edu/user/bex/pages/welcome/) "
                "| "
                "[Integrated Design Innovation Group](https://www.cmu.edu/me/idig/)")
    st.markdown(" - If you notice any bugs or have specific feature requests, please open an issue via "
                "[GitHub Issues](https://github.com/ajvetturini/hado/issues)")

    st.write("## A 30 second demo of the app functionalities")
    st.video('https://youtu.be/oEATQ5DcRzM',
             format="video/mp4",
             start_time=0,
             subtitles=None,
             loop=False,
             autoplay=True,
             muted=True)
