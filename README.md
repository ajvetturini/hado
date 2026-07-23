[![DOI](https://zenodo.org/badge/1310097407.svg)](https://doi.org/10.5281/zenodo.21515951)


# HADO: Hollowframe Automated DNA Origami Design 

This repository contains code for the automated design of hollowframe DNA origami nanostructures. The hado codebase is split into `app` and `core` directories corresponding to the user-interface and core algorithm functionalities, respectively. 
A more thourough description of the algorithms and methods used in this codebase can be found in the following publication and its SI material:

```commandline
Vetturini, A. J., Cagan,  J. and Taylor, R. E. 2026. Automated design of stiffness-tunable DNA origami hollowframes for self-assembling metamaterials. doi: TODO
```

### The Graphical User Interface (GUI)

The GUI is built using [streamlit](https://streamlit.io/) and allows users to easily design and visualize hollowframe DNA origami structures [online at this webpage](https://hado.streamlit.app/) without having to install anything. Overall, a user defines a list of 3D vertices in the format [(xi, yi, zi), ...] alongside an edge list in the format [(vi, vj), ...]. You also must specify the desired number of DNA helices in the bundle (which is analgous to a bundle-diameter parameter). Output files consist of standard CAD formats (caDNAno, scadnano, oxView, oxDNA) as well as the staple sequences needed for the design in a CSV format. Also, the design files that can be saved (and re-used in the GUI) are standard `json` files. 

An API for the programmatic core functionalities is also provided with documentation and tutorials found on [readthedocs here](todo). The API enables a user (or a [generative design toolkit](https://github.com/CMU-Integrated-Design-Innovation-Group/Mango)) to provide the input vertex / edge lists to enable design automation in scripted formats. 

### Local Installation

You can also run the GUI locally on your own machine, but this will require some basic knowledge of python and git. To do so:
1. Clone this repository 
```commandline
git clone https://github.com/ajvetturini/hado.git
```
2. Change into the repository directory: 
```commandline
cd hado
```
3. RECOMMENDED: Use a virtual environment to avoid dependency conflicts using venv or conda for example:
```commandline
conda create -n my_env python=3.12
conda activate my_env
```
3. Install the package / dependencies:

```bash
pip install .
```

### Running the App Locally

You should then be able to just run the following to locally run this application:

```bash
conda activate my_env  # If using a virtual enviornment, make sure you activate it
cd hado               # Make sure in hado directory to run streamlit_app.py
streamlit run streamlit_app.py
```

If you get any errors (for example, importing errors) then it is likely that `python` is pointed to an improper location. Try running this command:

```bash
python -m streamlit run streamlit_app.py
```

The app will be accessible at `http://localhost:8501` by default.
