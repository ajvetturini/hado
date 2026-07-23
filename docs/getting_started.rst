Getting Started
===============

This page explains installation and basic usage of hado from the perspective of the UI. The procedure described below will allow you to locally run the GUI instead of accessing it via the streamlit community cloud.

Please see the API page for the Pythonic description and the Tutorial for basic usage.

Installation
------------
Currently, due to active development, hado is not available via PyPI. Instead, use the following procedure (ideally within a virtual environment such as conda):

.. code-block:: bash

   git clone https://github.com/ajvetturini/hado.git
   cd hado
   pip install -e .

Running the App Locally
-----------------------
You should then be able to just run the following to locally run this application:

.. code-block:: python

   streamlit run streamlit_app.py

If you get any errors (for example, importing errors) then it is likely that python is pointed to an improper location. Try running this command:

.. code-block:: python

   python -m streamlit run streamlit_app.py

The app will be accessible at ``http://localhost:8501`` by default.