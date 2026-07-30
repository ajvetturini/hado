import streamlit as st


if __name__ == '__main__':
    st.title("FAQ: Frequently Asked Questions / Quick Demonstrations")
    st.write('This page is a constant work-in-progress, if you have questions that are not answered here, '
             'please reach out to us via [email](mailto:avetturi@andrew.cmu.edu) or GitHub issues!')

    with st.expander(expanded=False, label='Basic Questions',):
        st.markdown("**Q: How do I cite this tool?**")
        st.markdown('The article is currently undergoing peer-review but has been deposited into bioRxiv:')
        st.code('Vetturini, Anthony Joseph, Jonathan Cagan, and Rebecca E. Taylor. “Automated Design of Preprint, '
                'bioRxiv, July 24, 2026. https://doi.org/10.64898/2026.07.23.740378.', language="text")

        st.markdown("**Q: How do I access the Python API?**")
        st.markdown("**A**: There are installation instructions along with an example Jupyter notebook found in the "
                    "readthedocs [here](https://hado.readthedocs.io/en/latest/) ")

        st.markdown("**Q: What buffer conditions should I use for hollowframe DNA origami?**")
        st.write("**A**: The structures shown in the journal article are in 1X TAE (~8.3 pH), 12 mM MgCl$_2$, and use "
                 "100 nM of a staple mix (calibrated to 1000 nM) with 10 nM of scaffold.")

        st.markdown("**Q: What annealing ramp should I use for individual monomers?**")
        st.markdown("""
        **A:** All structures shown in the journal article use the following 24 hour annealing protocol:
        * **95°C** for 5 minutes
        * **70°C** for 15 min 
        * **70°C to 30°C** at -1C every 30 min
        * **30°C to 20°C** at -1C / 5 min
        * **Hold** at **4°C**
        """)

        st.markdown("**Q: What annealing ramp should I use for one-pot assembly?**")
        st.markdown("""
        **A:** All assembled materials use the following 3.25-day ramp:
        * **95°C** for 5 minutes
        * **65°C** for 2 hours
        * **65°C to 35°C** at -0.5C every 1 hour 
        * **35°C to 20°C** at -1C / 15 min
        * **Hold** at **4°C**
        
        Furthermore, when doing the one-pot, a 7.5X excess of the sticky end staples are included in the sample
        preparation alongside 12 mM "MgCl$_2$ and 1X TAE (~8.3 pH).
        """)

        # st.markdown("**Q: What are some other DNA origami automation algorithms?**")
        # st.markdown("""
        # **A:** Below is a list of some other DNA origami **automation** CAD packages that I have used. These tools all
        # offer varying degrees of automation (some may fully automate scaffold + staple design, others may require
        # more of your nown manual design decisions). I distinguish these using (F) for fully-automated, (S) for
        # semi-automated, and (M) for minimal-automation capabilities out-of-the-box.
        # * (F) [ATHENA](https://academic.oup.com/nar/article/49/18/10265/6368527) (2HB and 6HB, 2D / 3D wireframe DNA origami. Compiles DAEDALUS, METIS, PERDIX, and TALOS into one).
        # * (F) [DNAForge](https://academic.oup.com/nar/article/52/W1/W13/7673483) (1/2HB, tiles, RNA origami. Contains modern implementation of vHelix)
        # * (S) [MagicDNA](https://www.nature.com/articles/s41563-021-00978-5) (Sandbox of design features around DNA origami)
        # * (S) [ENSNano](https://github.com/thenlevy/ensnano) (Many automated components, has support for curved DNA origami)
        # * (S) [PyFuRNAce](https://www.nature.com/articles/s41467-025-66290-x) (Thorough RNA origami design engine)
        # * (S) [InSequio](https://www.biorxiv.org/content/10.1101/2024.03.27.586810v1) (Commercial DNA origami CAD package)
        # * (M) [caDNAno](https://academic.oup.com/nar/article/37/15/5001/2409858?guestAccessKey=) (Most popular / the original DNA origami CAD software)
        # * (M) [scadnano](https://github.com/UC-Davis-molecular-computing/scadnano) (Scriptable and more generalized version of caDNAno that can extend to other nucleic acids)
        #
        # If I am missing your work and you want it listed, please let me know and I'll happily add it.
        # """)

    with st.expander(expanded=False, label='Design Error Questions'):
        st.markdown("**Q: What geometries cause issues with this tool?**")
        st.markdown("""
        **A**: Thus far, I have found that these types of topological features may lead to issues:
        
        - Geometries with sharp angles (~<15-20 deg) between edges when using low-numbers of N per helix bundle. 
        - Geometries with sharp angles and short edges, sharp angles typically requires much longer edges
        - Large geometries. The algorithms here are scaled to traditional DNA origami sized-structures (~10K nts), and some (such as the autostapling) do not scale well to ultra-large structures (~50K+ nts). 
        - Very complex geometries (~40+ edges). These types of geometries could (currently) only be manufactured realistically using 1 or 2 helices per edge, thus other tools (vHelix, DAEDALUS) are more well-suited.
        """)

        st.markdown("**Q: Why does this take a while to run?**")
        st.markdown("""
        **A**: Presuming you structure is appropriately sized, the streamlit community cloud only provides so much 
        computational resources, so my first recommendation would be to try and run the app locally. If issues still 
        persist, please open a [GitHub issue](https://github.com/ajvetturini/hado/issues/new) with the *.hado file 
        attached to further investigate what issues might be popping up.
        """)

        st.markdown("**Q: How do I report an issue?**")
        st.markdown("""
                **A**: Please open a [GitHub issue](https://github.com/ajvetturini/hado/issues/new) and attach the 
                *.hado file so I can investigate. I am an academic (i.e., not a web app dev), so there will be rough 
                edges here (but rough edges can be fixed with feedback!). 
                """)


    # Demo videos (as a note-to-self, currently all videos are on the 19... youtube account)
    with st.expander(expanded=False, label="30-second demo of app functionalities"):
        st.video('https://youtu.be/oEATQ5DcRzM',
                 format="video/mp4",
                 start_time=0,
                 subtitles=None,
                 loop=False,
                 autoplay=False,
                 muted=True)

    with st.expander(expanded=False, label="10-minute discussion of core functionalities"):
        st.write('**Note**: Audio is used in this video.')
        st.video('https://youtu.be/jZ69819Izac',
                 format="video/mp4",
                 start_time=0,
                 subtitles=None,
                 loop=False,
                 autoplay=False,
                 muted=False)

    with st.expander(expanded=False, label="Running hado locally"):
        st.write('**Note**: Audio is used in this video.')
        st.video(r'https://youtu.be/lS1KgkD6fcs',
                 format="video/mp4",
                 start_time=0,
                 subtitles=None,
                 loop=False,
                 autoplay=False,
                 muted=False)

    with st.expander(expanded=False, label="Using the sticky end designer"):
        st.write('**Note**: Audio is used in this video.')
        st.video(r'https://youtu.be/ywL8AB8HdNI',
                 format="video/mp4",
                 start_time=0,
                 subtitles=None,
                 loop=False,
                 autoplay=False,
                 muted=False)
