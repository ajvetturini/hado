These are build instructions for the sphinx autodocs.

1) Change directory to hado/docs
2) make clean
3) make html

OR if there are no folders inside docs (i.e., not previously built) then:
1) Change to ./hado (NOT docs)
2) sphinx-build -b html docs docs/_build/html

If there are any unique errors then whatever you recently changed broke the build process.