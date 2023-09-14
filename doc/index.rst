.. cobsurv documentation master file

cobsurv  - Cobra Ensemble for Conditional Survival
===================================================

cobsurv is a project stands for Cobra Ensemble for Conditional Survival which serve the python package
**cobsurv** , cobsurv is a python package for survival analysis based on the proximity measure.
Currently it host only one method **COBRA Survival** but in future we will add different methods
based on the proximity measure. We will also provide natively build Random Forest and Newarest Neighbour
Survival, currently the package uses the [scikit-survival](https://github.com/sebp/scikit-survival/tree/master)
and [npsurvival](https://github.com/georgehc/npsurvival) for Random Forest and Nearest Neighbour Survival respectively.


The objective of the package to establish State of the Art proximity based cobsurv to predict the individual
survival function and time dependent effect of covariates.



.. grid:: 2
    :gutter: 3
    :class-container: overview-grid

    .. grid-item-card:: Install :fas:`download`
        :link: install
        :link-type: doc

        The easiest way to install scikit-survival is to use Python's package manager pip

        `pip install cobsurv`



    .. grid-item-card:: A jupyter notebook to start with COBRA Survival :fas:`book-open`
        :link: user_guide/index
        :link-type: doc

        This page contain, only one example of COBRA Survival right now but later
        we will add more examples of different methods.

    .. grid-item-card:: API Reference :fas:`cogs`
        :link: api/index
        :link-type: doc

        The reference guide contains a detailed description of the scikit-survival API. It describes which classes and functions are available
        and what their parameters are.


    .. grid-item-card:: Contributing :fas:`code`
        :link: contributing
        :link-type: doc

        If you come across a typo in the documentation or have ideas for adding new functionalities to cobsurv,
        we welcome your contributions! The contributing guidelines will assist you in the process of setting
        up a development environment and submitting your changes to the cobsurv team


.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   Install <install>
   user_guide/index
   api/index
   Contribute <contributing>
   release_notes
   Cite <cite>
