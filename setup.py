#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import setuptools


setuptools.setup(
    name='backtrader_plotting',

    version=0.1,

    description='Plotting package for Backtrader (Bokeh, Plotly)',

    # Author details
    author='verybadsolider',
    author_email='vbs@springrts.de',

    # Choose your license
    license='GPLv3+',

    # What does your project relate to?
    keywords=['trading', 'development', 'plotting'],

    packages=setuptools.find_packages(),
    
    package_data={'backtrader_plotting.bokeh': ['templates/*.j2']},
)
