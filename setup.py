#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='backtrader_plotting',

    version='1.0.0',

    description='Plotting package for Backtrader (Bokeh)',

    python_requires='>=3.6',

    author='verybadsolider',
    author_email='vbs@springrts.de',

    long_description=long_description,
    long_description_content_type="text/markdown",

    license='GPLv3+',

    # What does your project relate to?
    keywords=['trading', 'development', 'plotting'],

    packages=setuptools.find_packages(),
    
    package_data={'backtrader_plotting.bokeh': ['templates/*.j2']},

    install_requires=[
        'backtrader',
        'bokeh~=1.4.0',
        'jinja2',
        'pandas',
        'matplotlib',
        'markdown2',
    ],
)
