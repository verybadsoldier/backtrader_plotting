#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='backtrader_plotting',

    version='1.1.0',

    description='Plotting package for Backtrader (Bokeh)',

    python_requires='>=3.6',

    author='verybadsolider',
    author_email='vbs@springrts.de',

    long_description=long_description,
    long_description_content_type="text/markdown",

    license='GPLv3+',
    url="https://github.com/verybadsoldier/backtrader_plotting",
    project_urls={
        "Bug Tracker": "https://github.com/verybadsoldier/backtrader_plotting/issues",
        "Documentation": "https://github.com/verybadsoldier/backtrader_plotting/wiki",
        "Source Code": "https://github.com/verybadsoldier/backtrader_plotting",
        "Demos": "https://github.com/verybadsoldier/backtrader_plotting/tree/gh-pages",
    },

    # What does your project relate to?
    keywords=['trading', 'development', 'plotting', 'backtrader'],

    packages=setuptools.find_packages(),
    
    package_data={'backtrader_plotting.bokeh': ['templates/*.j2']},

    install_requires=[
        'backtrader',
        'bokeh~=2.0.0',
        'jinja2',
        'pandas',
        'matplotlib',
        'markdown2',
    ],
)
