import os
import os.path


modpath = os.path.dirname(os.path.abspath(__file__))
dataspath = 'datas'


def getdatadir(filename):
    return os.path.join(modpath, dataspath, filename)
