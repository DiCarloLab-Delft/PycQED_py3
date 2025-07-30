# -------------------------------------------
# Project root pointer
# -------------------------------------------
import os
from abc import ABCMeta
from pathlib import Path
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.absolute()
CONFIG_DIR = os.path.join(ROOT_DIR, 'data', 'class_configs')
UNITDATA_DIR = os.path.join(ROOT_DIR, 'data', 'unittest_data')
TEMP_DIR = os.path.join(ROOT_DIR, 'data', 'temp')
UI_STYLE_QSS = os.path.join(ROOT_DIR, 'style.qss')
FRAME_DIR = os.path.join(TEMP_DIR, 'frames')


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonABCMeta(ABCMeta, Singleton):
    pass
