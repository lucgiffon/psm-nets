#!/usr/bin/env python
import os
from setuptools import setup, find_packages
import sys
from pathlib import Path

NAME = 'palmnet'
DESCRIPTION = 'Applying PALM to neural network to express layers as fast-transform'
LICENSE = 'MIT'
# URL = 'https://gitlab.lis-lab.fr/qarma/{}'.format(NAME)
URL = 'https://gitlab.lis-lab.fr/luc.giffon/palmnet'
AUTHOR = 'Luc Giffon'
AUTHOR_EMAIL = ('luc.giffon@lis-lab.fr')
INSTALL_REQUIRES = ['numpy', 'daiquiri', 'matplotlib', 'pandas',
                    'docopt', 'pillow', 'scikit-learn', 'psutil', 'yafe', "click", "python-dotenv",
                    'xarray', 'keras', 'scipy==1.2.1', 'scikit-luc==2', 'tensorflow-model-optimization']
EXTRAS_REQUIRE = {
    'dev': ['coverage', 'pytest', 'pytest-cov', 'pytest-randomly'],
    'doc': ['nbsphinx', 'numpydoc', 'sphinx']}

CLASSIFIERS = [
    # 'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: MacOS :: MacOS X ',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.6']
PYTHON_REQUIRES = '>=3.7'
PROJECT_URLS = {'Bug Reports': URL + '/issues',
                'Source': URL}
KEYWORDS = 'fast transform, neural network compression'

###############################################################################
if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'\n")

if sys.version_info[:2] < (3, 7):
    errmsg = '{} requires Python 3.7 or later ({[0]:d}.{[1]:d} detected).'
    print(errmsg.format(NAME, sys.version_info[:2]))
    sys.exit(-1)


def get_version():
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    print(v_text, v_dict)
    return v_dict[NAME]


def set_version(path, VERSION):
    filename = os.path.join(path, '__init__.py')
    buf = ""
    for line in open(filename, "rb"):
        if not line.decode("utf8").startswith("__version__ ="):
            buf += line.decode("utf8")
    f = open(filename, "wb")
    f.write(buf.encode("utf8"))
    f.write(('__version__ = "%s"\n' % VERSION).encode("utf8"))


def setup_package():
    """Setup function"""
    # set version
    VERSION = get_version()

    here = Path(os.path.abspath(os.path.dirname(__file__)))
    with open(here / 'README.rst', encoding='utf-8') as f:
        long_description = f.read()

    mod_dir = Path("code") / NAME
    set_version(mod_dir, get_version())
    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=long_description,
          url=URL,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          keywords=KEYWORDS,
          packages=find_packages(where="code", exclude=['doc', 'dev']),
          package_dir={'': "code"},
          install_requires=INSTALL_REQUIRES,
          python_requires=PYTHON_REQUIRES,
          extras_require=EXTRAS_REQUIRE,
          project_urls=PROJECT_URLS)


if __name__ == "__main__":
    setup_package()