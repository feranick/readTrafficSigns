from setuptools import setup, find_packages

setup(
    name='readTrafficSigns',
    packages=find_packages(),
    install_requires=['numpy', 'h5py', 'tensorflow', 'pillow', 'opencv-contrib-python'],
    entry_points={'console_scripts' : ['readTrafficSigns=readTrafficSigns:readTrafficSigns','readTrafficSigns_GUI=readTrafficSigns_GUI:readTrafficSigns_GUI'
        ,'detectTrafficSigns=detectTrafficSigns:detectTrafficSigns']},
    py_modules=['readTrafficSigns','readTrafficSigns_GUI', 'detectTrafficSigns'],
    version='20221026a',
    description='Recognize Road Signs',
    long_description= """ Recognize Road Signs """,
    author='Nicola Ferralis',
    author_email='feranick@hotmail.com',
    url='https://github.com/feranick/readTrafficSigns',
    download_url='https://github.com/feranick/readTrafficSigns',
    keywords=['Machine learning', 'Road Signs'],
    license='GPLv2',
    platforms='any',
    classifiers=[
     'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
     'Development Status :: 5 - Production/Stable',
     'Programming Language :: Python',
     'Programming Language :: Python :: 3',
     'Programming Language :: Python :: 3.6',
     'Programming Language :: Python :: 3.7',
     'Programming Language :: Python :: 3.8',
     'Programming Language :: Python :: 3.9',
     'Programming Language :: Python :: 3.10',
     'Intended Audience :: Science/Research',
     'Topic :: Scientific/Engineering :: Physics',
     ],
)
