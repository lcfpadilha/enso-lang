from setuptools import setup

setup(
    name='enso-lang',
    version='0.1.0',
    py_modules=['enso', 'compiler'],
    install_requires=[
        'lark',
        'pydantic',
    ],
    entry_points={
        'console_scripts': [
            'enso = enso:main',
        ],
    },
)