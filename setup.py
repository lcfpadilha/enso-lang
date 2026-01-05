from setuptools import setup, find_packages

setup(
    name='enso-lang',
    version='0.1.0',
    py_modules=['enso', 'compiler'],
    packages=find_packages(),
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