from setuptools import setup, find_packages

setup(
    name="dinosaur",
    version="0.1.0",
    author="Aaron Medrano, Daniel Matarrita, Javier",
    description="An expert EDA library powered by pandas and matplotlib",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
        "numpy",
        "openpyxl" # For Excel support
    ],
    python_requires='>=3.6',
)
