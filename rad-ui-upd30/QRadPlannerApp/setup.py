from setuptools import setup, find_packages

setup(
    name="QRadPlannerApp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pydicom>=2.3.0",
        "vtk>=9.0.0",
        "scikit-image>=0.18.0"
    ],
    entry_points={
        'console_scripts': [
            'qradplanner = QRadPlannerApp.main:run_gui_app'
        ]
    }
)
