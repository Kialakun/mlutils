from setuptools import setup, find_packages

setup(
    name="mlutils",
    version="0.1",
    packages=find_packages(),
    install_requires=["filterpy","pandas","numpy","PIL","matplotlib"], # List dependencies here
    author="gr!m",
    description="Machine Learning Utils",
    url="https://github.com/Kialakun/mlutils",
    license="MIT",
)

