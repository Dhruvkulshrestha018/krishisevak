from setuptools import setup, find_packages

setup(
    name="krishisevak",
    version="0.0.1",
    author="Dhruv Kulshrestha",
    author_email="dhruvkul2004@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)