import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Dawnet",
    version="0.0.1",
    author="_john",
    author_email="trungduc1992@gmail.com",
    description="A deep learning package to inquire intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johntd54/dawnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
)
