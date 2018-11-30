import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Dawnet",
    version="0.0.2",
    author="_john",
    author_email="trungduc1992@gmail.com",
    description="A deep learning package to inquire intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johntd54/dawnet",
    packages=setuptools.find_packages(
        exclude=["tests.*", "tests", "*.tests", "*.tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    license='GNU General Public License (GPL)',
    keywords="dawnet deep learning inquire artificial intelligence",
    install_requires=[
        "numpy>=1.14.0",
        "Pillow>=5.3.0",
        "matplotlib>=2.1.2",
    ],
    python_requires=">=3.5"
)
