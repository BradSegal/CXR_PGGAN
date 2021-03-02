import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cxr-gan",
    version="0.0.1",
    author="Bradley Segal",
    author_email="brad@segal.co.za",
    description="Package for applying a progressively growing GAN to x-ray synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TODO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
          'numpy',
      ]
)
