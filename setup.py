import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="momped",
    version="0.1.0",
    author="Alex Lin",
    author_email="alex.lin416@outlook.com",
    description="Multi Object Multimodal Pose Estimation and Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexlin2/MOMPED",
    project_urls={
        "Bug Tracker": "https://github.com/alexlin2/MOMPED/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9.0',
            'black>=21.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'momped=momped.cli:main',
        ],
    },
    include_package_data=True,
    keywords='computer vision, 3D reconstruction, feature matching, SIFT',
)