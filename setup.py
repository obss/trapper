from setuptools import find_packages, setup

VERSION = {}  # type: ignore
with open("trapper/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


extras_require = {
    "dev": [
        "black==22.3.0",
        "flake8==3.9.2",
        "isort==5.9.2",
        "pytest>=6.2.4",
        "importlib-metadata>=1.1.0,<4.3;python_version<'3.8'",
        "pytest-cov>=2.12.1",
        "pylint>=2.11",
        "mypy>=0.9",
    ],
}

setup(
    name="trapper",
    version=VERSION["VERSION"],
    author="OBSS",
    url="https://github.com/obss/trapper",
    description="State-of-the-art NLP through transformer models in a modular design and consistent APIs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "test_fixtures",
            "test_fixtures.*",
            "scripts",
            "scripts.*",
        ]
    ),
    entry_points={"console_scripts": ["trapper=trapper.__main__:run"]},
    python_requires=">=3.7.1",
    install_requires=get_requirements(),
    extras_require=extras_require,
    include_package_data=True,
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="python, nlp, natural-language-processing, deep-learning, transformer, pytorch, transformers, allennlp, pytorch-transformers",
)
