from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="clinicalplan",
    version="0.1.4",
    description=(
        "ClinicalPLAN is a Python package for predicting postoperative risks from clinical notes using "
        "language models. It provides training and inference workflows for fine-tuned models, "
        "semi-supervised methods, and multi-task prediction of multiple clinical outcomes. "
        "The package is intended for clinical research and educational use, notably for the American College of Surgeons."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Charles Alba",
    author_email="alba@wustl.edu",
    url="https://github.com/cja5553/ACS_demo_postoperative_risk_prediction_with_clinical_notes",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    python_requires=">=3.9,<3.13",
    install_requires=[
        "transformers>=4.36,<5",
        "tokenizers>=0.15",
        "huggingface_hub>=0.20",
        "accelerate>=0.25",
        "datasets>=2.14",
        "safetensors>=0.4",

        "torch>=2.0",

        "numpy>=1.23",
        "pandas>=2.0",
        "pyarrow>=14",

        "tqdm>=4.66",
    ],
    extras_require={
        # Optional — not needed for `mtl_finetune` or `get_postoperative_outcome_scores`,
        # but will be used when we add the K-fold CV / downstream classifier module.
        "classifiers": [
            "scikit-learn>=1.3",
            "xgboost>=1.7",
        ],
        # Developer tools
        "dev": [
            "pytest>=7",
            "jupyter",
            "ipykernel",
            "ipywidgets>=8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="multi-task learning, NLP, clinical, perioperative, BERT, Bio_ClinicalBERT",
    include_package_data=True,
)