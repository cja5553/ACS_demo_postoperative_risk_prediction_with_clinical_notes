from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="clinicalplan",                      # change if you already have a name in mind
    version="0.1.0",
    description=(
        "ClinicalPLAN is a Python package for predicting postoperative risks from clinical notes using "
        "language models. It provides training and inference workflows for fine-tuned models, "
        "semi-supervised methods, and multi-task prediction of multiple clinical outcomes. "
        "The package is intended for clinical research and educational use, notably for the American College of Surgeons."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Charles Alba",                          # update with your preferred author name / email
    author_email="",
    url="https://github.com/cja5553/ACS_demo_postoperative_prediction_w_clinical_notes",  # update if needed
    license="Apache-2.0",                           # matches the forked HF code's license
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    python_requires=">=3.9,<3.12",                  # tested on 3.11; 3.9/3.10 should also work
    install_requires=[
        # Core HuggingFace stack — these must stay pinned because the vendored
        # `transformers` fork was derived from 4.30.2 and patches lines that
        # moved/changed in later releases.
        "transformers==4.30.2",
        "tokenizers==0.13.3",
        "huggingface_hub==0.16.4",
        "accelerate==0.20.3",
        "datasets==2.14.6",
        "safetensors==0.3.3",

        # PyTorch — not pinned to the +cu118 local wheel because PyPI doesn't
        # serve CUDA wheels directly. Users install torch themselves from
        # pytorch.org; we specify a compatible version range. See README.
        "torch>=2.0,<2.3",

        # Numeric / data stack that datasets==2.14.6 is compatible with.
        # numpy must be <2 because pyarrow 14 and torch 2.1 were built against 1.x.
        "numpy>=1.23,<2",
        "pandas>=2.0,<2.2",
        "pyarrow>=14,<15",

        # User-facing utilities
        "tqdm>=4.66",
    ],
    extras_require={
        # Optional — not needed for `mtl_finetune` or `get_postoperative_outcome_scores`,
        # but will be used when we add the K-fold CV / downstream classifier module.
        "classifiers": [
            "scikit-learn==1.3.2",
            "xgboost==1.7.6",
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
        "Development Status :: 1 - Planning",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="multi-task learning, NLP, clinical, perioperative, BERT, Bio_ClinicalBERT",
    include_package_data=True,
)