# Honours Research Project

# Project Setup

This project uses **Conda** for environment management and includes a `Makefile` to simplify setup and maintenance.

---

## Requirements

The following dependencies are required and will be installed automatically in the Conda environment:

- numpy==1.26.4  
- pgmpy==1.0.0  
- matplotlib==3.10.5  
- jupyter==1.1.1  
- tensorflow==2.20.0  
- pygraphviz (installed via conda-forge)

---

## Setup Instructions

### 1. Setup Environment & Install Dependencies
To create a new Conda environment with Python 3.12:
```bash

### 1. Creates conda environment
make venv

### 2. Activate Conda Environment
conda activate venv

### 3. Install Dependencies (patience required)
make install
