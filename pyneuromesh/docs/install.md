# 📦 Installation

## 🔧 Requirements

- Python 3.9+
- pip
- Git
- A working C++ compiler (for GMSH)
- (Optional) CUDA for GPU acceleration

## 🛠️ Install via GitHub (Developer Mode)

```bash
git clone https://github.com/jamesafful/pyneuromesh.git
cd pyneuromesh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
