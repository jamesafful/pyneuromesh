# PyNeuroMesh

**PyNeuroMesh** is a hybrid meshing pipeline that integrates neural implicit surface representations with classical mesh generation. It uses neural networks to learn signed distance functions (SDFs) and adaptive refinement fields to guide high-quality mesh generation using GMSH.

---

## 🔑 Features

- **Neural SDF Models** — Learn continuous geometry via MLPs.
- **GMSH Integration** — Surface extraction via marching cubes, meshing with GMSH.
- **Adaptive Refinement** — Use `RefineNet` to predict local mesh size.
- **Training Pipelines** — Includes synthetic and simulation-aware training scripts.
- **Python + CLI Access** — Scriptable modules and runnable command-line tools.

---

## 📦 Installation

```bash
git clone https://github.com/jamesafful/pyneuromesh.git
cd pyneuromesh
pip install -r requirements.txt
