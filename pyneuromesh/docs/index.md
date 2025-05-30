# PyNeuroMesh

A differentiable Python framework for neural mesh generation and adaptation — combining implicit neural representations, simulation-driven refinement, and open-source meshing tools like GMSH.

## 🚀 Why PyNeuroMesh?

- 🧠 Neural SDFs to represent complex geometry.
- 🔁 Physics-informed mesh refinement using simulation feedback.
- 🧰 Export-ready meshes for OpenFOAM, SU2, and FEM solvers.
- 📦 Modular, open-source, and research-ready.

## 🧬 Core Components

- `GeometryNet`: Learnable implicit shape models (SDF-based).
- `MeshGen`: Extract and export meshes using marching cubes + GMSH.
- `RefineNet`: Learns how to refine mesh adaptively based on residuals.
- `ProxySimulation`: Mock solver interface — pluggable with real solvers.

## 📖 Documentation Sections

- [Installation](install.md)
- [CLI & Usage](usage.md)
- [Examples](examples.md)
- [Architecture](architecture.md)

## 🔗 GitHub

[View the source on GitHub →](https://github.com/jamesafful/pyneuromesh)
