# One latent that knows geometry, physics *and* manufacturability

I built a system where you hand over a box and a load, and get back a shelf
bracket that fits, holds, and can actually be machined — designed, stress-
checked and solver-certified on local hardware.

**Raw data.** L-brackets as unit-gradient distance fields — no meshes anywhere
— labelled by a from-scratch immersed FEA solver across 10 000 load cases, with
forces calibrated so peak stress spans 0.1–1.5 × yield: the regime where design
decisions actually matter.

![raw data](docs/dataset_record.png)

**Training.** Stage A teaches one 416-token latent to reconstruct the whole
distance field. Stage B then reads stress, deflection, machinability and wall
thickness out of *that same latent*. Attention is SE(3)-equivariant by
construction — rotate the part and the load, and the physics rotates with them
to 4.5 × 10⁻⁷.

![training](docs/training_curves.png)

**Architecture.** Learning where rules are weak (judgment, fast evaluation),
exact CSG where rules are perfect (geometry), simulation where trust matters.

![architecture](docs/system_diagram.png)

**Result.** Drag the load in 3D and a bracket appears in ~0.5 s — sized by beam
theory, ranked by the surrogate, certified by FEA, exported as STL. Change the
load *direction* and the strategy changes: centred gusset for a downward load,
twin offset ribs for an off-axis one.

![result](docs/direction_aware.png)

Encoding geometry **and** physics **and** manufacturability in one shared
latent — with typed condition tokens, so a new domain is a head, not a rewrite
— is the unusual part. Code, weights and the honest failure log (including the
verification bug that made every thin design "pass"): AGPL-3.0.
