# Graph Transformation Learning Prototypes

Early PhD portfolio repository for graph transformation learning experiments, graph machine learning prototypes, and benchmark-driven model exploration.

This repository documents the first stage of my PhD research development: standalone task-level experiments, first attempts at modelling graph transformations, and an unfinished prototype for unifying multiple tasks under a shared benchmark architecture.

The code is included as a record of research progression rather than as a maintained software package, final benchmark suite, leaderboard, or official reproducibility artifact. Some scripts may not run, compile, reproduce old results, or remain compatible with current package versions. Some files require missing checkpoints, generated datasets, external downloads, solver installations, RDKit, PyTorch Geometric, transformer checkpoints, substantial compute, or HPC-specific paths.

Some files are intentionally preserved as early attempts or incomplete prototypes. They may contain mistakes, modelling shortcuts, leakage, duplicated code, or abandoned experimental ideas.

## Repository structure

The repository has two main directories.

```text
individual_task_implementations/
    Standalone experiments for individual graph transformation or
    graph-prediction tasks.

unified_benchmark_prototype/
    Unfinished prototype for running multiple graph transformation tasks
    through a shared benchmark architecture.
```

### `individual_task_implementations/`

This directory contains standalone experiments for individual graph transformation or graph-prediction tasks. Each folder corresponds to one task and typically includes some mixture of dataset generation, model training, benchmark code, and evaluation adapters.

These folders show the earliest direct attempts to formalise each task and test different model families before the work moved toward shared framework design.

### `unified_benchmark_prototype/`

This directory contains an intermediate prototype between the standalone task folders and the later organised research framework. It is included because it shows the transition from isolated task scripts toward reusable benchmark design.

It includes:

- a shared `TaskGenerator` abstraction;
- graph-generation utilities;
- a `BenchmarkManager` for dataset generation, splitting, benchmark exposure, and evaluation;
- prototype task definitions for symmetric closure and transitive closure;
- prototype models such as `SymmetricClosureMLP` and `RecurrentClosure`;
- a combined `pipeline.py` entry point;
- an archived HPC training-output log.

## Running the experiments

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Most folders were designed to be run as standalone scripts from inside the corresponding task directory. Typical entry points are named `model.py`, `benchmark.py`, `benchmark_test.py`, task-specific training scripts, or `pipeline.py` in the unified prototype.

Example pattern:

```bash
cd individual_task_implementations/<task_folder>
python benchmark_test.py
```

Many experiments require additional setup, such as trained checkpoint files, generated datasets, external datasets, solver installations, RDKit, PyTorch Geometric, transformer checkpoints, or substantial compute. Some paths may reflect the original research machine or HPC environment and may need editing before anything can be run.

## Task portfolio and attempted models

The table below lists the task-level experiment folders and the model families attempted in each one.

| Task | Folder | Target problem | Models and approaches attempted |
|---:|---|---|---|
| 1 | `individual_task_implementations/algebraic_connectivity/` | Edge ranking and graph modification for improving algebraic connectivity, using changes in the Fiedler value as the target signal. | GraphSAGE encoder; pairwise candidate-edge scorer; delta-lambda prediction head; greedy edge-selection baseline; spectral candidate-edge comparison against algebraic-connectivity changes. |
| 2 | `individual_task_implementations/edge_betweenness_centrality/` | Predicting edge betweenness centrality or structural edge importance on generated graphs. | Linear edge-feature model; MLP edge regressor; GCN baseline; GraphSAGE baseline; shared generated-graph benchmark for comparing edge-level predictors. |
| 3 | `individual_task_implementations/feedback_edge_set/` | Identifying removable or cycle-related edges for feedback-edge-set-style graph editing. | Supervised Random Forest edge classifier; unsupervised/random-label Random Forest prototype; shared graph-feature utilities; cycle-aware edge-removal evaluation pipelines. |
| 4 | `individual_task_implementations/graph_coloring/` | Learning colour assignments for graph-colouring instances. | GCN colouring model; Potts-style colouring loss; colour-usage regularisation; DIMACS graph-colouring loader; heuristic-inspired same-instance evaluation. |
| 5 | `individual_task_implementations/transitive_closure_completion/` | Recovering missing transitive-closure edges in directed graphs. | Recurrent MLP closure model; pointwise MLP baseline; closure-feature stack using powers of the adjacency matrix; validation-thresholded edge prediction; recurrent closure benchmark wrappers. |
| 6 | `individual_task_implementations/symm_closure/` | Learning the symmetric-closure transformation `A OR A.T`. | Pointwise `SymmetricClosureMLP` using `[A[i,j], A[j,i]]`; 1x1 convolution CNN using `[A, A.T]` channels; CNN ablation without transpose access; MLP ablation without transpose access; GCN baseline; GIN baseline; early GAE-GCN and GAE-GIN feasibility experiments; adjacency-matrix CNN feasibility experiments. |
| 7 | `individual_task_implementations/spanning_tree_labelling/` | Labelling planted spanning-tree edges inside graphs with distractor edges. | EdgeScorer MLP; CNN edge-labeler over adjacency-style tensors; contextual edge-feature MLP; REINFORCE-style contextual policy model; supervised EdgeTransformer; MST-style transformer policy model; actor-critic transformer variant; local parameter-space search and heatmap visualisation prototypes. |
| 8 | `individual_task_implementations/shortcut_elimination/` | Predicting shortcut edges to remove from directed graphs. | Exhaustive 3x3 graph training set; pointwise MLP using `[I, I^2]` edge features; shortcut-elimination transform; generated shortcut benchmark over larger directed graph families. |
| 9 | `individual_task_implementations/shape_completion/` | Completing simple cycle-shaped structures such as triangles, squares, pentagons, and hexagons from incomplete adjacency matrices. | Random Forest feature-importance diagnostic; EdgeMLP; DeepEdgeMLP; handcrafted edge features including degree, adjacency powers, principal eigenvector features, and clustering coefficients; permutation-augmented edge dataset. |
| 10 | `individual_task_implementations/retweet_prediction/` | Directed edge-existence prediction on the SNAP Higgs Twitter retweet network. | Early handcrafted edge-feature MLP; improved GATConv encoder; edge MLP decoder; directed negative sampling; node structural features from the training graph; degree-difference edge features; train/validation/test edge loaders. |
| 11 | `individual_task_implementations/node_core_number/` | Predicting node core numbers on synthetic graph families. | Continuous MLP node regressor; continuous GCN; DeepGraphSAGE; graph Transformer using TransformerConv; generated Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, and tree-style benchmark graphs. |
| 12 | `individual_task_implementations/molecule_bond_typing/` | Classifying molecular bond types from RDKit-derived atom-pair descriptors. | Random Forest bond classifier; RDKit atom and bond descriptor features; electronegativity features; benchmark adapter for single, double, triple, and aromatic bond classes. |
| 13 | `individual_task_implementations/minimal_chordal_graph_estimation/` | Predicting chordal-completion fill edges using NetworkX chordal-completion targets. | Fixed-size adjacency MLP; 2D CNN over adjacency matrices; flattened-adjacency Transformer encoder; autoencoder classifier; chordal-completion benchmark over cycle, grid, tree, Erdos-Renyi, and Barabasi-Albert graphs. |
| 14 | `individual_task_implementations/intra-community_edge_detection/` | Classifying whether a Facebook edge connects nodes from different Louvain communities. | Logistic Regression baseline; Random Forest classifier; MLP classifier; handcrafted edge features including degree, degree difference, common neighbours, Jaccard coefficient, Adamic-Adar score, and triangle membership. |
| 15 | `individual_task_implementations/minimal_chordal_graph_completion/` | Candidate-edge prediction for a true minimum-fill-in-style chordal-completion prototype. | Pyomo/CBC binary optimisation data generator; flat-feature MLP; 1D CNN classifier; TransformerClassifier; AutoencoderClassifier; Random Forest classifier; GraphSAGE edge classifier with node features and edge-pair features; feature-order and scaler adapters for benchmark evaluation. |
| 16 | `individual_task_implementations/graph_denoising/` | Knowledge-graph noise detection on GOLD/ATOMIC/ConceptNet-style triples. | BERT text encoder for entity and relation embeddings; GraphSAGE edge classifier; RGCN edge classifier; CompGCN-style classifier prototype; hard-negative mining; Recall@k ranking; ATOMIC A-05/A-10/A-20 variants; ConceptNet C-20 variant; ATOMIC10X top-percent noisy-triple scoring script. |

## Unified benchmark prototype components

The `unified_benchmark_prototype/` directory works on the same general research theme as the task-level folders: representing graph transformation tasks as task objects, generating benchmark graphs through a manager, and evaluating model predictions through task-specific metrics.

| Component | Role in the prototype |
|---|---|
| `benchmark/task_base.py` | Defines the abstract task interface for label generation and default evaluation. |
| `benchmark/utils/graph_generators.py` | Provides shared synthetic graph generators for the prototype benchmark manager. |
| `benchmark/benchmark_manager.py` | Generates graph datasets, delegates label generation to task objects, creates train/validation/test splits, and supports benchmark-style evaluation. |
| `benchmark/tasks/symmetric_closure.py` | Defines the symmetric-closure task `A OR A.T`. |
| `benchmark/tasks/transitive_closure.py` | Defines k-hop transitive-closure completion and task-specific evaluation metrics. |
| `models/SymmetricClosureMLP.py` | Implements a pointwise MLP baseline for symmetric closure using `[A[i,j], A[j,i]]` features. |
| `models/TransitiveClosureMLP.py` | Implements a recurrent closure MLP that repeatedly applies a shared update over `[A, P, A_rwP, PA_rw]` features. |
| `pipeline.py` | Prototype combined benchmark runner for symmetric closure and transitive closure. |
| `HPC Outputs/RecMLP Training.out` | Archived training log from the recurrent closure MLP experiment. |
