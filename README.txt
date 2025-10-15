# This folder holds the up-to-date version of the MolAnchors method.

# The aim of the methodology is to identify molecular fragments responisble for ML predictions.

# The model can be simply installed by going into the directory of "setup.py" and calling:
pip install -e .
# Alternatively the folder itself can be copied / cloned directly 

# Exemplary usage:
from MolAnchor.MolAnchor import *

mol = Chem.MolFromSmiles(smile)

## When using fingerprint
frag_gen = MolAnchors(mol, model, target_class = 1, fragment_scheme="BRICS", representation="ECFP", original_fp=fp, bit_inf=bitinf)
frag_combs = frag_gen.predict_frag_combinations()
anchor_df_cpd = frag_gen.identify_anchors(frag_combs, cutoff=0.95, allow_frag_combinations=True)

## When using graphs
### Two functions have to be defined:
1. How a given Molecule is represented by a graph (only implemented for networkx):

Default:

def default_mol_to_nx(mol):
    """
    Default function to convert a molecule to a NetworkX graph.

    Parameters:
        mol (rdkit.Chem.Mol): The input molecule.

    Returns:
        graph (networkx.Graph): The resulting NetworkX graph.
    """

    symbols = [
        "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'
    ]

    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]

    G = nx.Graph()

    for atom in mol.GetAtoms():
        symbol = [0.] * len(symbols)
        symbol[symbols.index(atom.GetSymbol())] = 1.

        hybridization_atom = [0.] * len(hybridizations)
        hybridization_atom[hybridizations.index(
            atom.GetHybridization())] = 1

        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=symbol,
                   atomic_weight=sigmoid(Chem.GetPeriodicTable().GetAtomicWeight(atom.GetSymbol())),
                   n_valence=float(atom.GetTotalValence()),
                   n_hydrogens=float(atom.GetTotalNumHs()),
                   hybridization=hybridization_atom
                   )

    for bond in mol.GetBonds():

        bond_type_atom =bond.GetBondType()
        single = 1. if bond_type_atom == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type_atom == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type_atom == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type_atom == Chem.rdchem.BondType.AROMATIC else 0.

        conjugation = [0.] * 2
        if bond.GetIsConjugated():
            conjugation[0] = 1.
        else:
            conjugation[1] = 1.

        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   single=single,
                   double=double,
                   triple=triple,
                   aromatic=aromatic,
                   bond_conjugation=conjugation
                   )
    return G



2. How the data is loaded into the model and predicted:

Default:

def default_graph_predict(model, frag_graphs):
    """
    Predicts the output based on a list of graph fragments using the provided model.

    This function converts the given list of NetworkX graph fragments (`frag_graphs`)
    into a format suitable for the model.

    Args:
        model: The model that will be used to make predictions.
        frag_graphs (list of networkx.Graph): list of molecular graphs
    Returns:
        numpy.ndarray: A 1D array of integer predictions, rounded from the model's output.
    """
    # Convert graphs into model input
    frag_x = [from_networkx(graph,
                            group_node_attrs=["atomic_num", "is_aromatic", "atomic_weight",
                                              "n_valence", "n_hydrogens", "atom_symbol", "hybridization"],
                            group_edge_attrs=["single", "double", "triple", "aromatic", "bond_conjugation"])
              for graph in frag_graphs]

    anchor_data = GraphDataset(frag_x)

    predictions = model.predict(anchor_data).numpy().round().astype(int).flatten()

    return predictions


frag_gen = MolAnchors(mol, graph_model, target_class=1, fragment_scheme="BRICS", representation="graphs",  
graph_func=default_mol_to_nx,
graph_predict=default_graph_predict)

frag_combs = frag_gen.predict_frag_combinations()
anchor_df_cpd = frag_gen.identify_anchors(frag_combs, cutoff=0.95, allow_frag_combinations=True)


