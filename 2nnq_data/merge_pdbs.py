import os
from openmm.app import PDBFile, Modeller

# --- User Configuration ---
# Absolute path to the protein PDB file.
protein_pdb_path = '/home/kuoweihuang/aimnet/2nnq_data/processed_2nnq_with_H.pdb'

# Absolute path to the ligand file. This script now supports .mol2 files
# by using the Open Babel library for on-the-fly conversion.
ligand_path = '/home/kuoweihuang/aimnet/2nnq_data/ligand_1.mol2'

# Desired output path for the merged PDB file.
output_pdb_path = '/home/kuoweihuang/aimnet/2nnq_data/complex_2nnq.pdb'
# --- End of Configuration ---


def merge_structures(protein_path, ligand_path, output_path):
    """
    Merges a protein PDB and a ligand PDB into a single complex PDB file.

    Args:
        protein_path (str): Path to the protein PDB file.
        ligand_path (str): Path to the ligand PDB file.
        output_path (str): Path to save the merged PDB file.
    """
    # --- File Existence Check ---
    if not os.path.exists(protein_path):
        print(f"Error: Protein file not found at {protein_path}")
        return
    if not os.path.exists(ligand_path):
        print(f"Error: Ligand file not found at {ligand_path}")
        print("Please ensure the ligand file exists and the path is correct.")
        return

    # 1. Load the protein structure
    print(f"Loading protein from: {os.path.basename(protein_path)}")
    protein = PDBFile(protein_path)

    # 2. Load the ligand structure, converting from .mol2 if necessary
    print(f"Loading ligand from: {os.path.basename(ligand_path)}")
    temp_pdb_path = None
    if ligand_path.endswith('.mol2'):
        print("-> Ligand is in .mol2 format. Converting to a temporary PDB file...")
        try:
            import openbabel.openbabel as openbabel
        except ImportError:
            print("\nError: The 'openbabel' library is required to read .mol2 files.")
            print("Please install it, for example using: 'conda install -c conda-forge openbabel'")
            return

        temp_pdb_path = ligand_path.replace('.mol2', '_temp.pdb')
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("mol2", "pdb")
        mol = openbabel.OBMol()

        if not obConversion.ReadFile(mol, ligand_path):
            print(f"Error: Open Babel could not read the ligand file: {ligand_path}")
            return

        # Set a standard residue name for all atoms in the ligand to avoid issues
        for atom in openbabel.OBMolAtomIter(mol):
            res = atom.GetResidue()
            if res:
                res.SetName("LIG")

        obConversion.WriteFile(mol, temp_pdb_path)
        ligand = PDBFile(temp_pdb_path)
        print(f"-> Successfully converted to {os.path.basename(temp_pdb_path)}")
    else:
        ligand = PDBFile(ligand_path)

    # 3. Create a Modeller object and add the structures
    modeller = Modeller(protein.topology, protein.positions)
    print("\nAdding ligand to the protein structure...")
    modeller.add(ligand.topology, ligand.positions)

    # 4. Write the merged structure to a new PDB file
    print(f"Saving merged complex to: {os.path.basename(output_path)}")
    with open(output_path, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    # 5. Clean up the temporary file if it was created
    if temp_pdb_path and os.path.exists(temp_pdb_path):
        os.remove(temp_pdb_path)
        print(f"Removed temporary file: {os.path.basename(temp_pdb_path)}")

    print("\nMerge complete!")


if __name__ == '__main__':
    merge_structures(protein_pdb_path, ligand_path, output_pdb_path)