# filename: extract_ligand.py
from Bio.PDB import PDBParser, PDBIO, Select

# --- Configuration ---
INPUT_PDB = "4J0P.pdb"
LIGAND_RESNAME = "1H8"
OUTPUT_LIGAND_PDB = "ligand.pdb"

# --- Script ---
class LigandSelect(Select):
    """A selector to extract only the specified ligand residue."""
    def accept_residue(self, residue):
        # The residue is a HETATM and its name matches our ligand
        if residue.id[0].startswith("H_") and residue.resname == LIGAND_RESNAME:
            return 1
        else:
            return 0

print(f"Reading from '{INPUT_PDB}'...")
parser = PDBParser(QUIET=True)
structure = parser.get_structure("complex", INPUT_PDB)

io = PDBIO()
io.set_structure(structure)
io.save(OUTPUT_LIGAND_PDB, select=LigandSelect())

print(f"✅ Ligand '{LIGAND_RESNAME}' extracted and saved to '{OUTPUT_LIGAND_PDB}'.")

