from rdkit.Chem import Lipinski
from synthelite.context.stock.queries import StockQueryMixin


class CriteriaStock(StockQueryMixin):
    def __contains__(self, mol):
        return Lipinski.HeavyAtomCount(mol.rd_mol) < 5


stock = CriteriaStock()
