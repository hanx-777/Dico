import torch

from dico_rank.atom_svd import extract_svd_atoms_from_response_matrix


def test_exact_svd_atoms_match_reference_up_to_sign():
    left = torch.linalg.qr(torch.randn(5, 3, generator=torch.Generator().manual_seed(1))).Q
    right = torch.linalg.qr(torch.randn(4, 3, generator=torch.Generator().manual_seed(2))).Q
    singular = torch.tensor([5.0, 2.0, 0.5])
    response = left @ torch.diag(singular) @ right.T

    atoms = extract_svd_atoms_from_response_matrix(response, top_k=2, module_name="m", cost=9)

    assert [atom.atom_index for atom in atoms] == [0, 1]
    assert torch.allclose(torch.tensor([atom.singular_value for atom in atoms]), singular[:2], atol=1e-5)
    for idx, atom in enumerate(atoms):
        recovered = torch.outer(atom.u, atom.v)
        reference = torch.outer(left[:, idx], right[:, idx])
        # SVD vectors are sign-ambiguous; the rank-one atom is equivalent up to a global sign.
        assert torch.max(torch.abs(torch.abs(recovered) - torch.abs(reference))) < 1e-5


def test_spectral_ratios_are_normalized_over_returned_atoms():
    response = torch.diag(torch.tensor([3.0, 1.0, 0.5]))

    atoms = extract_svd_atoms_from_response_matrix(response, top_k=2, module_name="m", cost=6)

    assert abs(sum(atom.spectral_ratio for atom in atoms) - 1.0) < 1e-6
    assert atoms[0].spectral_ratio > atoms[1].spectral_ratio
