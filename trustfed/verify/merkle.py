"""RFC 6962-style domain-separated SHA-256 Merkle Tree (`Step 06`)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def hash_leaf(data: bytes) -> bytes:
    """Compute domain-separated SHA-256 leaf digest `SHA256(0x00 || data)`."""
    return hashlib.sha256(_LEAF_PREFIX + data).digest()


def hash_node(left: bytes, right: bytes) -> bytes:
    """Compute domain-separated SHA-256 internal node digest `SHA256(0x01 || left || right)`."""
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


@dataclass(frozen=True)
class MerkleInclusionProof:
    """Inclusion proof linking a leaf payload to a Merkle root."""

    leaf_index: int
    leaf_hash: bytes
    siblings: tuple[tuple[bytes, str], ...]
    root_hash: bytes


class MerkleTree:
    """Binary SHA-256 Merkle tree over ordered tenant commitment bytes."""

    def __init__(self, leaf_payloads: Sequence[bytes]) -> None:
        if not leaf_payloads:
            raise ValueError("MerkleTree requires at least one leaf payload.")
        self._num_leaves = len(leaf_payloads)
        leaves = [hash_leaf(p) for p in leaf_payloads]
        self._levels: list[list[bytes]] = [leaves]

        current = leaves
        while len(current) > 1:
            next_level: list[bytes] = []
            for idx in range(0, len(current), 2):
                left = current[idx]
                right = current[idx + 1] if idx + 1 < len(current) else current[idx]
                next_level.append(hash_node(left, right))
            self._levels.append(next_level)
            current = next_level

    @property
    def root(self) -> bytes:
        """32-byte SHA-256 Merkle root."""
        return self._levels[-1][0]

    @property
    def num_leaves(self) -> int:
        return self._num_leaves

    def get_proof(self, leaf_index: int) -> MerkleInclusionProof:
        """Generate an O(log K) Merkle inclusion proof for leaf `leaf_index`."""
        if leaf_index < 0 or leaf_index >= self._num_leaves:
            raise IndexError(f"leaf_index {leaf_index} out of range [0, {self._num_leaves}).")
        idx = int(leaf_index)
        siblings: list[tuple[bytes, str]] = []
        for level in self._levels[:-1]:
            if idx % 2 == 0:
                sib_idx = idx + 1 if idx + 1 < len(level) else idx
                siblings.append((level[sib_idx], "R"))
            else:
                sib_idx = idx - 1
                siblings.append((level[sib_idx], "L"))
            idx //= 2

        return MerkleInclusionProof(
            leaf_index=int(leaf_index),
            leaf_hash=self._levels[0][int(leaf_index)],
            siblings=tuple(siblings),
            root_hash=self.root,
        )


def verify_merkle_proof(
    leaf_payload: bytes,
    proof: MerkleInclusionProof,
    expected_root: bytes,
) -> bool:
    """Verify that `leaf_payload` matches `proof` and reconstructs `expected_root`."""
    digest = hash_leaf(leaf_payload)
    if digest != proof.leaf_hash:
        return False
    for sib_hash, side in proof.siblings:
        if side == "R":
            digest = hash_node(digest, sib_hash)
        elif side == "L":
            digest = hash_node(sib_hash, digest)
        else:
            return False
    return digest == expected_root and proof.root_hash == expected_root
