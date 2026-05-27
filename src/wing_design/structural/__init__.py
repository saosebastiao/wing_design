"""Coarse structural sizing for the wingsail (Phase 3) + volumetric FE (later)."""
from .beam import TubeSparSizing, size_tube_spar

__all__ = ["TubeSparSizing", "size_tube_spar"]
