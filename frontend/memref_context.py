from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from xdsl.dialects.memref import Alloc


@dataclass
class MemrefContext:
    """
    Context that relates identifiers from the AST to memref locations
    used in the flat representation.

    TODO: citation to original code:
    https://github.com/xdsl/training-intro/practical/src/tiny_py_to_standard.py
    """
    dictionary: Dict[str, Alloc] = field(default_factory=dict)
    parent_scope: Optional[MemrefContext] = None

    def __getitem__(self, identifier: str) -> Optional[Alloc]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        mem_location = self.dictionary.get(identifier, None)
        if mem_location:
            return mem_location
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __setitem__(self, identifier: str, alloc: Alloc):
        """Relate the given identifier and SSA value in the current scope"""
        self.dictionary[identifier] = alloc

    def copy(self):
        ctx = MemrefContext()
        ctx.dictionary = dict(self.dictionary)
        return ctx

