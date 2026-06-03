"""ChainManager — registry for creating, storing and running chains."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.chains.chain import Chain
from nexus_llm.chains.sequential import SequentialChain
from nexus_llm.chains.parallel import ParallelChain
from nexus_llm.chains.conditional import ConditionalChain

logger = logging.getLogger(__name__)


class ChainNotFoundError(KeyError):
    """Raised when a requested chain does not exist in the registry."""


class ChainManager:
    """Central registry and factory for :class:`Chain` instances.

    Provides convenience methods to create, look up, list, and execute chains
    by name.

    Example
    -------
    >>> mgr = ChainManager()
    >>> chain = mgr.create_chain("echo", steps=[lambda x: x])
    >>> mgr.run_chain("echo", "hello")
    'hello'
    """

    def __init__(self) -> None:
        self._chains: Dict[str, Chain] = {}

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def create_chain(
        self,
        name: str,
        steps: Optional[List[Callable]] = None,
        *,
        chain_type: str = "sequential",
        max_retries: int = 0,
        retry_delay: float = 0.0,
        max_workers: Optional[int] = None,
        default_chain: Optional[Chain] = None,
    ) -> Chain:
        """Create a new chain and register it.

        Parameters
        ----------
        name:
            Unique identifier for the chain.
        steps:
            Initial list of callables.
        chain_type:
            One of ``"sequential"``, ``"parallel"``, or ``"conditional"``.
        max_retries:
            Only applies to ``"sequential"`` chains.
        retry_delay:
            Only applies to ``"sequential"`` chains.
        max_workers:
            Only applies to ``"parallel"`` chains.
        default_chain:
            Only applies to ``"conditional"`` chains.

        Returns
        -------
        The newly created :class:`Chain`.

        Raises
        ------
        ValueError
            If *name* is already registered or *chain_type* is unknown.
        """
        if name in self._chains:
            raise ValueError(f"Chain {name!r} already exists")

        chain: Chain
        if chain_type == "sequential":
            chain = SequentialChain(
                name=name,
                steps=steps,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        elif chain_type == "parallel":
            chain = ParallelChain(
                name=name,
                steps=steps,
                max_workers=max_workers,
            )
        elif chain_type == "conditional":
            chain = ConditionalChain(
                name=name,
                default_chain=default_chain,
            )
        else:
            raise ValueError(f"Unknown chain_type {chain_type!r}")

        self._chains[name] = chain
        logger.info("Created %s chain %r", chain_type, name)
        return chain

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_chain(self, name: str) -> Chain:
        """Return the chain registered under *name*.

        Raises
        ------
        ChainNotFoundError
            If *name* is not registered.
        """
        if name not in self._chains:
            raise ChainNotFoundError(name)
        return self._chains[name]

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_chains(self) -> List[str]:
        """Return a sorted list of registered chain names."""
        return sorted(self._chains.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_chain(self, name: str, input_data: Any = None) -> Any:
        """Look up a chain by *name* and execute it.

        Parameters
        ----------
        name:
            Registered chain name.
        input_data:
            Data to pass to :meth:`Chain.run`.

        Raises
        ------
        ChainNotFoundError
            If *name* is not registered.
        """
        chain = self.get_chain(name)
        logger.info("Running chain %r", name)
        return chain.run(input_data)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def remove_chain(self, name: str) -> None:
        """Unregister a chain."""
        if name not in self._chains:
            raise ChainNotFoundError(name)
        del self._chains[name]
        logger.info("Removed chain %r", name)

    def clear(self) -> None:
        """Remove all registered chains."""
        self._chains.clear()
