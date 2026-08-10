from __future__ import annotations

import socket

import pytest


@pytest.mark.unit
def test_unit_suite_denies_network_sockets() -> None:
    with pytest.raises(RuntimeError, match="tried to use socket"):
        socket.socket()
