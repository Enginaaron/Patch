import pytest

from tests.rover.helpers import Harness


@pytest.fixture
def harness():
    """Factory for ``Harness`` objects; everything built through it is released
    and shut down afterwards, even when the test fails half-way."""
    built: list[Harness] = []

    def factory(*args, **kwargs) -> Harness:
        instance = Harness(*args, **kwargs)
        built.append(instance)
        return instance

    yield factory
    for instance in built:
        instance.close()
