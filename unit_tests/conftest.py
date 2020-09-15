import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--cases", action="store", default="15", help="The number of random test cases"
    )

@pytest.fixture
def cases(request):
    return request.config.getoption("--cases")
