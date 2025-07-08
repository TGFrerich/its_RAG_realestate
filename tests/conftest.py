import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="function")
def tmp_path_factory(request):
    """
    Create a temporary directory for a test function.
    """
    def _tmp_path_factory(name="test_dir"):
        # Create a temporary directory using the test name
        test_name = request.node.name
        temp_dir = Path(tempfile.gettempdir()) / f"{test_name}_{name}"
        
        # Ensure the directory is clean before the test
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        return temp_dir

    yield _tmp_path_factory

    # Teardown: Clean up the created temporary directories
    test_name = request.node.name
    for item in Path(tempfile.gettempdir()).glob(f"{test_name}_*"):
        if item.is_dir():
            shutil.rmtree(item)
