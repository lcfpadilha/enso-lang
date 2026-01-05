"""
Unit tests for the Ensō bundler.
"""
import os
import tempfile
import pytest
from core.bundler import bundle


class TestBundle:
    """Tests for the bundle() function."""
    
    def test_simple_file_no_imports(self):
        """A file with no imports should be returned as-is."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.enso', delete=False) as f:
            f.write('struct Foo { name: String }')
            f.flush()
            
            result = bundle(f.name)
            assert 'struct Foo { name: String }' in result
        
        os.unlink(f.name)
    
    def test_single_import(self):
        """A file with one import should inline the imported content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the imported file
            lib_path = os.path.join(tmpdir, 'lib.enso')
            with open(lib_path, 'w') as f:
                f.write('struct Helper { value: Int }')
            
            # Create the main file
            main_path = os.path.join(tmpdir, 'main.enso')
            with open(main_path, 'w') as f:
                f.write('import "lib.enso";\nstruct Main { h: Helper }')
            
            result = bundle(main_path)
            
            # Both contents should be present
            assert 'struct Helper { value: Int }' in result
            assert 'struct Main { h: Helper }' in result
            # Import statement should be replaced
            assert 'import "lib.enso"' not in result
    
    def test_nested_imports(self):
        """Imports within imported files should also be resolved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base.enso
            base_path = os.path.join(tmpdir, 'base.enso')
            with open(base_path, 'w') as f:
                f.write('struct Base { id: Int }')
            
            # Create lib.enso that imports base.enso
            lib_path = os.path.join(tmpdir, 'lib.enso')
            with open(lib_path, 'w') as f:
                f.write('import "base.enso";\nstruct Lib { b: Base }')
            
            # Create main.enso that imports lib.enso
            main_path = os.path.join(tmpdir, 'main.enso')
            with open(main_path, 'w') as f:
                f.write('import "lib.enso";\nstruct Main { l: Lib }')
            
            result = bundle(main_path)
            
            # All three structs should be present
            assert 'struct Base { id: Int }' in result
            assert 'struct Lib { b: Base }' in result
            assert 'struct Main { l: Lib }' in result
    
    def test_cycle_detection(self):
        """Circular imports should be detected and skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a.enso that imports b.enso
            a_path = os.path.join(tmpdir, 'a.enso')
            with open(a_path, 'w') as f:
                f.write('import "b.enso";\nstruct A { }')
            
            # Create b.enso that imports a.enso (cycle!)
            b_path = os.path.join(tmpdir, 'b.enso')
            with open(b_path, 'w') as f:
                f.write('import "a.enso";\nstruct B { }')
            
            result = bundle(a_path)
            
            # Should contain cycle detection comment
            assert 'Cycle detected' in result
            # Both structs should still be present
            assert 'struct A { }' in result
            assert 'struct B { }' in result
    
    def test_missing_import_raises_error(self):
        """Importing a non-existent file should raise FileNotFoundError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.enso', delete=False) as f:
            f.write('import "nonexistent.enso";')
            f.flush()
            
            with pytest.raises(FileNotFoundError):
                bundle(f.name)
        
        os.unlink(f.name)
    
    def test_subdirectory_imports(self):
        """Imports from subdirectories should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            subdir = os.path.join(tmpdir, 'lib')
            os.makedirs(subdir)
            
            # Create file in subdirectory
            helper_path = os.path.join(subdir, 'helper.enso')
            with open(helper_path, 'w') as f:
                f.write('struct Helper { }')
            
            # Create main file that imports from subdirectory
            main_path = os.path.join(tmpdir, 'main.enso')
            with open(main_path, 'w') as f:
                f.write('import "lib/helper.enso";\nstruct Main { }')
            
            result = bundle(main_path)
            
            assert 'struct Helper { }' in result
            assert 'struct Main { }' in result
