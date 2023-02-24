def test_import():
    try:
        import mhw_library
    except ImportError:
        no_requests = True
