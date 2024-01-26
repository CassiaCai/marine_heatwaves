def test_import():
    try:
        import mhw_measures
    except ImportError:
        no_requests = True
