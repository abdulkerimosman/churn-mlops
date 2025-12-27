
try:
    import evidently
    print(f"Evidently version: {evidently.__version__}")
    from evidently.metric_preset import DataDriftPreset
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
