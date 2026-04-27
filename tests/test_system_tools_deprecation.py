import importlib
import warnings


def test_system_tools_emits_deprecation_warning_on_import() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        import src.tools.system_tools as system_tools

        importlib.reload(system_tools)

    assert any(
        warning.category is DeprecationWarning and "deprecated" in str(warning.message).lower()
        for warning in captured
    )

