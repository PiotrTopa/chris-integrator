"""Build script: pyproject.toml has the metadata; this file adds the native core,
`chris_integrator.native._chris_native` (one C file, the CPython limited API), which the package needs.

    pip install .                           builds it: a C compiler and the Python headers are needed

`python -m chris_integrator.native` checks an installation. docs/NATIVE.md."""
from setuptools import Extension, setup

ext = Extension(
    "chris_integrator.native._chris_native",
    sources=["chris_integrator/native/src/chris_native.c"],
    py_limited_api=True,                    # one binary for every CPython >= 3.11 (abi3)
    extra_compile_args=["-std=c99", "-O3", "-ffp-contract=off", "-fno-fast-math", "-fvisibility=hidden"],
    libraries=["m"],
)
setup(ext_modules=[ext], options={"bdist_wheel": {"py_limited_api": "cp311"}})
