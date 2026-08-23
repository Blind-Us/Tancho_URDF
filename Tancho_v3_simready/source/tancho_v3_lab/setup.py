from setuptools import find_packages, setup


setup(
    name="tancho_v3_lab",
    version="0.1.0",
    description="SimReady Isaac Lab environments for Tancho v3",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["psutil"],
    python_requires=">=3.10",
)
