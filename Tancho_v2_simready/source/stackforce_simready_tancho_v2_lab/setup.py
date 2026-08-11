from setuptools import find_packages, setup


setup(
    name="stackforce_simready_tancho_v2_lab",
    version="0.1.0",
    description="StackForce SimReady Isaac Lab export for Template-TanchoV2-Direct-v0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["psutil"],
    python_requires=">=3.10",
)
