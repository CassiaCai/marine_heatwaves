from setuptools import find_packages, setup

# Dependencies
with open("requirements.txt") as f:
    requirements = f.readlines()
INSTALL_REQUIRES = [t.strip() for t in requirements]

with open("requirements-dev.txt) as f:
     test_requirements = f.readlines()
TEST_REQUIRES = [t.strip() for t in test_requirements]

setup(
    name="",
    version="0.1",
    author="Cassia Cai",
    author_email="fmc2855@uw.edu",
    maintainer="Cassia Cai",
    maintainer_email="fmc2855@uw.edu",
    description="Library to calculate marine heatwave measures",
    url="https://github.com/CassiaCai/marine_heatwaves",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
)
