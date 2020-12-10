# HOW TO CREATE A DIST PACKAGE FOR PyPI
# python3 setup.py sdist bdist_wheel

import setuptools
import os
from glob import glob
from distutils.core import setup, Extension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

petlink32_c_module = Extension('tomolab.ScannerGeometries.PetLink.petlink32_c',
                               sources=[os.path.join('tomolab', 'ScannerGeometries','PetLink', 'petlink32_c.c')])
test_simplewrap_module = Extension('tomolab.Tests.tests_simplewrap.test_simplewrap_c',
                                   sources=[os.path.join('tomolab', 'Tests','tests_simplewrap', 'test_simplewrap_c.c')])
test_matrices_module = Extension('tomolab.Tests.tests_simplewrap.test_matrices_c',
                                 sources=[os.path.join('tomolab', 'Tests','tests_simplewrap', 'test_matrices_c.c')])
mMR_listmode_module = Extension('tomolab.ScannerGeometries.Siemens_Biograph_mMR.listmode_c', sources=[os.path.join('tomolab','ScannerGeometries','Siemens_Biograph_mMR','listmode_c.c')])
mMR_physiological_module = Extension('tomolab.ScannerGeometries.Siemens_Biograph_mMR.physiological_c', sources=[os.path.join('tomolab','ScannerGeometries','Siemens_Biograph_mMR','physiological_c.c')])


setup(
    name="tomolab", # Replace with your own username
    version="0.1.0",
    author="Michele Scipioni",
    author_email="mscipioni@mgh.harvard.edu",
    description="Tomographic Vision - PET, SPECT, CT, MRI reconstruction and processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomographyLab/TomoLab",
	license='MIT',
	keywords=[
        "PET",
        "SPECT",
        "MRI",
        "computer vision",
        "artificial intelligence",
        "emission tomography",
        "transmission tomography",
        "tomographic reconstruction",
        "nuclear magnetic resonance",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.6',
	install_requires=[
		"dicom2nifti",
        "h5py",
        "ipy_table",
		"ipython",
		"ipyvolume",
		"ipywidgets",
        "matplotlib",
        "nibabel",
		"numpy",
        "pydicom",
        "scipy",
		"scikit-image",
        "svgwrite",
      ],
    packages=setuptools.find_packages(),
    data_files=[
        (os.path.join('tomolab','Visualization', 'DisplayNode', 'static'),
         glob(os.path.join('tomolab','Visualization','DisplayNode', 'static', '*.*'))),
        (os.path.join('tomolab', 'Visualization','DisplayNode', 'static', 'openseadragon'),
         glob(os.path.join('tomolab','Visualization', 'DisplayNode', 'static', 'openseadragon', '*.*'))),
        (os.path.join('tomolab','Visualization', 'DisplayNode', 'static', 'openseadragon', 'images'),
         glob(os.path.join('tomolab', 'Visualization','DisplayNode', 'static', 'openseadragon', 'images', '*.*'))),
        (os.path.join('tomolab', 'Visualization','DisplayNode', 'static', 'tipix'),
         glob(os.path.join('tomolab','Visualization', 'DisplayNode', 'static', 'tipix', '*.*'))),
        (os.path.join('tomolab','Visualization', 'DisplayNode', 'static', 'tipix', 'js'),
         glob(os.path.join('tomolab','Visualization', 'DisplayNode', 'static', 'tipix', 'js', '*.*'))),
        (os.path.join('tomolab','Visualization', 'DisplayNode', 'static', 'tipix', 'style'),
         glob(os.path.join('tomolab', 'Visualization','DisplayNode', 'static', 'tipix', 'style', '*.*'))),
        (os.path.join('tomolab', 'Visualization','DisplayNode', 'static', 'tipix', 'images'),
         glob(os.path.join('tomolab', 'Visualization','DisplayNode', 'static', 'tipix', 'images', '*.*')))
    ],
    package_data={
        "tomolab": [
            "Data/*.pdf",
            "Data/*.png",
            "Data/*.jpg",
            "Data/*.svg",
            "Data/*.nii",
            "Data/*.dcm",
            "Data/*.h5",
            "Data/*.txt",
            "Data/*.dat",
        ]
    },
    ext_modules=[petlink32_c_module, test_simplewrap_module, test_matrices_module, mMR_listmode_module, mMR_physiological_module],
    
)
