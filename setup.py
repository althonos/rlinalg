import os
import glob
import configparser
import setuptools
import setuptools.extension
import setuptools.command.build_ext
import distutils.command.clean
import distutils.command.sdist
import distutils.dir_util

import fmodpy


class sdist(distutils.command.sdist.sdist):
    """A `sdist` that generates a `pyproject.toml` on the fly.
    """

    def run(self):
        # build `pyproject.toml` from `setup.cfg`
        c = configparser.ConfigParser()
        c.add_section("build-system")
        c.set("build-system", "requires", str(self.distribution.setup_requires))
        c.set("build-system", 'build-backend', '"setuptools.build_meta"')
        with open("pyproject.toml", "w") as pyproject:
            c.write(pyproject)
        # run the rest of the packaging
        distutils.command.sdist.sdist.run(self)


class build_ext(setuptools.command.build_ext.build_ext):

    def run(self):
        self.build_extensions()

    def build_extension(self, ext):
        if ext.language != "fortran":
            return setuptools.command.build_ext.build_ext.build_extension(self, ext)

        path = self.get_ext_fullpath(ext.name)
        basename = ext.name.split(".")[-1]
        output_dir = os.path.abspath(os.path.dirname(path))
        self.mkpath(output_dir)

        fmodpy.fimport(
            ext.sources[0],
            name=ext.name.split(".")[-1],
            output_dir=output_dir,
            blas=True,
            lapack=True,
            build_dir=self.build_temp,
            rebuild=bool(self.force),
        )


class clean(distutils.command.clean.clean):

    def run(self):
        distutils.command.clean.clean.run(self)
        if self.all:
            for ext in self.distribution.ext_modules:
                path = ext.name.replace(".", os.path.sep)
                if os.path.exists(path):
                    distutils.dir_util.remove_tree(
                        path, dry_run=self.dry_run, verbose=self.verbose
                    )


setuptools.setup(
    cmdclass={
        "build_ext": build_ext,
        "clean": clean,
        "sdist": sdist,
    },
    ext_modules=[
        setuptools.Extension(
            "rlinalg.linpack._dqrdc2",
            language="fortran",
            sources=["vendor/r-source/src/appl/dqrdc2.f"],
        ),
        setuptools.Extension(
            "rlinalg.linpack._dqrutl",
            language="fortran",
            sources=["vendor/r-source/src/appl/dqrsl.f"],
        ),
    ],
)
