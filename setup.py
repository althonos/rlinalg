import os
import glob
import configparser
import shutil

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

    def _merge_sources(self, sources, output):
        self.mkpath(self.build_temp)
        with open(output, "w") as dst:
            for source in sources:
                with open(source) as src:
                    shutil.copyfileobj(src, dst)

    def build_extension(self, ext):
        if ext.language != "fortran":
            return setuptools.command.build_ext.build_ext.build_extension(self, ext)

        path = self.get_ext_fullpath(ext.name)
        basename = ext.name.split(".")[-1]
        output_dir = os.path.abspath(os.path.dirname(path))
        self.mkpath(output_dir)

        if len(ext.sources) > 1:
            source = os.path.join(self.build_temp, "{}.f".format(basename))
            self.make_file(ext.sources, source, self._merge_sources, (ext.sources, source))
        else:
            source = ext.sources[0]

        ext_dir = os.path.join(output_dir, basename)
        if os.path.exists(ext_dir) and self.force:
            shutil.rmtree(ext_dir)

        fmodpy.fimport(
            source,
            name=basename,
            output_dir=output_dir,
            blas=True,
            lapack=True,
            build_dir=os.path.join(self.build_temp, "fmodpy"),
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
            sources=["vendor/r-source/src/appl/dqrutl.f", "vendor/r-source/src/appl/dqrsl.f"],
        ),
    ],
)
