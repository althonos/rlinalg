python_sources = [
    '__init__.py',
]

python.install_sources(python_sources, subdir : 'rlinalg/linpack')

fortan_sources = [
     '../../vendor/r-source/src/appl/dqrdc2.f',
     '../../vendor/r-source/src/appl/dqrutl.f',
     '../../vendor/r-source/src/appl/dtrco.f',
     '../../vendor/r-source/src/appl/dqrsl.f',
]

linpackmodule = custom_target('_linpackmodule.c',
  input: ['_linpack.pyf', fortan_sources],
  output: ['_linpackmodule.c', '_linpack-f2pywrappers.f'],
  command: [python, '-m', 'numpy.f2py', '-c', '@INPUT@', '--build-dir', '@OUTDIR@' ],
)

python.extension_module('_linpack',
  linpackmodule, fortan_sources, incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : [py_dep, blas_dep],
  install: true,
  subdir : 'rlinalg/linpack',
)
