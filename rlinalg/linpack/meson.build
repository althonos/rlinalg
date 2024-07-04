python_sources = [
    '__init__.py',
]

python.install_sources(python_sources, subdir: 'rlinalg.linpack')

fortan_sources = [
     '../../vendor/r-source/src/appl/dqrdc2.f',
     '../../vendor/r-source/src/appl/dqrutl.f',
     '../../vendor/r-source/src/appl/dtrco.f',
     '../../vendor/r-source/src/appl/dqrsl.f',
]

custom_target('_linpackmodule',
  output: ['_linpackmodule.c', '_linpack-f2pywrappers.f'],
  input: ['_linpack.pyf'] + fortan_sources,
  command: [python, '-m', 'numpy.f2py', '-m', '_linpack', '@INPUT@' ],
)

python.extension_module('_linpack',
  fortan_sources + ['_linpackmodule.c'],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : [py_dep, blas_dep],
  install : true
)
