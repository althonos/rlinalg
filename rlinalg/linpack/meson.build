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

pyf_sources = [
  '_dqrdc2.pyf',
  '_dqrsl.pyf',
  '_dqrutl.pyf',
  '_dqrtco.pyf'
]

linpackmodule = custom_target('_linpackmodule.c',
  input: '_linpack.pyf',
  output: '_linpackmodule.c',
  command: [python, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@' ],
  depend_files : ['_dqrdc2.pyf']
)

python.extension_module('_linpack',
  linpackmodule, fortan_sources, incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : [py_dep, blas_dep],
  install: true,
  subdir : 'rlinalg/linpack',
)
