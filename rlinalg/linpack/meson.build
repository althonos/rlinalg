add_languages('fortran')

fortan_sources = [
     '../../vendor/r-source/src/appl/dqrdc2.f'
]

_dqrdc2_module = custom_target('_linpackmodule',
  output: '_linpackmodule.c',
  input: ['_linpack.pyf'] + fortan_sources,
  command: [py, '-m', 'numpy.f2py', '-m', '_linpack', '@INPUT@' ],
)

py.extension_module('_linpack',
  fortan_sources + ['_linpackmodule.c'],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : [py_dep, lapack_dep],
  install : true
)
