add_languages('fortran')

_dqrdc2_module = custom_target('dqrdc2module',
  output: 'dqrdc2module.c',
  input: 'dqrdc2.pyf',
  command: [py, '-m', 'f2py', '-m', 'dqrdc2', '../../vendor/r-source/src/appl/dqrdc.f', '@INPUT@']
)

py.extension_module('dqrdc2',
  [
    '../../vendor/r-source/src/appl/dqrdc2.f',  # note: this assumes f2py was manually run before!
    'dqrdc2module.c'
  ],
  # incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : [py_dep, lapack_dep],
  install : true
)
