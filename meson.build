project(
    'rlinalg', 
    ['c', 'fortran'],
    default_options : {
      'fortran_std' : 'legacy',
    }
)

fortran = meson.get_compiler('fortran')
if fortran.get_id() == 'gcc' 
  if get_option('buildtype') == 'debug'
    add_global_arguments(
      '-fcheck=all',
      language: 'fortran'
    )
  else
    add_global_arguments(
      '-fcheck=no-all',
      language: 'fortran'
    )
  endif
endif

# load Python dependencies
py_mod = import('python')
python = py_mod.find_installation(pure: false)
py_dep = python.dependency()

# find include folders for NumPy
incdir_numpy = run_command(python,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(python,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()
inc_np = include_directories(incdir_numpy, incdir_f2py)

# find BLAS dependncy
if get_option('blas') == 'blas'
  blas_dep = declare_dependency(link_args : ['-lblas'])
else
  blas_dep = dependency(get_option('blas'))
endif

# build package
subdir('rlinalg')
