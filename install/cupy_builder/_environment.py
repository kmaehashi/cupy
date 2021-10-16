import distutils.util
import os.path
import subprocess
from typing import Any, Optional, Sequence

import cupy_builder
import cupy_builder.install_utils as utils
from cupy_builder import install_build
from cupy_builder.install_build import PLATFORM_WIN32, _tempdir


class Environment:
    def __init__(self, ctx: cupy_builder.Context):
        self._use_hip = ctx.use_hip

        self.cuda_path: Optional[str] = None
        self.nvcc_path: Optional[str] = None
        self.rocm_path: Optional[str] = None
        self.hipcc_path: Optional[str] = None
        if not ctx.use_stub:
            if not ctx.use_hip:
                self.cuda_path = get_cuda_path()
                self.nvcc_path = compiler_path = get_nvcc_path()
            else:
                self.rocm_path = get_rocm_path()
                self.hipcc_path = compiler_path = get_hipcc_path()

        self.default_compiler_settings = get_compiler_setting(ctx.use_hip)
        self.default_compiler_options = get_compiler_base_options(
            compiler_path)
        self.compute_capabilities: Optional[Sequence[int]] = None

    def configure(self, compiler: Any, settings: Any) -> None:
        if not self._use_hip:
            self.compute_capabilities = (
                install_build.check_compute_capabilities(compiler, settings))


def get_rocm_path():
    return os.environ.get('ROCM_HOME', '')


def get_cuda_path():
    nvcc_path = utils.search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        utils.print_warning(
            'nvcc path != CUDA_PATH',
            'nvcc path: %s' % cuda_path_default,
            'CUDA_PATH: %s' % cuda_path)

    if os.path.exists(cuda_path):
        _cuda_path = cuda_path
    elif cuda_path_default is not None:
        _cuda_path = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path = '/usr/local/cuda'
    else:
        _cuda_path = None

    return _cuda_path


def get_nvcc_path():
    nvcc = os.environ.get('NVCC', None)
    if nvcc:
        return distutils.util.split_quoted(nvcc)

    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None

    if PLATFORM_WIN32:
        nvcc_bin = 'bin/nvcc.exe'
    else:
        nvcc_bin = 'bin/nvcc'

    nvcc_path = os.path.join(cuda_path, nvcc_bin)
    if os.path.exists(nvcc_path):
        return [nvcc_path]
    else:
        return None


def get_hipcc_path():
    hipcc = os.environ.get('HIPCC', None)
    if hipcc:
        return distutils.util.split_quoted(hipcc)

    rocm_path = get_rocm_path()
    if rocm_path is None:
        return None

    if PLATFORM_WIN32:
        hipcc_bin = 'bin/hipcc.exe'
    else:
        hipcc_bin = 'bin/hipcc'

    hipcc_path = os.path.join(rocm_path, hipcc_bin)
    if os.path.exists(hipcc_path):
        return [hipcc_path]
    else:
        return None


def get_compiler_setting(use_hip):
    cuda_path = None
    rocm_path = None

    if use_hip:
        rocm_path = get_rocm_path()
    else:
        cuda_path = get_cuda_path()

    include_dirs = []
    library_dirs = []
    define_macros = []
    extra_compile_args = []

    if cuda_path:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if PLATFORM_WIN32:
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))

    if rocm_path:
        include_dirs.append(os.path.join(rocm_path, 'include'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'hip'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'rocrand'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'hiprand'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'roctracer'))
        library_dirs.append(os.path.join(rocm_path, 'lib'))

    if use_hip:
        extra_compile_args.append('-std=c++11')

    if PLATFORM_WIN32:
        nvtoolsext_path = os.environ.get('NVTOOLSEXT_PATH', '')
        if os.path.exists(nvtoolsext_path):
            include_dirs.append(os.path.join(nvtoolsext_path, 'include'))
            library_dirs.append(os.path.join(nvtoolsext_path, 'lib', 'x64'))
        else:
            define_macros.append(('CUPY_NO_NVTX', '1'))

    # For CUB, we need the complex and CUB headers. The search precedence for
    # the latter is:
    #   1. built-in CUB (for CUDA 11+ and ROCm)
    #   2. CuPy's CUB bundle
    # Note that starting CuPy v8 we no longer use CUB_PATH

    # for <cupy/complex.cuh>
    cupy_header = os.path.join(
        cupy_builder.get_context().source_root, 'cupy/_core/include')
    _jitify_path = os.path.join(cupy_header, 'cupy/jitify')
    if cuda_path:
        cuda_cub_path = os.path.join(cuda_path, 'include', 'cub')
        if not os.path.exists(cuda_cub_path):
            cuda_cub_path = None
    elif rocm_path:
        cuda_cub_path = os.path.join(rocm_path, 'include', 'hipcub')
        if not os.path.exists(cuda_cub_path):
            cuda_cub_path = None
    else:
        cuda_cub_path = None
    if cuda_cub_path:
        _cub_path = cuda_cub_path
    elif not use_hip:  # CuPy's bundle doesn't work for ROCm
        _cub_path = os.path.join(cupy_header, 'cupy', 'cub')
    else:
        raise Exception('Please install hipCUB and retry')
    include_dirs.insert(0, _cub_path)
    include_dirs.insert(1, cupy_header)

    install_build._cub_path = _cub_path
    install_build._jitify_path = _jitify_path

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'language': 'c++',
        'extra_compile_args': extra_compile_args,
    }


def _match_output_lines(output_lines, regexs):
    # Matches regular expressions `regexs` against `output_lines` and finds the
    # consecutive matching lines from `output_lines`.
    # `None` is returned if no match is found.
    if len(output_lines) < len(regexs):
        return None

    matches = [None] * len(regexs)
    for i in range(len(output_lines) - len(regexs)):
        for j in range(len(regexs)):
            m = re.match(regexs[j], output_lines[i + j])
            if not m:
                break
            matches[j] = m
        else:
            # Match found
            return matches

    # No match
    return None


def get_compiler_base_options(compiler_path):
    """Returns base options for nvcc compiler.

    """
    # Try compiling a dummy code.
    # If the compilation fails, try to parse the output of compilation
    # and try to compose base options according to it.
    # compiler_path is the path to nvcc (CUDA) or hipcc (ROCm/HIP)
    with _tempdir() as temp_dir:
        test_cu_path = os.path.join(temp_dir, 'test.cu')
        test_out_path = os.path.join(temp_dir, 'test.out')
        with open(test_cu_path, 'w') as f:
            f.write('int main() { return 0; }')
        proc = subprocess.Popen(
            compiler_path + ['-o', test_out_path, test_cu_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdoutdata, stderrdata = proc.communicate()
        stderrlines = stderrdata.split(b'\n')
        if proc.returncode != 0:

            # No supported host compiler
            matches = _match_output_lines(
                stderrlines,
                [
                    b'^ERROR: No supported gcc/g\\+\\+ host compiler found, '
                    b'but .* is available.$',
                    b'^ *Use \'nvcc (.*)\' to use that instead.$',
                ])
            if matches is not None:
                base_opts = matches[1].group(1)
                base_opts = base_opts.decode('utf8').split(' ')
                return base_opts

            # Unknown error
            raise RuntimeError(
                'Encountered unknown error while testing nvcc:\n' +
                stderrdata.decode('utf8'))

    return []
