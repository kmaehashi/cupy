import os


_cuda_path = None


def get_cuda_path():
    global _cuda_path
    if _cuda_path is None:
        _cuda_path = os.getenv('CUDA_PATH', None)
        if _cuda_path is not None:
            return _cuda_path

        for p in os.getenv('PATH', '').split(os.pathsep):
            for cmd in ('nvcc', 'nvcc.exe'):
                nvcc_path = os.path.join(p, cmd)
                if not os.path.exists(nvcc_path):
                    continue
                nvcc_dir = os.path.dirname(os.path.abspath(nvcc_path))
                _cuda_path = os.path.normpath(os.path.join(nvcc_dir, '..'))
                return _cuda_path

        if os.path.exists('/usr/local/cuda'):
            _cuda_path = '/usr/local/cuda'

    return _cuda_path


def _setup_cuda_path():
    cuda_path = get_cuda_path()
    if cuda_path is None:
        warnings.warn(
            'CUDA path could not be detected, consider setting CUDA_PATH')
        return
    os.add_dll_directory(os.path.join(cuda_path, 'bin'))
