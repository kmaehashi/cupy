#!/usr/bin/env python

import datetime
import subprocess
import sys


def _log(*msg):
    sys.stderr.write('[{}] {}\n'.format(datetime.datetime.now(), ' '.join(msg)))


class Environment:

    PKG_LIST_COMMAND = []

    def get_package_versions(
            self, package, version_prefix=None, version_suffix=None):
        cmd = self.PKG_LIST_COMMAND + [package]

        _log('Looking for ', package, 'with command', str(cmd))
        out = subprocess.check_output(cmd).decode('utf-8')
        versions = [
            line.split()[1] for line in out.splitlines()
            if line.startswith(package)]
        if len(versions) == 0:
            raise RuntimeError('no versions found')

        match_versions = [
            ver for ver in versions
            if ((version_prefix is None or ver.startswith(version_prefix) and
                (version_suffix is None or ver.endswith(version_suffix))))]
        _log('Versions found:', str(versions))
        _log('Versions matched:', str(match_versions))
        if len(match_versions) == 0:
            raise RuntimeError('no matching versions found')
        return match_versions


class Ubuntu(Environment):

    PKG_LIST_COMMAND = ['apt', 'list', '--all-versions']

    def install_command(self, packages, version):
        return (['apt', 'install', '-y'] +
                ['{}={}'.format(p, version) for p in packages])

    def cutensor(self, version):
        major = int(version[0])
        packages = ['libcutensor{}'.format(major), 'libcutensor-dev']
        return self.install_command(
            packages,
            self.get_package_versions(packages[0], version)[0])

    def nccl(self, version, cuda_version):
        major = int(version[0])
        packages = ['libnccl{}'.format(major), 'libnccl-dev']
        suffix = '+cuda{}'.format(cuda_version)
        return self.install_command(
            packages,
            self.get_package_versions(packages[0], version, suffix)[0])

    def cudnn(self, version, cuda_version):
        major = int(version[0])
        packages = ['libcudnn{}'.format(major), 'libcudnn{}-dev'.format(major)]
        suffix = '+cuda{}'.format(cuda_version)
        return self.install_command(
            packages,
            self.get_package_versions(packages[0], version, suffix)[0])


class RHEL(Environment):

    PKG_LIST_COMMAND = ['yum', 'list', '--showduplicates']

    def install_command(self, packages, version):
        return (['mum', 'install', '-y'] +
                ['{}-{}'.format(p, version) for p in packages])

    def cutensor(self, version):
        major = int(version[0])
        packages = ['libcutensor{}'.format(major), 'libcutensor-dev']
        return self.install_packages(
            packages,
            self.get_package_versions(packages[0], version)[0])

    def nccl(self, version, cuda_version):
        major = int(version[0])
        packages = ['libnccl{}'.format(major), 'libnccl-devel']
        suffix = '.cuda{}'.format(cuda_version)
        return self.install_command(
            packages,
            self.get_package_versions(packages[0], version, suffix)[0])

    def cudnn(self, version, cuda_version):
        major = int(version[0])
        packages = ['libcudnn{}'.format(major), 'libcudnn{}-dev'.format(major)]
        suffix = '.cuda{}'.format(cuda_version)
        return self.install_command(
            packages,
            self.get_package_versions(packages[0], version, suffix)[0])


def main():
    print(Ubuntu().cutensor('1.2'))


if __name__ == '__main__':
    main()
