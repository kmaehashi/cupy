# Bootstrap script for FlexCI.

Param([String]$target)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"

$log_file = "%TMPDIR%\log.txt"

echo "Environment Variables:"
cmd.exe /C set

$is_pull_request = IsPullRequestTest
if ($is_pull_request) {
    echo "Testing Pull-Request."
}

echo "Starting: ${target}"
echo "****************************************************************************************************"
# TODO
CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.ps1" "${target}" > "${log_file}" 2>&1
$test_retval = $?
echo "****************************************************************************************************"
echo "Build & Test: Exit with status ${test_retval}"

if ($is_pull_request) {
    # Upload cache when testing a branch, even when test failed.
    echo "Uploading cache..."
    # TODO
    CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${target}" cache_put >> "${log_file}" 2>&1
    echo "Upload: Exit with status $?"

    # Notify.
    if (${test_retval} -ne 0 ) {
        pip3 install -q slack-sdk gitterpy
        .\.pfnci\flexci_notify.py "TEST FAILED"
    }
}

echo "Uploading the log..."
gsutil -m -q cp "${log_file}" "gs://chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/"

echo "****************************************************************************************************"
echo "Full log is available at:"
echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/log.txt"
echo "****************************************************************************************************"

exit ${test_retval}
