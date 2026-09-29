"""Unit tests for thermal.doctor."""

from thermal.doctor import CheckStatus, core_ready, run_doctor


def test_run_doctor_returns_checks_with_valid_status():
    checks = run_doctor()
    assert len(checks) > 0
    for check in checks:
        assert isinstance(check.status, CheckStatus)
        assert check.name


def test_core_ready_false_when_required_check_fails():
    checks = run_doctor()
    checks[0].required = True
    checks[0].status = CheckStatus.FAIL
    assert core_ready(checks) is False


def test_core_ready_true_when_only_optional_checks_fail():
    checks = run_doctor()
    for check in checks:
        if not check.required:
            check.status = CheckStatus.FAIL
        else:
            check.status = CheckStatus.OK
    assert core_ready(checks) is True
