"""Simulate an *unauthenticated* AWS user so we can be confident the labeler still
works for someone who just downloaded it and has never configured AWS.

This never touches the real ``~/.aws``: it points the AWS CLI at empty throwaway config
/ credentials files (and an empty weights cache) via environment variables, exercises
the ``setup_models`` / ``embpred_backend`` surface, and asserts each AWS failure raises
the *typed* exception the GUI knows how to explain.

Run (no real AWS access required):
    mamba run -n emb_labeler python scripts/test_unauthenticated.py

Scenarios:
  A. CLI present, no credentials   -> AwsAuthError, is_authenticated() False
  B. CLI missing (stripped PATH)   -> AwsUnavailableError, friendly auth_status()
  C. Local weights present offline -> model usable, NO AWS call attempted
  D. Weights cache relocation      -> EMB_LABELER_MODELS_DIR honored, default under ~/.cache
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile

# Import from the repo root regardless of where this is run from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import embpred_backend as eb  # noqa: E402
import setup_models as sm  # noqa: E402

_FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(label)


@contextlib.contextmanager
def env(**overrides):
    """Temporarily set/unset env vars (value None means unset). Restores on exit."""
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def unauthenticated_aws(models_dir: str):
    """AWS CLI present but with empty creds/config, and an empty weights cache."""
    tmp = tempfile.mkdtemp(prefix="emb_unauth_")
    empty_config = os.path.join(tmp, "config")
    empty_creds = os.path.join(tmp, "credentials")
    open(empty_config, "w").close()
    open(empty_creds, "w").close()
    with env(
        EMB_LABELER_MODELS_DIR=models_dir,
        AWS_CONFIG_FILE=empty_config,
        AWS_SHARED_CREDENTIALS_FILE=empty_creds,
        AWS_ACCESS_KEY_ID=None,
        AWS_SECRET_ACCESS_KEY=None,
        AWS_SESSION_TOKEN=None,
        AWS_PROFILE=None,
        AWS_DEFAULT_REGION="us-east-1",
        # Don't let the CLI hang trying the EC2 instance-metadata endpoint.
        AWS_EC2_METADATA_DISABLED="true",
    ):
        yield


def scenario_a_no_credentials() -> None:
    print("Scenario A — CLI present, no credentials:")
    with tempfile.TemporaryDirectory() as models_dir, unauthenticated_aws(models_dir):
        check("aws CLI detected on PATH", sm.aws_cli_available())
        check("aws_identity() is None", sm.aws_identity() is None)
        authed, msg = sm.auth_status()
        check("auth_status() reports not authenticated", authed is False, msg)
        check("is_authenticated() is False", sm.is_authenticated() is False)

        try:
            sm.list_s3_models()
            check("list_s3_models raises AwsAuthError", False, "no exception raised")
        except sm.AwsAuthError as exc:
            check("list_s3_models raises AwsAuthError", True, str(exc).split(".")[0])
        except Exception as exc:  # noqa: BLE001
            check("list_s3_models raises AwsAuthError", False, f"got {type(exc).__name__}: {exc}")

        try:
            sm.download_model("any-model", include_rcnn=False)
            check("download_model raises AwsAuthError", False, "no exception raised")
        except sm.AwsAuthError:
            check("download_model raises AwsAuthError", True)
        except Exception as exc:  # noqa: BLE001
            check("download_model raises AwsAuthError", False, f"got {type(exc).__name__}")


def scenario_b_no_cli() -> None:
    print("Scenario B — AWS CLI not installed (stripped PATH):")
    with tempfile.TemporaryDirectory() as empty_bin, env(PATH=empty_bin):
        check("aws_cli_available() is False", sm.aws_cli_available() is False)
        authed, msg = sm.auth_status()
        check("auth_status() not authenticated + mentions install", authed is False and "CLI" in msg, msg)
        try:
            sm.list_s3_models()
            check("list_s3_models raises AwsUnavailableError", False, "no exception")
        except sm.AwsUnavailableError:
            check("list_s3_models raises AwsUnavailableError", True)
        except Exception as exc:  # noqa: BLE001
            check("list_s3_models raises AwsUnavailableError", False, f"got {type(exc).__name__}")


def scenario_c_local_offline() -> None:
    print("Scenario C — local weights present, fully offline:")
    with tempfile.TemporaryDirectory() as models_dir, unauthenticated_aws(models_dir):
        model = "Some-Local-ResNet50"
        # Drop placeholder weight files (presence checks are path-only, no torch load).
        open(os.path.join(models_dir, f"{model}.pth"), "w").close()
        open(os.path.join(models_dir, eb.RCNN_FILENAME), "w").close()

        check("models_dir() honors override", os.path.abspath(eb.models_dir()) == os.path.abspath(models_dir))
        check("list_local_models() finds the model", model in sm.local_models())
        check("missing_files() is empty", sm.missing_files(model) == [])
        check("is_ready() is True", sm.is_ready(model) is True)
        # The key property: a ready local model needs no AWS round-trip at all.
        check("rcnn_present() is True", eb.rcnn_present() is True)


def scenario_d_cache_location() -> None:
    print("Scenario D — weights cache relocation:")
    with tempfile.TemporaryDirectory() as models_dir, env(EMB_LABELER_MODELS_DIR=models_dir):
        check("override path used", os.path.abspath(eb.models_dir()) == os.path.abspath(models_dir))
    with env(EMB_LABELER_MODELS_DIR=None):
        default = eb.models_dir()
        expected = os.path.join(os.path.expanduser("~"), ".cache", "emb_labeler", "models")
        check("default under ~/.cache/emb_labeler/models", os.path.abspath(default) == os.path.abspath(expected), default)
        check("default is NOT inside embpred_deploy package", "embpred_deploy" not in default)


def main() -> int:
    scenario_a_no_credentials()
    scenario_b_no_cli()
    scenario_c_local_offline()
    scenario_d_cache_location()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED: {_FAILURES}")
        return 1
    print("All unauthenticated-user checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
