"""Model-weight setup: check which required weights are present in embpred's MODELS_DIR
and download missing ones from the private S3 bucket via the AWS CLI.

``install_weights.py`` in embpred_deploy only unzips a local archive; the README
documents `aws s3 cp` from ``s3://cfai-model-weights``. This module bridges that gap:
it resolves ``MODELS_DIR`` (through :mod:`embpred_backend`), lists/downloads weights,
and surfaces auth problems as typed exceptions the GUI can explain to the user.

No Qt here — the GUI owns the prompts/progress dialogs and walks the user through AWS
auth when needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, List, Optional

import embpred_backend as eb

BUCKET = "cfai-model-weights"
RCNN_FILENAME = eb.RCNN_FILENAME


class AwsUnavailableError(RuntimeError):
    """The AWS CLI is not installed / not on PATH."""


class AwsAuthError(RuntimeError):
    """AWS credentials are missing or lack access to the bucket."""


# --------------------------------------------------------------------------------------
# AWS CLI helpers.
# --------------------------------------------------------------------------------------
def aws_cli_available() -> bool:
    return shutil.which("aws") is not None


def _require_aws() -> None:
    if not aws_cli_available():
        raise AwsUnavailableError(
            "The AWS CLI ('aws') was not found on PATH. Install it from "
            "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html "
            "then run `aws configure`."
        )


def _classify_aws_error(stderr: str) -> Optional[Exception]:
    lowered = stderr.lower()
    if "unable to locate credentials" in lowered or "no credentials" in lowered:
        return AwsAuthError(
            "No AWS credentials found. Run `aws configure` (or set AWS_PROFILE / env vars) "
            "with an identity that can read the cfai-model-weights bucket."
        )
    if "accessdenied" in lowered or "access denied" in lowered or "forbidden" in lowered:
        return AwsAuthError(
            "Access denied to the cfai-model-weights bucket. Ask the project maintainer "
            "for s3:GetObject permission on that bucket."
        )
    if "expired" in lowered and "token" in lowered:
        return AwsAuthError("AWS credentials/token expired. Re-authenticate (e.g. `aws sso login`).")
    return None


def auth_status() -> "tuple[bool, str]":
    """Return ``(authenticated, human-readable status/guidance)``.

    Single source of truth for the GUI's AWS state: drives the model status chip and
    whether S3-backed actions (e.g. Re-run Predictions) are enabled. Never raises —
    turns the "no CLI" / "no creds" cases into a friendly message instead.
    """
    if not aws_cli_available():
        return False, (
            "AWS CLI not found on PATH. Install it and run `aws configure` to download "
            "models from the cfai-model-weights bucket."
        )
    arn = aws_identity()
    if arn:
        return True, f"Signed in to AWS as {arn}"
    return False, (
        "Not signed in to AWS. Run `aws configure` (or `aws sso login`) with access to "
        "the cfai-model-weights bucket to list/download models."
    )


def is_authenticated() -> bool:
    """True if the AWS CLI exists and reports a caller identity."""
    return auth_status()[0]


def aws_identity() -> Optional[str]:
    """Return the caller's ARN if authenticated, else None (CLI must exist)."""
    _require_aws()
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--query", "Arn", "--output", "text"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    arn = result.stdout.strip()
    return arn or None


# --------------------------------------------------------------------------------------
# Listing / presence.
# --------------------------------------------------------------------------------------
def list_s3_models() -> List[str]:
    """Model names (``*.pth`` stems) available in the S3 bucket.

    Raises AwsUnavailableError / AwsAuthError on CLI / credential problems.
    """
    _require_aws()
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{BUCKET}/"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise AwsAuthError("Timed out listing the S3 bucket.") from exc
    if result.returncode != 0:
        error = _classify_aws_error(result.stderr)
        raise error if error else RuntimeError(f"`aws s3 ls` failed: {result.stderr.strip()}")

    models: List[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        filename = parts[-1]
        if filename.endswith(".pth"):
            models.append(os.path.splitext(filename)[0])
    return sorted(models)


def local_models() -> List[str]:
    return eb.list_local_models()


def missing_files(model_name: str) -> List[str]:
    """Required filenames (``{model}.pth`` + ``rcnn.pt``) absent from MODELS_DIR."""
    return eb.missing_files(model_name)


def is_ready(model_name: str) -> bool:
    return not missing_files(model_name)


# --------------------------------------------------------------------------------------
# Downloading.
# --------------------------------------------------------------------------------------
def _download_key(
    filename: str,
    dest_dir: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    _require_aws()
    os.makedirs(dest_dir, exist_ok=True)
    source = f"s3://{BUCKET}/{filename}"
    dest = os.path.join(dest_dir, filename)
    if progress_cb:
        progress_cb(f"Downloading {filename} …")

    process = subprocess.Popen(
        ["aws", "s3", "cp", source, dest],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: List[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            captured.append(line)
            message = line.strip()
            if progress_cb and message:
                progress_cb(message)
            if should_cancel is not None and should_cancel():
                process.terminate()
                raise RuntimeError(f"Download of {filename} cancelled.")
    finally:
        process.wait()

    if process.returncode != 0:
        stderr = "".join(captured)
        error = _classify_aws_error(stderr)
        if error:
            raise error
        raise RuntimeError(f"Failed to download {filename}: {stderr.strip()}")


def download_model(
    model_name: str,
    include_rcnn: bool = True,
    overwrite: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[str]:
    """Download the classifier checkpoint (and rcnn.pt) into MODELS_DIR.

    Returns the list of filenames downloaded. Skips files already present unless
    ``overwrite`` is set. Raises AwsUnavailableError / AwsAuthError on CLI / auth issues.
    """
    dest_dir = eb.models_dir()
    wanted = [f"{model_name}.pth"]
    if include_rcnn:
        wanted.append(RCNN_FILENAME)

    downloaded: List[str] = []
    for filename in wanted:
        target = os.path.join(dest_dir, filename)
        if os.path.exists(target) and not overwrite:
            if progress_cb:
                progress_cb(f"{filename} already present — skipping.")
            continue
        _download_key(filename, dest_dir, progress_cb=progress_cb, should_cancel=should_cancel)
        downloaded.append(filename)

    if downloaded:
        eb.clear_model_cache()  # force reload of freshly downloaded weights
    return downloaded
