"""TLS options against a mock server with a self-signed certificate."""

from __future__ import annotations

import shutil
import ssl
import subprocess

import pytest

from llm_load_tester.metrics import ErrorCategory
from llm_load_tester.mock_server import MockOpenAIServer, MockStreamConfig

from .conftest import run_benchmark

pytestmark = pytest.mark.skipif(not shutil.which("openssl"), reason="needs openssl")


@pytest.fixture
async def https_server(tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    # Like the ModelArts deployment: self-signed, no subjectAltName.
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-subj", "/O=Default Company Ltd", "-keyout", str(key), "-out", str(cert)],
        check=True, capture_output=True,
    )
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert, key)
    server = MockOpenAIServer(MockStreamConfig(output_tokens=4, ttft_s=0.01, token_interval_s=0.005))
    await server.start(ssl_context=context)
    try:
        yield server, cert
    finally:
        await server.stop()


async def test_self_signed_certificate_is_rejected_by_default(https_server) -> None:
    server, _ = https_server
    result = await run_benchmark(server)
    assert result.request_metrics[0].error == ErrorCategory.CONNECTION_ERROR


async def test_pinned_certificate_is_accepted(https_server) -> None:
    server, cert = https_server
    result = await run_benchmark(server, tls_ca_file=str(cert), tls_check_hostname=False)
    assert result.request_metrics[0].error == ErrorCategory.NONE
    assert result.manifest["served_models"] == ["mock-model"]


async def test_pinned_certificate_rejects_a_different_server(https_server, tmp_path) -> None:
    server, _ = https_server
    other = tmp_path / "other.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-subj", "/O=Someone Else", "-keyout", str(tmp_path / "k2.pem"), "-out", str(other)],
        check=True, capture_output=True,
    )
    result = await run_benchmark(server, tls_ca_file=str(other), tls_check_hostname=False)
    assert result.request_metrics[0].error == ErrorCategory.CONNECTION_ERROR


async def test_verification_off_is_opt_in_and_recorded(https_server) -> None:
    server, _ = https_server
    result = await run_benchmark(server, tls_verify=False)
    assert result.request_metrics[0].error == ErrorCategory.NONE
    assert result.manifest["config"]["tls_verify"] is False
