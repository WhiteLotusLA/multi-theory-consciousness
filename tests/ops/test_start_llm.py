"""start_llm.sh: the command each backend gets, from Settings alone.

The script is run with --dry-run, or with a stub ``llama-server`` that is a
dummy the test owns. No model server is started and no real server answers:
the one health probe goes to a stand-in on an ephemeral loopback port.
"""

import shlex
import sys
import time

import pytest

from tests.cli.test_http_client import fake_api

from .conftest import LOOPBACK, Ops, pid_alive

pytestmark = pytest.mark.slow

CLASSIC_HELP = "usage: llama-server [options]\\n  --mlock\\n  --no-mmap\\n"
MODERN_HELP = "usage: llama-server [options]\\n  -lm, --load-mode MODE\\n"


def _llama(ops: Ops, help_text: str = CLASSIC_HELP) -> None:
    model = ops.project / "models" / "model.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"")
    ops.set(
        LLM_PROVIDER="local", LLM_BACKEND="llama_cpp", MODEL_PATH="models/model.gguf"
    )
    ops.stubs.add(
        "llama-server",
        f"""
        case "$1" in --help) printf '{help_text}'; exit 0 ;; esac
        sleep 60
        """,
    )


def _command(result) -> list:
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("would run:")
    )
    return shlex.split(line[len("would run:") :])


def test_empty_backend_exits_2_naming_the_key(ops):
    ops.set(LLM_PROVIDER="local", LLM_BACKEND="")
    for arguments in ((), ("--dry-run",), ("--background",)):
        result = ops.run("start_llm.sh", *arguments)
        assert result.returncode == 2, arguments
        assert "LLM_BACKEND is empty" in result.stderr
        assert "MTC_LLM_BACKEND" in result.stderr
        assert "mlx_lm or llama_cpp" in result.stderr
    assert ops.stubs.calls() == []
    assert ops.run("start_llm.sh", "--wait", "soon").returncode == 2
    assert ops.run("start_llm.sh", "--wait").returncode == 2


def test_mlx_lm_command(ops):
    model = ops.project / "models" / "converted"
    model.mkdir(parents=True)
    ops.set(
        LLM_PROVIDER="local",
        LLM_BACKEND="mlx_lm",
        MODEL_PATH="./models/converted",
        CONTEXT_SIZE=4096,
        GPU_LAYERS=20,
        THREADS=6,
        USE_MLOCK=True,
    )
    result = ops.run("start_llm.sh", "--dry-run")
    assert result.returncode == 0, result.stderr
    # The model, the host and the port, and nothing else: the server has no
    # flag for the context size or the llama.cpp settings.
    assert _command(result) == [
        "python3",
        "-m",
        "mlx_lm.server",
        "--model",
        str(model),
        "--host",
        LOOPBACK,
        "--port",
        str(ops.ports["LLM_PORT"]),
    ]
    assert ops.stubs.calls() == []
    background = ops.run("start_llm.sh", "--dry-run", "--background")
    assert "llm.pid" in background.stdout and "llm.log" in background.stdout
    assert not (ops.project / "data").exists()


def test_llama_cpp_command_maps_settings_flags(ops):
    _llama(ops)
    model = str(ops.project / "models" / "model.gguf")
    port = str(ops.ports["LLM_PORT"])
    base = ["llama-server", "-m", model, "--host", LOOPBACK, "--port", port]

    # The Settings defaults: -ngl left out at -1, -t left out at 0, mmap on
    # and mlock off need no flag.
    default = ops.run("start_llm.sh", "--dry-run")
    assert default.returncode == 0, default.stderr
    assert _command(default) == base + ["-c", "32768", "-b", "512"]

    ops.set(
        CONTEXT_SIZE=8192,
        BATCH_SIZE=256,
        GPU_LAYERS=20,
        THREADS=6,
        USE_MMAP=False,
        USE_MLOCK=True,
    )
    tuned = ops.run("start_llm.sh", "--dry-run")
    assert _command(tuned) == base + [
        "-c",
        "8192",
        "-b",
        "256",
        "-ngl",
        "20",
        "-t",
        "6",
        "--no-mmap",
        "--mlock",
    ]
    ops.set(GPU_LAYERS=0, THREADS=0, USE_MMAP=True, USE_MLOCK=True)
    partial = _command(ops.run("start_llm.sh", "--dry-run"))
    assert partial == base + ["-c", "8192", "-b", "256", "-ngl", "0", "--mlock"]
    ops.set(USE_MMAP=False, USE_MLOCK=False)
    assert _command(ops.run("start_llm.sh", "--dry-run"))[-1] == "--no-mmap"
    # Only the help text was asked of the server program.
    assert {tuple(call) for call in ops.stubs.calls("llama-server")} == {
        ("llama-server", "--help")
    }

    # A build with the single loading-mode option gets that form instead.
    _llama(ops, MODERN_HELP)
    expectations = {
        (True, False): [],
        (True, True): ["--load-mode", "mmap+mlock"],
        (False, True): ["--load-mode", "mlock"],
        (False, False): ["--load-mode", "none"],
    }
    for (mmap, mlock), tail in expectations.items():
        ops.set(USE_MMAP=mmap, USE_MLOCK=mlock, GPU_LAYERS=-1)
        command = _command(ops.run("start_llm.sh", "--dry-run"))
        assert command == base + ["-c", "8192", "-b", "256"] + tail, (mmap, mlock)
        assert "--mlock" not in command and "--no-mmap" not in command


def test_missing_model_path_exits_2(ops):
    ops.set(
        LLM_PROVIDER="local", LLM_BACKEND="llama_cpp", MODEL_PATH="models/absent.gguf"
    )
    for arguments in ((), ("--dry-run",)):
        result = ops.run("start_llm.sh", *arguments)
        assert result.returncode == 2
        assert "MODEL_PATH (models/absent.gguf) is not a file" in result.stderr
    # mlx_lm wants a model directory, and the default path does not exist.
    ops.env.pop("MTC_MODEL_PATH")
    ops.set(LLM_BACKEND="mlx_lm")
    result = ops.run("start_llm.sh")
    assert result.returncode == 2
    assert "is not a directory" in result.stderr
    ops.set(MODEL_PATH="~/models/x")
    assert ops.run("start_llm.sh").returncode == 2
    assert [call for call in ops.stubs.calls() if call[1:] != ["--help"]] == []


def test_already_running_exits_0(ops):
    _llama(ops)
    ops.stubs.remove("curl")  # the probe goes to this test's own port
    with fake_api() as server:
        ops.env["MTC_LLM_PORT"] = str(server.port)
        server.reply("GET", "/health", 200, {"status": "ok"})
        result = ops.run("start_llm.sh", "--background")
        assert result.returncode == 0, result.stderr
        assert "already answers" in result.stdout
        assert [r["path"] for r in server.requests] == ["/health"]
        # Something else on the port, answering no health path: nothing is
        # started on top of it and the run fails.
        server.reply("GET", "/health", 500, {})
        server.reply("GET", "/v1/models", 500, {})
        busy = ops.run("start_llm.sh", "--background")
        assert busy.returncode == 1
        assert "does not answer the health paths" in busy.stderr
    assert not (ops.project / "data" / "run" / "llm.pid").exists()
    assert [
        call for call in ops.stubs.calls("llama-server") if call[1:2] == ["-m"]
    ] == []


def test_background_start_writes_pid_and_log(ops):
    _llama(ops)
    result = ops.run("start_llm.sh", "--background")
    assert result.returncode == 0, result.stderr
    pid = ops.adopt_pid_file("llm.pid")
    assert pid is not None and pid_alive(pid)
    assert f"PID {pid}" in result.stdout
    assert (ops.project / "logs" / "llm.log").is_file()
    # The detached stub records its call a moment after the script returns.
    deadline = time.monotonic() + 10
    launched: list = []
    while not launched and time.monotonic() < deadline:
        launched = [c for c in ops.stubs.calls("llama-server") if c[1:2] == ["-m"]]
        time.sleep(0.05)
    assert len(launched) == 1
    assert launched[0][-4:] == ["-c", "32768", "-b", "512"]
    # --wait is bounded: the dummy never answers a health path.
    ops.cleanup()
    waited = ops.run("start_llm.sh", "--background", "--wait", "1")
    assert waited.returncode == 1
    assert "no answer after 1 seconds" in waited.stderr


def test_missing_server_program_exits_2(ops):
    _llama(ops)
    ops.stubs.remove("llama-server")
    result = ops.run(
        "start_llm.sh",
        "--background",
        PATH=f"{ops.stubs.bin}:{sys.exec_prefix}/bin:/usr/bin:/bin",
    )
    assert result.returncode == 2
    assert "llama-server is not on PATH" in result.stderr
