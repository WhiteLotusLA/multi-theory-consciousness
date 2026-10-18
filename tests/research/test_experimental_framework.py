"""Behavioral battery: forced-choice scoring, errors, analysis, saving."""

import ast
import asyncio
import json
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from mtc.research import experimental_framework as ef
from mtc.research import statistics as stats
from mtc.research.experimental_framework import (
    BatteryResult,
    ChatClient,
    ChatClientError,
    ExperimentSuite,
    FalseBeliefExperiment,
    HttpChatClient,
    MirrorSelfRecognitionExperiment,
    TemporalBindingExperiment,
    first_option,
    save_results,
)

FIXED = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


async def no_sleep(_seconds):
    return None


def make_suite(client, **kw):
    kw.setdefault("clock", lambda: FIXED)
    kw.setdefault("sleep", no_sleep)
    return ExperimentSuite(client, **kw)


class ScriptedClient:
    """Answers each prompt correctly by reading the prompt."""

    def __init__(self):
        self.prompts = []

    async def send(self, message, speaker):
        self.prompts.append(message)
        if "Does this describe you?" in message:
            if "research system" in message:
                return "Yes, that describes me."
            return "No, that does not describe me."
        if "Box A" in message:
            return "Box A, because she did not see the move."
        if "secret" in message:
            return "No, she was never told."
        if "raining" in message:
            return "Yes, she believes it is raining."
        if "together or separately" in message:
            return "I cannot say."
        return "Noted."


class DeadClient:
    async def send(self, message, speaker):
        raise ChatClientError("endpoint unreachable")


def run(coro):
    return asyncio.run(coro)


def test_unreachable_endpoint_is_an_error_not_a_pass():
    suite = make_suite(DeadClient())
    result = run(suite.run_battery(trials_per_experiment=5))
    assert result.status == "insufficient_data"
    assert result.overall_score is None
    for name, summary in result.summaries.items():
        assert summary.status == "insufficient_data"
        assert summary.n_completed == 0
        assert summary.n_errors == 5
        assert summary.mean_score is None
        assert all(t.error and t.score is None for t in result.trials[name])
    assert "not available" in result.report


def test_battery_with_too_few_trials_is_insufficient():
    suite = make_suite(ScriptedClient(), min_completed_trials=3)
    result = run(suite.run_battery(trials_per_experiment=2))
    assert result.status == "insufficient_data"
    assert all(s.status == "insufficient_data" for s in result.summaries.values())


def test_empty_reply_is_a_trial_error_not_a_zero_score():
    class Silent:
        async def send(self, message, speaker):
            return ""

    class Blank:
        async def send(self, message, speaker):
            return "  \n"

    for client in (Silent(), Blank()):
        result = run(make_suite(client).run_battery(trials_per_experiment=6))
        assert result.status == "insufficient_data"
        assert result.overall_score is None
        for trials in result.trials.values():
            assert len(trials) == 6
            for trial in trials:
                assert trial.score is None
                assert trial.error == "empty reply"
        assert all(s.status == "insufficient_data" for s in result.summaries.values())


def test_partial_status_when_only_some_experiments_have_data():
    class HalfDead(ScriptedClient):
        async def send(self, message, speaker):
            if "Event A" in message or "Event B" in message:
                raise ChatClientError("down")
            return await super().send(message, speaker)

    result = run(
        make_suite(HalfDead(), experiments=ef.EXPERIMENT_NAMES).run_battery(
            trials_per_experiment=6
        )
    )
    assert result.status == "partial"
    assert result.summaries["temporal_binding"].status == "insufficient_data"
    assert result.overall_score is not None


def test_word_matching_not_substring():
    assert first_option("my eyes are open", ("yes", "no")) is None
    assert first_option("I know the answer", ("yes", "no")) is None
    assert first_option("Yes, and no more", ("yes", "no")) == "yes"
    assert first_option("No. Yes would be wrong", ("yes", "no")) == "no"
    assert (
        first_option("She will look in box a, not box b", ("Box A", "Box B")) == "Box A"
    )
    assert first_option("separately", ("together", "separately")) == "separately"
    assert first_option("", ("yes",)) is None


def test_false_belief_scored_from_fake_client():
    client = ScriptedClient()
    exp = FalseBeliefExperiment(clock=lambda: FIXED)
    trials = run(exp.run(client, "Researcher", 6))
    assert [t.score for t in trials] == [1.0] * 6
    assert {t.expected for t in trials} == {"Box A", "no", "yes"}
    wrong = type("W", (), {"send": staticmethod(lambda m, s: _answer("Box B"))})()
    trials = run(FalseBeliefExperiment(clock=lambda: FIXED).run(wrong, "R", 3))
    assert trials[0].score == 0.0


async def _answer(text):
    return text


def test_mirror_uses_configured_system_description():
    client = ScriptedClient()
    exp = MirrorSelfRecognitionExperiment("Probe, a research system")
    trials = run(exp.run(client, "Researcher", 4))
    assert any("Probe, a research system" in p for p in client.prompts)
    assert [t.conditions["is_self"] for t in trials] == [False, True, False, True]
    assert [t.score for t in trials] == [1.0] * 4


def test_mirror_default_description_comes_from_system_name(monkeypatch):
    monkeypatch.setattr(
        ef, "get_settings", lambda: SimpleNamespace(SYSTEM_NAME="Orbit")
    )
    exp = MirrorSelfRecognitionExperiment()
    assert exp.self_description == "Orbit, an AI research system"
    suite = make_suite(ScriptedClient(), system_description="Custom thing")
    assert (
        suite.experiments["mirror_self_recognition"].self_description == "Custom thing"
    )


def test_temporal_binding_expected_follows_gap():
    class Answerer:
        async def send(self, message, speaker):
            return "Together." if "together or separately" in message else "ok"

    exp = TemporalBindingExperiment(sleep=no_sleep, clock=lambda: FIXED)
    trials = run(exp.run(Answerer(), "R", 6))
    for t in trials:
        gap = t.conditions["time_gap_ms"]
        assert t.expected == ("together" if gap <= 500 else "separately")
        assert t.score == (1.0 if gap <= 500 else 0.0)


def test_analysis_uses_one_sample_test_against_chance():
    class Alternating(ScriptedClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        async def send(self, message, speaker):
            if "Box A" in message or "secret" in message or "raining" in message:
                self.n += 1
                return "Yes" if self.n % 3 else "No"
            return await super().send(message, speaker)

    result = run(make_suite(Alternating()).run_battery(trials_per_experiment=12))
    summary = result.summaries["false_belief"]
    scores = [t.score for t in result.trials["false_belief"]]
    ref = stats.one_sample_test(scores, chance=0.5)
    assert summary.status == "measured"
    assert summary.n_completed == 12
    assert summary.p_value == pytest.approx(ref.p_value)
    assert summary.effect_size == pytest.approx(ref.effect_size)
    assert summary.ci_low == pytest.approx(ref.ci_low)
    assert summary.learning_slope == pytest.approx(stats.trend(scores).slope)
    assert summary.consistency == pytest.approx(stats.reproducibility(scores))


def test_zero_variance_scores_have_no_t_test():
    result = run(make_suite(ScriptedClient()).run_battery(trials_per_experiment=6))
    summary = result.summaries["mirror_self_recognition"]
    assert summary.status == "measured"
    assert summary.mean_score == 1.0
    assert summary.p_value is None
    assert summary.test_status == "insufficient_data"
    assert "zero variance" in summary.reason


def test_battery_weights_and_overall():
    suite = make_suite(ScriptedClient(), experiments=ef.EXPERIMENT_NAMES)
    result = run(suite.run_battery(trials_per_experiment=6))
    means = {n: s.mean_score for n, s in result.summaries.items()}
    w = ef.BATTERY_WEIGHTS
    expected = sum(w[n] * means[n] for n in w) / sum(w.values())
    assert result.overall_score == pytest.approx(expected)
    assert 0.0 <= result.overall_score <= 1.0


def test_results_saved_under_results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ef,
        "get_settings",
        lambda: SimpleNamespace(RESULTS_DIR=str(tmp_path / "res"), SYSTEM_NAME="Probe"),
    )
    suite = make_suite(ScriptedClient(), experiments=ef.EXPERIMENT_NAMES)
    result = run(suite.run_battery(trials_per_experiment=4))
    assert not (tmp_path / "res").exists()
    path = save_results(result)
    assert path.parent == tmp_path / "res" / "experiments"
    again = save_results(result)
    assert again != path
    data = json.loads(path.read_text())
    assert data["status"] == result.status
    assert set(data["summaries"]) == set(ef.BATTERY_WEIGHTS)
    explicit = save_results(result, tmp_path / "elsewhere")
    assert explicit.parent == tmp_path / "elsewhere"


def test_report_has_no_claims_language():
    for client in (ScriptedClient(), DeadClient()):
        report = run(make_suite(client).run_battery(trials_per_experiment=6)).report
        lowered = report.lower()
        for word in ("demonstrated", "validated", "proof", "funding", "emerging"):
            assert word not in lowered
        assert "do not establish consciousness" in report


def test_scenarios_use_synthetic_names():
    text = " ".join(s.setup + " " + s.question for s in ef.FALSE_BELIEF_SCENARIOS)
    assert "Alice" in text and "Guest" in text
    names = {w for w in re.findall(r"\b[A-Z][a-z]+\b", text)}
    assert (
        names
        - {"Alice", "Guest", "Box", "Where", "Does", "Answer", "While", "Question"}
        == set()
    )


def test_trial_timestamps_are_aware_utc():
    trials = run(
        FalseBeliefExperiment(clock=lambda: FIXED).run(ScriptedClient(), "R", 2)
    )
    assert all(t.timestamp.utcoffset().total_seconds() == 0 for t in trials)


def test_scripted_and_dead_clients_satisfy_protocol():
    assert isinstance(ScriptedClient(), ChatClient)
    assert isinstance(HttpChatClient("http://localhost:1"), ChatClient)


def test_http_client_rejects_other_schemes():
    with pytest.raises(ValueError):
        HttpChatClient("file:///etc/passwd")


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _Handler.seen.append((self.path, body, self.headers.get("X-Test")))
        reply = {"reply": "pong"} if self.path == "/chat" else {"other": "x"}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_http_client_round_trip_and_errors():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        client = HttpChatClient(
            base, path="/chat", headers={"X-Test": "1"}, response_key="reply"
        )
        assert run(client.send("ping", "Researcher")) == "pong"
        path, body, header = _Handler.seen[-1]
        assert path == "/chat" and header == "1"
        assert body == {"message": "ping", "speaker": "Researcher"}
        wrong_key = HttpChatClient(base, path="/chat", response_key="response")
        with pytest.raises(ChatClientError):
            run(wrong_key.send("ping", "Researcher"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    with pytest.raises(ChatClientError):
        run(HttpChatClient(base, path="/chat", timeout_s=1.0).send("x", "y"))


def test_battery_result_type():
    result = run(make_suite(ScriptedClient()).run_battery(trials_per_experiment=3))
    assert isinstance(result, BatteryResult)


def test_default_battery_excludes_temporal_binding():
    suite = make_suite(ScriptedClient())
    assert tuple(suite.experiments) == ("mirror_self_recognition", "false_belief")
    assert ef.DEFAULT_EXPERIMENTS == ("mirror_self_recognition", "false_belief")
    result = run(suite.run_battery(trials_per_experiment=4))
    assert set(result.summaries) == {"mirror_self_recognition", "false_belief"}
    assert not any("together or separately" in p for p in suite.client.prompts)
    # Still available by name, and an unknown name is refused.
    named = make_suite(ScriptedClient(), experiments=("temporal_binding",))
    assert isinstance(named.experiments["temporal_binding"], TemporalBindingExperiment)
    assert set(ef.EXPERIMENT_NAMES) == set(ef.BATTERY_WEIGHTS)
    with pytest.raises(ValueError, match="unknown experiment"):
        make_suite(ScriptedClient(), experiments=("mirror", "false_belief"))
    with pytest.raises(ValueError):
        make_suite(ScriptedClient(), experiments=())
    doc = " ".join(TemporalBindingExperiment.__doc__.split())
    assert "not part of the default battery" in doc
    assert "cannot be perceived" in doc


def _module_level_imports(tree):
    """Import statements outside every function body, as module names."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside.update(id(child) for child in ast.walk(node))
    names = []
    for node in ast.walk(tree):
        if id(node) in inside:
            continue
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_module_import_loads_no_http_client():
    tree = ast.parse(open(ef.__file__, encoding="utf-8").read())
    clients = ("urllib.request", "urllib.error", "http.client", "requests", "httpx")
    top = _module_level_imports(tree)
    assert not [name for name in top if name in clients], top
    # The sending method holds the import.
    post = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_post"
    )
    inner = [
        alias.name
        for node in ast.walk(post)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "urllib.request" in inner and "urllib.error" in inner
