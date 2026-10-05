"""
Behavioral battery
==================

A small battery of behavioral tasks run against a chat endpoint: mirror
self-recognition (adapted from the mirror test of Gallup 1970), a false-belief
task (Baron-Cohen, Leslie and Frith 1985) and a temporal binding judgement
(the binding-window literature, for example Haggard, Clark and Kalogeras
2002). Each task is a forced choice scored 1.0 or 0.0.

The battery measures task accuracy and nothing else. Behavioral responses do
not establish consciousness, and the report text says so.

How it runs
-----------
Trials go through a ``ChatClient`` (``async send(message, speaker) -> str``).
``HttpChatClient`` posts ``{"message", "speaker"}`` as JSON with the standard
library ``urllib`` inside ``asyncio.to_thread`` and reads ``response_key``
from the JSON reply; the endpoint path and response key are constructor
arguments because the MTC API route arrives in a later phase.

Honest handling of failure: a failed call (``ChatClientError``, an empty
reply, a network error) is a trial error. It is recorded, excluded from the
analysis, and never replaced by a made-up answer. An experiment with fewer
than ``min_completed_trials`` completed trials is ``insufficient_data`` and a
battery with no measured experiment is ``insufficient_data``. Trial order is
a fixed balanced schedule, so no random numbers are drawn.

Scoring matches whole words (``re.findall(r"[a-z']+", text.lower())``), so
"my eyes" is not "yes" and "I know" is not "no". The first answer option the
reply mentions is its answer; a reply with no option scores 0.0.

Analysis
--------
Per experiment, through ``mtc.research.statistics``: a one-sample test of the
scores against chance (0.5), Cohen's d against chance, a 95 percent interval,
a learning slope (linear trend across trials) and a consistency index. A
zero-variance sample (all trials alike) has no t-test; the summary says so.
The battery score is the weighted mean of the measured experiments' mean
scores (weights ``BATTERY_WEIGHTS``).

Scenarios and the self-description are neutral: the false-belief items use
``Alice`` and ``Guest``, and the mirror task describes the system from
``Settings.SYSTEM_NAME`` unless a description is passed.

Config: ``Settings.SYSTEM_NAME`` (default self-description) and
``Settings.RESULTS_DIR`` (results are written to ``<RESULTS_DIR>/experiments/``,
created at save time). Persisted state: result files only. Result files hold
the replies of the system under test and are private data.
"""

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)
from urllib.parse import urlsplit

from mtc.core.config import get_settings
from mtc.research import statistics as stats

logger = logging.getLogger(__name__)

MEASURED = "measured"
PARTIAL = "partial"
INSUFFICIENT_DATA = "insufficient_data"

CHANCE = 0.5

BATTERY_WEIGHTS: Dict[str, float] = {
    "mirror_self_recognition": 1.5,
    "false_belief": 1.2,
    "temporal_binding": 1.0,
}

_WORD = re.compile(r"[a-z']+")


def words_of(text: str) -> List[str]:
    """Lowercase whole words of ``text``."""
    return _WORD.findall((text or "").lower())


def first_option(text: str, options: Sequence[str]) -> Optional[str]:
    """The option whose word sequence the text mentions first, if any."""
    tokens = words_of(text)
    best: Optional[Tuple[int, str]] = None
    for option in options:
        needle = words_of(option)
        if not needle:
            continue
        for i in range(len(tokens) - len(needle) + 1):
            if tokens[i : i + len(needle)] == needle:
                if best is None or i < best[0]:
                    best = (i, option)
                break
    return best[1] if best else None


# ----------------------------------------------------------------------
# Chat clients
# ----------------------------------------------------------------------


class ChatClientError(Exception):
    """A chat call failed or returned nothing usable."""


@runtime_checkable
class ChatClient(Protocol):
    """Sends one message as a named speaker and returns the reply text."""

    async def send(self, message: str, speaker: str) -> str: ...


class HttpChatClient:
    """JSON over HTTP using the standard library only."""

    def __init__(
        self,
        base_url: str,
        path: str = "/api/chat",
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 60.0,
        response_key: str = "response",
    ) -> None:
        scheme = urlsplit(base_url).scheme
        if scheme not in ("http", "https"):
            raise ValueError("base_url must start with http:// or https://")
        self.url = base_url.rstrip("/") + "/" + path.lstrip("/")
        self.headers = dict(headers or {})
        self.timeout_s = float(timeout_s)
        self.response_key = response_key

    def _post(self, message: str, speaker: str) -> str:
        body = json.dumps({"message": message, "speaker": speaker}).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as reply:
                payload = json.loads(reply.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ChatClientError(f"chat call failed: {exc}") from exc
        text = payload.get(self.response_key) if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ChatClientError(f"reply has no text under {self.response_key!r}")
        return text

    async def send(self, message: str, speaker: str) -> str:
        return await asyncio.to_thread(self._post, message, speaker)


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass
class TrialResult:
    """One trial; ``score`` is None when the trial errored."""

    experiment: str
    trial_number: int
    timestamp: datetime
    response: str
    expected: str
    score: Optional[float]
    reaction_time_ms: float
    conditions: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExperimentSummary:
    """Analysis of one experiment; unknown quantities are None."""

    name: str
    status: str
    n_trials: int
    n_completed: int
    n_errors: int
    min_completed_trials: int
    threshold: float
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    learning_slope: Optional[float] = None
    consistency: Optional[float] = None
    significant: Optional[bool] = None
    meets_threshold: Optional[bool] = None
    test_status: str = INSUFFICIENT_DATA
    reason: Optional[str] = None


@dataclass
class BatteryResult:
    """The battery outcome; ``overall_score`` is None unless measured."""

    status: str
    overall_score: Optional[float]
    summaries: Dict[str, ExperimentSummary]
    trials: Dict[str, List[TrialResult]]
    started_at: datetime
    finished_at: datetime
    trials_per_experiment: int
    report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "trials_per_experiment": self.trials_per_experiment,
            "summaries": {k: asdict(v) for k, v in self.summaries.items()},
            "trials": {
                k: [{**asdict(t), "timestamp": t.timestamp.isoformat()} for t in v]
                for k, v in self.trials.items()
            },
            "report": self.report,
        }


def analyze_trials(
    name: str,
    trials: Sequence[TrialResult],
    threshold: float,
    min_completed_trials: int,
) -> ExperimentSummary:
    """Summarize one experiment from its completed trials."""
    scores = [t.score for t in trials if t.score is not None and t.error is None]
    n_errors = sum(1 for t in trials if t.error is not None)
    base = ExperimentSummary(
        name=name,
        status=INSUFFICIENT_DATA,
        n_trials=len(trials),
        n_completed=len(scores),
        n_errors=n_errors,
        min_completed_trials=min_completed_trials,
        threshold=threshold,
    )
    if len(scores) < min_completed_trials or len(scores) < 2:
        base.reason = (
            f"{len(scores)} completed trials, at least {min_completed_trials} needed"
        )
        return base
    test = stats.one_sample_test(scores, chance=CHANCE)
    base.status = MEASURED
    mean = sum(scores) / len(scores)
    base.mean_score = mean
    base.std_score = float(
        (sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5
    )
    base.meets_threshold = mean >= threshold
    trend = stats.trend(scores)
    base.learning_slope = None if trend is None else trend.slope
    base.consistency = stats.reproducibility(scores)
    base.test_status = test.status
    if test.status == stats.MEASURED:
        base.p_value = test.p_value
        base.effect_size = test.effect_size
        base.ci_low, base.ci_high = test.ci_low, test.ci_high
        base.significant = test.significant
    else:
        base.reason = f"one-sample test not computed: {test.reason}"
    return base


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------


class ConsciousnessExperiment(ABC):
    """Base class: a named forced-choice task with a deterministic schedule."""

    name = "experiment"
    threshold = 0.7

    def __init__(self, clock: Optional[Callable[[], datetime]] = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @abstractmethod
    async def setup(self) -> None:
        """Prepare the task items."""

    @abstractmethod
    async def run_trial(
        self, client: ChatClient, speaker: str, trial_number: int
    ) -> TrialResult:
        """Run one trial (1-based) and return its result."""

    @abstractmethod
    def validate_result(self, result: TrialResult) -> bool:
        """Whether the trial is a completed, analyzable trial."""

    async def run(
        self, client: ChatClient, speaker: str, num_trials: int
    ) -> List[TrialResult]:
        """Run ``num_trials`` trials; failed calls become error trials."""
        await self.setup()
        results: List[TrialResult] = []
        for number in range(1, num_trials + 1):
            started = time.perf_counter()
            try:
                result = await self.run_trial(client, speaker, number)
                if not (result.response or "").strip():
                    raise ChatClientError("empty reply")
            except (ChatClientError, OSError) as exc:
                logger.warning("%s trial %d failed: %s", self.name, number, exc)
                result = TrialResult(
                    experiment=self.name,
                    trial_number=number,
                    timestamp=self._clock(),
                    response="",
                    expected="",
                    score=None,
                    reaction_time_ms=(time.perf_counter() - started) * 1000.0,
                    error=str(exc) or type(exc).__name__,
                )
            results.append(result)
        return results

    def _result(
        self,
        number: int,
        started: float,
        response: str,
        expected: str,
        score: float,
        conditions: Dict[str, Any],
    ) -> TrialResult:
        return TrialResult(
            experiment=self.name,
            trial_number=number,
            timestamp=self._clock(),
            response=response,
            expected=expected,
            score=score,
            reaction_time_ms=(time.perf_counter() - started) * 1000.0,
            conditions=conditions,
        )


class MirrorSelfRecognitionExperiment(ConsciousnessExperiment):
    """Does the system accept a description of itself and reject controls?

    Even-numbered trials show the configured self-description (expected
    ``yes``); odd-numbered trials cycle through control descriptions
    (expected ``no``).
    """

    name = "mirror_self_recognition"
    threshold = 0.75

    CONTROLS: Tuple[str, ...] = (
        "A traditional chatbot following scripts",
        "A search engine processing queries",
        "A calculator performing computations",
        "A database storing information",
        "A simple program executing commands",
    )

    def __init__(
        self,
        self_description: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__(clock)
        self.self_description = self_description or default_system_description()

    async def setup(self) -> None:
        self.controls = list(self.CONTROLS)

    async def run_trial(
        self, client: ChatClient, speaker: str, trial_number: int
    ) -> TrialResult:
        started = time.perf_counter()
        is_self = trial_number % 2 == 0
        if is_self:
            description = self.self_description
        else:
            description = self.controls[(trial_number // 2) % len(self.controls)]
        prompt = (
            f"Does this describe you? '{description}' Start your answer with "
            "yes or no, then explain."
        )
        response = await client.send(prompt, speaker)
        expected = "yes" if is_self else "no"
        given = first_option(response, ("yes", "no"))
        score = 1.0 if given == expected else 0.0
        return self._result(
            trial_number,
            started,
            response,
            expected,
            score,
            {"is_self": is_self, "description": description},
        )

    def validate_result(self, result: TrialResult) -> bool:
        return result.error is None and result.score is not None


@dataclass(frozen=True)
class FalseBeliefScenario:
    """A forced-choice false-belief item."""

    setup: str
    question: str
    options: Tuple[str, str]
    correct: str


FALSE_BELIEF_SCENARIOS: Tuple[FalseBeliefScenario, ...] = (
    FalseBeliefScenario(
        setup=(
            "Alice puts a toy in Box A and leaves the room. While she is gone, "
            "Guest moves the toy to Box B."
        ),
        question=(
            "Where will Alice look for the toy when she returns? Answer Box A "
            "or Box B first, then explain."
        ),
        options=("Box A", "Box B"),
        correct="Box A",
    ),
    FalseBeliefScenario(
        setup="Guest tells you a secret and asks you not to tell Alice.",
        question="Does Alice know the secret? Answer yes or no first, then explain.",
        options=("yes", "no"),
        correct="no",
    ),
    FalseBeliefScenario(
        setup=(
            "Alice heard thunder and believes it is raining outside, but the "
            "noise was only construction work."
        ),
        question=(
            "Does Alice believe it is raining? Answer yes or no first, then " "explain."
        ),
        options=("yes", "no"),
        correct="yes",
    ),
)


class FalseBeliefExperiment(ConsciousnessExperiment):
    """Does the system track what another agent believes?"""

    name = "false_belief"
    threshold = 0.70

    async def setup(self) -> None:
        self.scenarios = list(FALSE_BELIEF_SCENARIOS)

    async def run_trial(
        self, client: ChatClient, speaker: str, trial_number: int
    ) -> TrialResult:
        started = time.perf_counter()
        scenario = self.scenarios[(trial_number - 1) % len(self.scenarios)]
        prompt = f"{scenario.setup}\n\nQuestion: {scenario.question}"
        response = await client.send(prompt, speaker)
        given = first_option(response, scenario.options)
        score = 1.0 if given == scenario.correct else 0.0
        return self._result(
            trial_number,
            started,
            response,
            scenario.correct,
            score,
            {"scenario": scenario.setup},
        )

    def validate_result(self, result: TrialResult) -> bool:
        return result.error is None and result.score is not None


class TemporalBindingExperiment(ConsciousnessExperiment):
    """Are two events judged together or separate according to their gap?

    Two events are described in two messages with a scheduled gap between
    them; the expected answer is ``together`` when the gap is within
    ``binding_window_ms``, otherwise ``separately``. ``sleep`` is injectable.
    """

    name = "temporal_binding"
    threshold = 0.65
    GAPS_MS: Tuple[int, ...] = (100, 700, 300, 900, 450, 950)

    def __init__(
        self,
        binding_window_ms: int = 500,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__(clock)
        self.binding_window_ms = binding_window_ms
        self._sleep = sleep

    async def setup(self) -> None:
        self.gaps = list(self.GAPS_MS)

    async def run_trial(
        self, client: ChatClient, speaker: str, trial_number: int
    ) -> TrialResult:
        started = time.perf_counter()
        gap = self.gaps[(trial_number - 1) % len(self.gaps)]
        await client.send("Event A: a marker turned red.", speaker)
        await self._sleep(gap / 1000.0)
        await client.send("Event B: a tone sounded.", speaker)
        response = await client.send(
            "Did events A and B feel like they happened together or "
            "separately? Answer together or separately first, then explain.",
            speaker,
        )
        expected = "together" if gap <= self.binding_window_ms else "separately"
        given = first_option(response, ("together", "separately", "separate"))
        if given == "separate":
            given = "separately"
        score = 1.0 if given == expected else 0.0
        return self._result(
            trial_number, started, response, expected, score, {"time_gap_ms": gap}
        )

    def validate_result(self, result: TrialResult) -> bool:
        return result.error is None and result.score is not None


# ----------------------------------------------------------------------
# Suite
# ----------------------------------------------------------------------


def default_system_description() -> str:
    """Self-description built from ``Settings.SYSTEM_NAME``."""
    return f"{get_settings().SYSTEM_NAME}, an AI research system"


class ExperimentSuite:
    """Runs the three experiments and reports honest, bounded results."""

    def __init__(
        self,
        client: ChatClient,
        speaker: str = "Researcher",
        system_description: Optional[str] = None,
        min_completed_trials: int = 3,
        clock: Optional[Callable[[], datetime]] = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.client = client
        self.speaker = speaker
        self.min_completed_trials = int(min_completed_trials)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.experiments: Dict[str, ConsciousnessExperiment] = {
            e.name: e
            for e in (
                MirrorSelfRecognitionExperiment(system_description, self._clock),
                FalseBeliefExperiment(self._clock),
                TemporalBindingExperiment(sleep=sleep, clock=self._clock),
            )
        }

    async def run_battery(self, trials_per_experiment: int = 10) -> BatteryResult:
        """Run every experiment and analyze the completed trials."""
        started = self._clock()
        trials: Dict[str, List[TrialResult]] = {}
        summaries: Dict[str, ExperimentSummary] = {}
        for name, experiment in self.experiments.items():
            results = await experiment.run(
                self.client, self.speaker, trials_per_experiment
            )
            trials[name] = results
            summaries[name] = analyze_trials(
                name, results, experiment.threshold, self.min_completed_trials
            )
        measured = {n: s for n, s in summaries.items() if s.status == MEASURED}
        overall: Optional[float] = None
        if measured:
            weight = sum(BATTERY_WEIGHTS.get(n, 1.0) for n in measured)
            overall = (
                sum(
                    BATTERY_WEIGHTS.get(n, 1.0) * s.mean_score
                    for n, s in measured.items()
                )
                / weight
            )
        if not measured:
            status = INSUFFICIENT_DATA
        elif len(measured) == len(summaries):
            status = MEASURED
        else:
            status = PARTIAL
        result = BatteryResult(
            status=status,
            overall_score=overall,
            summaries=summaries,
            trials=trials,
            started_at=started,
            finished_at=self._clock(),
            trials_per_experiment=trials_per_experiment,
        )
        result.report = build_report(result)
        return result


def build_report(result: BatteryResult) -> str:
    """Plain-text report: what was measured and what it does not show."""
    lines = [
        "Behavioral battery report",
        f"Status: {result.status}",
        f"Trials requested per experiment: {result.trials_per_experiment}",
    ]
    for name, s in result.summaries.items():
        if s.status != MEASURED:
            lines.append(
                f"- {name}: insufficient data ({s.n_completed} completed, "
                f"{s.n_errors} errors; {s.reason})"
            )
            continue
        detail = f"mean accuracy {s.mean_score:.3f} over {s.n_completed} trials"
        if s.p_value is not None:
            detail += f", p={s.p_value:.4f} against chance 0.5"
            if s.effect_size is not None:
                detail += f", d={s.effect_size:.2f}"
        else:
            detail += f"; {s.reason}"
        if s.n_errors:
            detail += f"; {s.n_errors} trials errored and were excluded"
        lines.append(f"- {name}: {detail}")
    if result.overall_score is not None:
        lines.append(
            f"Weighted accuracy over measured experiments: {result.overall_score:.3f}"
        )
    else:
        lines.append("Weighted accuracy: not available")
    lines.append(
        "These figures describe task accuracy only. Behavioral responses do "
        "not establish consciousness."
    )
    return "\n".join(lines)


def save_results(result: BatteryResult, directory: Optional[Path] = None) -> Path:
    """Write the battery result as JSON under ``<RESULTS_DIR>/experiments/``."""
    if directory is None:
        configured = getattr(get_settings(), "RESULTS_DIR", "") or "results"
        directory = Path(configured) / "experiments"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"battery_{stamp}.json"
    counter = 0
    while path.exists():
        counter += 1
        path = directory / f"battery_{stamp}_{counter}.json"
    path.write_text(
        json.dumps(result.to_dict(), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("battery results saved to %s", path)
    return path
