"""Live Polymarket trade feed over the public RTDS WebSocket.

RTDS ("real-time data service") broadcasts every fill on Polymarket without
authentication at ``wss://ws-live-data.polymarket.com``. Subscribing to the
``activity`` topic with type ``trades`` gives the streaming counterpart of the
REST ``/trades`` endpoint wrapped in `polymarket_api.py`, so this module's only
job is transport plus normalization onto the same `RawTrade` dataclass the rest
of the pipeline already consumes.

Envelope schema, verified against live frames on 2026-07-25::

    {"connection_id": "...", "topic": "activity", "type": "trades",
     "timestamp": 1785027308241,          # envelope time, milliseconds
     "payload": {"asset": "...", "conditionId": "0x...", "price": 0.45,
                 "proxyWallet": "0x...", "side": "BUY", "size": 4.33,
                 "timestamp": 1785027308,  # trade time, SECONDS
                 "transactionHash": "0x...", ...}}

The payload field names are exactly the ones `RawTrade.from_dict` already
parses, so normalization delegates to it rather than re-deriving the mapping.
The server also emits empty text frames as keepalives; those are dropped.

Nothing else in `src/data` imports this module, and the WebSocket library is
imported lazily inside the default socket factory, so `websocket-client` stays
an optional dependency for everyone who only touches historical data.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

from src.data.polymarket_api import RawTrade

log = logging.getLogger(__name__)

RTDS_URL = "wss://ws-live-data.polymarket.com"

# The one subscription we need: every fill, across every market. RTDS has no
# server-side market filter on this topic, so `--markets` filtering is done
# client-side in `RTDSClient`.
ACTIVITY_TRADES_SUBSCRIPTION: dict[str, Any] = {
    "action": "subscribe",
    "subscriptions": [{"topic": "activity", "type": "trades"}],
}

DEFAULT_CONNECT_TIMEOUT = 15.0  # seconds for the WebSocket handshake
DEFAULT_RECV_TIMEOUT = 20.0  # seconds a blocking recv waits before an idle tick
DEFAULT_STALE_AFTER = 120.0  # seconds without a frame before we warn
DEFAULT_BACKOFF_BASE = 1.0  # seconds; doubles per consecutive failure
DEFAULT_BACKOFF_CAP = 60.0  # seconds; ceiling on the doubling
DEFAULT_BACKOFF_JITTER = 0.25  # fraction of the delay drawn uniformly at random


class RTDSError(RuntimeError):
    """Raised when the RTDS connection cannot be established or maintained."""


class RTDSMessageError(ValueError):
    """Raised when a frame is not a decodable RTDS activity/trades message."""


class RTDSSocket(Protocol):
    """Minimal WebSocket surface `RTDSClient` needs.

    Implemented by the adapter around `websocket-client` below and by the fake
    sockets used in tests. Implementations must raise `TimeoutError` (and not a
    library-specific exception) when `recv` times out, so the client can tell an
    idle feed apart from a broken one.
    """

    def send(self, payload: str) -> None:
        """Send one text frame."""

    def recv(self) -> str:
        """Block until the next text frame arrives; raise TimeoutError on idle."""

    def close(self) -> None:
        """Close the connection, ignoring errors on an already-dead socket."""


# ---------------- Message normalization ----------------


def trade_from_message(message: str | bytes) -> RawTrade | None:
    """Normalize one RTDS frame into a `RawTrade`.

    Delegates every field mapping to `RawTrade.from_dict` — the RTDS payload
    uses the same names as the REST `/trades` rows (`proxyWallet`,
    `conditionId`, `transactionHash`, ...), so the tolerant fallbacks live in
    exactly one place.

    Args:
        message: One raw text frame off the socket.

    Returns:
        The normalized trade, or ``None`` for frames that carry no trade: the
        server's empty keepalive frames and any other topic/type.

    Raises:
        RTDSMessageError: If the frame is not JSON, is not an object, carries a
            non-object payload, or holds field values that do not coerce
            (e.g. a non-numeric price).
    """
    if isinstance(message, bytes):
        message = message.decode("utf-8", errors="replace")
    if not message.strip():
        return None  # keepalive frame

    try:
        envelope = json.loads(message)
    except json.JSONDecodeError as e:
        raise RTDSMessageError(f"undecodable frame: {e}") from e
    if not isinstance(envelope, dict):
        raise RTDSMessageError(f"frame is {type(envelope).__name__}, expected object")

    if envelope.get("topic") != "activity" or envelope.get("type") != "trades":
        return None

    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RTDSMessageError(
            f"activity/trades payload is {type(payload).__name__}, expected object"
        )

    # The trade's own `timestamp` is unix SECONDS; the envelope's is
    # milliseconds. Only fall back to the envelope when the payload omits its
    # own clock, converting units on the way.
    if payload.get("timestamp") is None and envelope.get("timestamp") is not None:
        payload = dict(payload)  # never mutate the caller's dict
        payload["timestamp"] = float(envelope["timestamp"]) / 1000.0

    try:
        return RawTrade.from_dict(payload)
    except (TypeError, ValueError) as e:
        raise RTDSMessageError(f"uncoercible trade payload: {e}") from e


# ---------------- Transport ----------------


class _WebsocketClientSocket:
    """Adapter translating `websocket-client` into the `RTDSSocket` protocol.

    Exists only to map `WebSocketTimeoutException` onto the builtin
    `TimeoutError`, keeping `RTDSClient` free of any library-specific except
    clauses (and therefore testable with a plain fake socket).
    """

    def __init__(self, inner: Any) -> None:
        """Wrap a connected `websocket.WebSocket`."""
        self._inner = inner

    def send(self, payload: str) -> None:
        """Send one text frame."""
        self._inner.send(payload)

    def recv(self) -> str:
        """Receive one text frame, raising TimeoutError when the feed is idle."""
        import websocket  # deferred: optional dependency, see module docstring

        try:
            frame = self._inner.recv()
        except websocket.WebSocketTimeoutException as e:
            raise TimeoutError(str(e)) from e
        return frame.decode("utf-8", "replace") if isinstance(frame, bytes) else frame

    def close(self) -> None:
        """Close the underlying socket, swallowing teardown errors."""
        try:
            self._inner.close()
        except Exception:  # noqa: BLE001 - teardown must never mask the real error
            log.debug("error closing RTDS socket", exc_info=True)


def default_socket_factory(
    url: str = RTDS_URL,
    *,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    recv_timeout: float = DEFAULT_RECV_TIMEOUT,
) -> RTDSSocket:
    """Open a real RTDS WebSocket connection.

    Args:
        url: WebSocket endpoint.
        connect_timeout: Seconds allowed for the handshake.
        recv_timeout: Seconds a blocking `recv` waits before reporting an idle
            tick. Bounded so a silently-dead connection surfaces as staleness
            instead of hanging forever.

    Returns:
        A connected socket satisfying `RTDSSocket`.

    Raises:
        RTDSError: If `websocket-client` is not installed.
    """
    try:
        import websocket  # deferred: optional dependency, see module docstring
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RTDSError(
            "The live RTDS feed needs the 'websocket-client' package. "
            "Install it with: pip install 'websocket-client>=1.8'"
        ) from e

    inner = websocket.create_connection(url, timeout=connect_timeout)
    inner.settimeout(recv_timeout)
    return _WebsocketClientSocket(inner)


def backoff_delay(
    attempt: int,
    rng: random.Random,
    *,
    base: float = DEFAULT_BACKOFF_BASE,
    cap: float = DEFAULT_BACKOFF_CAP,
    jitter: float = DEFAULT_BACKOFF_JITTER,
) -> float:
    """Capped exponential backoff with multiplicative jitter.

    Implements ``delay = min(cap, base * 2**(attempt - 1)) * (1 + jitter * u)``
    with ``u ~ Uniform(0, 1)``. Jitter is one-sided and small so the sequence
    stays recognizably 1s, 2s, 4s, ... while still de-synchronizing clients that
    all lost the same upstream connection.

    Args:
        attempt: 1-based index of the consecutive failure being backed off.
        rng: Source of randomness; pass explicitly for reproducible tests.
        base: Delay after the first failure, in seconds.
        cap: Ceiling on the un-jittered delay, in seconds.
        jitter: Maximum fractional inflation of the delay.

    Returns:
        Seconds to sleep before the next connection attempt.

    Raises:
        ValueError: If `attempt` is not at least 1.
    """
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt}")
    # 2**(attempt-1) overflows to inf for absurd attempt counts; min() with the
    # cap keeps that harmless, so no extra guard is needed.
    return min(cap, base * 2.0 ** (attempt - 1)) * (1.0 + jitter * rng.random())


class RTDSClient:
    """Streaming client for the RTDS activity/trades topic.

    Yields normalized `RawTrade` objects forever, reconnecting with capped
    exponential backoff whenever the socket drops. Malformed frames are logged
    and skipped rather than killing the stream — a single bad frame must not
    end a long-running collection run.

    All I/O is behind the injected `socket_factory`, `sleep` and `clock`
    callables, so the whole reconnect/staleness state machine is testable
    without a network.
    """

    def __init__(
        self,
        *,
        socket_factory: Callable[[], RTDSSocket] | None = None,
        condition_ids: Iterable[str] | None = None,
        stale_after: float = DEFAULT_STALE_AFTER,
        max_reconnects: int | None = None,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        backoff_jitter: float = DEFAULT_BACKOFF_JITTER,
        rng: random.Random | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the client.

        Args:
            socket_factory: Zero-argument factory returning a fresh connected
                socket. Defaults to `default_socket_factory` against
                `RTDS_URL`; tests inject a fake.
            condition_ids: If given, only trades on these market condition ids
                are yielded. Matching is case-insensitive.
            stale_after: Seconds without any frame before a warning is logged.
            max_reconnects: Cap on consecutive failed connection attempts before
                `RTDSError` is raised. ``None`` retries forever.
            backoff_base: Delay after the first failed attempt, in seconds.
            backoff_cap: Ceiling on the backoff delay, in seconds.
            backoff_jitter: Maximum fractional inflation of each delay.
            rng: Source of jitter randomness; defaults to a fresh
                `random.Random`. Pass one for deterministic tests.
            sleep: Blocking sleep, injectable for tests.
            clock: Monotonic clock used for staleness, injectable for tests.
        """
        self._socket_factory = socket_factory or default_socket_factory
        self.condition_ids = (
            frozenset(c.lower() for c in condition_ids) if condition_ids else None
        )
        self.stale_after = float(stale_after)
        self.max_reconnects = max_reconnects
        self._backoff_base = float(backoff_base)
        self._backoff_cap = float(backoff_cap)
        self._backoff_jitter = float(backoff_jitter)
        self._rng = rng if rng is not None else random.Random()
        self._sleep = sleep
        self._clock = clock
        self._last_frame_at = 0.0
        self._last_stale_warning_at = 0.0

    def stream(self, *, max_trades: int | None = None) -> Iterator[RawTrade]:
        """Yield normalized trades from the live feed, reconnecting as needed.

        Args:
            max_trades: Stop after yielding this many trades. ``None`` streams
                until the caller stops consuming (or interrupts).

        Yields:
            One `RawTrade` per matching activity/trades frame, in arrival order.

        Raises:
            RTDSError: If `max_reconnects` consecutive connection attempts fail.
        """
        n_yielded = 0
        failures = 0
        while max_trades is None or n_yielded < max_trades:
            if failures:
                self._backoff(failures)
            try:
                sock = self._socket_factory()
                sock.send(json.dumps(ACTIVITY_TRADES_SUBSCRIPTION))
            except RTDSError:
                raise  # a missing dependency will not fix itself on retry
            except Exception as e:  # noqa: BLE001 - any connect error is retryable
                failures += 1
                log.warning("RTDS connect failed (attempt %d): %s", failures, e)
                continue

            log.info("RTDS connected; subscribed to activity/trades")
            self._last_frame_at = self._clock()
            self._last_stale_warning_at = self._last_frame_at
            try:
                for trade in self._read_frames(sock):
                    # A delivered frame proves the endpoint is healthy, so the
                    # next drop backs off from 1s again rather than from the
                    # capped delay of an old outage.
                    failures = 0
                    if trade is None:
                        continue
                    yield trade
                    n_yielded += 1
                    if max_trades is not None and n_yielded >= max_trades:
                        return
            except Exception as e:  # noqa: BLE001 - any read error is retryable
                failures += 1
                log.warning("RTDS stream dropped (attempt %d): %s", failures, e)
            else:
                # Unreachable today — `_read_frames` loops forever and only ever
                # leaves via an exception. Kept as the backoff guard for the day
                # it grows a clean exit: falling through with failures == 0 would
                # reconnect with no delay at all, hammering the endpoint.
                failures += 1
                log.warning("RTDS stream ended cleanly (attempt %d)", failures)
            finally:
                sock.close()

    def _read_frames(self, sock: RTDSSocket) -> Iterator[RawTrade | None]:
        """Yield trades (or ``None`` per non-trade frame) until the socket dies.

        ``None`` is yielded rather than swallowed so the caller can reset its
        backoff counter on keepalives too — those still prove liveness.
        """
        while True:
            try:
                raw = sock.recv()
            except TimeoutError:
                # Idle feed, not a failure: RTDS is quiet between fills.
                self._warn_if_stale()
                continue
            self._last_frame_at = self._clock()
            yield self._normalize(raw)

    def _normalize(self, raw: str) -> RawTrade | None:
        """Normalize one frame, dropping keepalives, filtered and bad messages."""
        try:
            trade = trade_from_message(raw)
        except RTDSMessageError as e:
            log.warning("skipping malformed RTDS frame: %s", e)
            return None
        if trade is None:
            return None
        if self.condition_ids is not None:
            if trade.condition_id.lower() not in self.condition_ids:
                return None
        return trade

    def _backoff(self, failures: int) -> None:
        """Sleep before retry number `failures`, or give up past the cap."""
        if self.max_reconnects is not None and failures > self.max_reconnects:
            raise RTDSError(
                f"RTDS unreachable after {failures - 1} consecutive attempts"
            )
        delay = backoff_delay(
            failures,
            self._rng,
            base=self._backoff_base,
            cap=self._backoff_cap,
            jitter=self._backoff_jitter,
        )
        log.info("reconnecting to RTDS in %.2fs", delay)
        self._sleep(delay)

    def _warn_if_stale(self) -> None:
        """Log a warning once per `stale_after` window while the feed is silent."""
        now = self._clock()
        silent_for = now - self._last_frame_at
        if silent_for < self.stale_after:
            return
        if now - self._last_stale_warning_at < self.stale_after:
            return
        self._last_stale_warning_at = now
        log.warning("no RTDS frame for %.0fs (feed may be stalled)", silent_for)
