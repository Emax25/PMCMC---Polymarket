"""Tests for the live RTDS trade feed (`src/data/rtds.py`).

No test touches the network: every case drives `RTDSClient` through an injected
fake socket. The normalization fixture below is a byte-for-byte capture of a
real ``activity``/``trades`` frame taken from wss://ws-live-data.polymarket.com
on 2026-07-25, so the field mapping is pinned to the wire format rather than to
documentation.
"""

from __future__ import annotations

import json
import logging
import random

import pytest

from src.data.polymarket_api import RawTrade
from src.data.rtds import (
    ACTIVITY_TRADES_SUBSCRIPTION,
    RTDSClient,
    RTDSError,
    RTDSMessageError,
    backoff_delay,
    trade_from_message,
)

# Verbatim live RTDS frame (line breaks added only to satisfy the line-length
# rule; the JSON content is unmodified).
LIVE_FRAME = (
    '{"connection_id":"gVOKZI6JNWeIKEjigA==","payload":{"asset":"8677303472474444'
    '8744514889729172499268401134795843352779107239076687775715536","bio":"",'
    '"conditionId":"0x6c0529846fd87559429fb702416d4da1c9566cfd15317f9faaf62c4364'
    '83a71a","eventSlug":"btc-updown-5m-1785027300","icon":"https://polymarket-'
    'upload.s3.us-east-2.amazonaws.com/BTC+fullsize.png","name":"riceapplepie",'
    '"outcome":"Up","outcomeIndex":0,"price":0.45,"profileImage":"",'
    '"proxyWallet":"0x1015bb260154F51E5F432cB0a3227C1619FCBac8",'
    '"pseudonym":"Agile-Analogy","side":"BUY","size":4.33,'
    '"slug":"btc-updown-5m-1785027300","timestamp":1785027308,'
    '"title":"Bitcoin Up or Down - July 25, 8:55PM-9:00PM ET",'
    '"transactionHash":"0x0a8e0353b2432ea36cb7ea163f212bddd8f1d1fe9078ef91c3d262'
    'b3f87c4874"},"timestamp":1785027308241,"topic":"activity","type":"trades"}'
)

LIVE_CONDITION_ID = "0x6c0529846fd87559429fb702416d4da1c9566cfd15317f9faaf62c436483a71a"


def make_frame(**payload_overrides) -> str:
    """Build a synthetic activity/trades frame off the pinned live template."""
    envelope = json.loads(LIVE_FRAME)
    envelope["payload"].update(payload_overrides)
    return json.dumps(envelope)


class FakeSocket:
    """Scripted `RTDSSocket`: replays frames, then raises the scripted error.

    List items that are exception instances are raised instead of returned,
    which is how tests script timeouts and mid-stream disconnects.
    """

    def __init__(self, script: list) -> None:
        """Store the frame/exception script to replay on successive `recv`s."""
        self._script = list(script)
        self.sent: list[str] = []
        self.closed = False

    def send(self, payload: str) -> None:
        """Record the subscription frame the client sends on connect."""
        self.sent.append(payload)

    def recv(self) -> str:
        """Return the next scripted frame, or raise the next scripted error."""
        if not self._script:
            raise ConnectionError("peer closed the connection")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        """Mark the socket closed so tests can assert clean teardown."""
        self.closed = True


def scripted_factory(sockets: list) -> tuple:
    """Return a (factory, opened) pair replaying `sockets` one connect at a time.

    Exception items in `sockets` are raised from the factory itself, simulating
    a failed connect. Once the list is exhausted, every further connect fails.
    """
    remaining = list(sockets)
    opened: list[FakeSocket] = []

    def factory() -> FakeSocket:
        if not remaining:
            raise ConnectionError("endpoint unreachable")
        item = remaining.pop(0)
        if isinstance(item, BaseException):
            raise item
        opened.append(item)
        return item

    return factory, opened


def build_client(sockets: list, **kwargs):
    """Build an RTDSClient over a scripted factory with instant, seeded backoff."""
    factory, opened = scripted_factory(sockets)
    sleeps: list[float] = []
    ticks = iter(range(10_000))
    kwargs.setdefault("backoff_jitter", 0.0)
    client = RTDSClient(
        socket_factory=factory,
        rng=random.Random(0),
        sleep=sleeps.append,
        clock=lambda: float(next(ticks)),
        **kwargs,
    )
    return client, opened, sleeps


# ---------------- Normalization ----------------


def test_live_frame_normalizes_to_raw_trade():
    """The pinned live frame maps onto RawTrade with wallet/ts/price/size."""
    trade = trade_from_message(LIVE_FRAME)

    assert isinstance(trade, RawTrade)
    assert trade.wallet == "0x1015bb260154F51E5F432cB0a3227C1619FCBac8"
    assert trade.timestamp == 1785027308  # payload seconds, not envelope ms
    assert isinstance(trade.price, float) and trade.price == pytest.approx(0.45)
    assert isinstance(trade.size, float) and trade.size == pytest.approx(4.33)
    assert trade.side == "BUY"
    assert trade.condition_id == LIVE_CONDITION_ID
    assert trade.asset_id.startswith("867730347247444487445148897291724992684")
    assert trade.transaction_hash.startswith("0x0a8e0353b2432ea36cb7ea163f212bdd")


def test_string_price_and_size_coerce_to_float():
    """RTDS occasionally ships numerics as strings; RawTrade coercion holds."""
    trade = trade_from_message(make_frame(price="0.61", size="12.5"))
    assert trade.price == pytest.approx(0.61)
    assert trade.size == pytest.approx(12.5)


def test_empty_keepalive_frame_yields_none():
    """The server's empty text frames are keepalives, not trades."""
    assert trade_from_message("") is None
    assert trade_from_message("   ") is None


def test_other_topics_are_ignored():
    """Frames from another topic/type are skipped without raising."""
    envelope = json.loads(LIVE_FRAME)
    envelope["topic"] = "comments"
    assert trade_from_message(json.dumps(envelope)) is None


def test_envelope_timestamp_fallback_converts_ms_to_seconds():
    """A payload with no clock falls back to the envelope's milliseconds."""
    envelope = json.loads(LIVE_FRAME)
    del envelope["payload"]["timestamp"]
    trade = trade_from_message(json.dumps(envelope))
    assert trade.timestamp == 1785027308


@pytest.mark.parametrize(
    "bad",
    [
        "{not json",
        "[1, 2, 3]",
        '{"topic":"activity","type":"trades","payload":"nope"}',
        '{"topic":"activity","type":"trades","payload":{"price":"abc"}}',
    ],
)
def test_malformed_frames_raise(bad):
    """Undecodable frames and uncoercible payloads surface as RTDSMessageError."""
    with pytest.raises(RTDSMessageError):
        trade_from_message(bad)


# ---------------- Backoff ----------------


def test_backoff_doubles_and_caps():
    """Delays follow base * 2**(n-1) until the cap, jitter disabled."""
    rng = random.Random(0)
    delays = [backoff_delay(n, rng, base=1.0, cap=8.0, jitter=0.0) for n in range(1, 7)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_backoff_jitter_stays_within_band_and_is_seeded():
    """Jitter inflates by at most `jitter` and is reproducible from the seed."""
    a = [backoff_delay(n, random.Random(7), base=1.0, jitter=0.25) for n in (1, 2)]
    b = [backoff_delay(n, random.Random(7), base=1.0, jitter=0.25) for n in (1, 2)]
    assert a == b
    assert 1.0 <= a[0] <= 1.25
    assert 2.0 <= a[1] <= 2.5


def test_backoff_rejects_zero_attempt():
    """Attempt indices are 1-based; 0 is a programming error."""
    with pytest.raises(ValueError):
        backoff_delay(0, random.Random(0))


# ---------------- Client behaviour ----------------


def test_client_subscribes_to_activity_trades():
    """The client sends the activity/trades subscription on connect."""
    sock = FakeSocket([LIVE_FRAME])
    client, opened, _ = build_client([sock], max_reconnects=0)

    trades = list(client.stream(max_trades=1))

    assert len(trades) == 1
    assert json.loads(opened[0].sent[0]) == ACTIVITY_TRADES_SUBSCRIPTION


def test_malformed_message_is_skipped_and_stream_continues(caplog):
    """A bad frame is logged and dropped; the following good frame still lands."""
    sock = FakeSocket(["{not json", make_frame(price=0.7)])
    client, _, _ = build_client([sock], max_reconnects=0)

    with caplog.at_level(logging.WARNING, logger="src.data.rtds"):
        trades = list(client.stream(max_trades=1))

    assert [t.price for t in trades] == [pytest.approx(0.7)]
    assert any("malformed" in r.message for r in caplog.records)


def test_reconnect_preserves_delivered_trades_and_backs_off():
    """A drop after 3 frames reconnects, loses nothing, and backs off once."""
    first = FakeSocket(
        [
            make_frame(price=0.1),
            make_frame(price=0.2),
            make_frame(price=0.3),
            ConnectionError("dropped"),
        ]
    )
    second = FakeSocket([make_frame(price=0.4), make_frame(price=0.5)])
    client, opened, sleeps = build_client([first, second], backoff_base=1.0)

    prices = [t.price for t in client.stream(max_trades=5)]

    assert prices == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert len(opened) == 2
    assert first.closed  # the dead socket is torn down before reconnecting
    # Frames delivered on `first` reset the failure counter, so the single drop
    # backs off from the base delay rather than from a stale capped one.
    assert sleeps == [1.0]


def test_consecutive_connect_failures_double_the_delay():
    """Two failed connects in a row produce a 1s then 2s backoff."""
    sock = FakeSocket([LIVE_FRAME])
    client, _, sleeps = build_client(
        [ConnectionError("down"), ConnectionError("still down"), sock],
        backoff_base=1.0,
    )

    trades = list(client.stream(max_trades=1))

    assert len(trades) == 1
    assert sleeps == [1.0, 2.0]


def test_max_reconnects_gives_up_with_rtds_error():
    """Past `max_reconnects` consecutive failures the client raises."""
    client, _, _ = build_client([], max_reconnects=2)

    with pytest.raises(RTDSError, match="consecutive attempts"):
        list(client.stream(max_trades=1))


def test_condition_id_filter_keeps_only_matching_markets():
    """`condition_ids` drops trades from every other market, case-insensitively."""
    other = "0xbbb000000000000000000000000000000000000000000000000000000000bb01"
    sock = FakeSocket(
        [
            make_frame(conditionId=other, price=0.1),
            make_frame(price=0.2),
            make_frame(conditionId=other, price=0.3),
            make_frame(price=0.4),
        ]
    )
    client, _, _ = build_client(
        [sock], condition_ids=[LIVE_CONDITION_ID.upper()], max_reconnects=0
    )

    trades = list(client.stream(max_trades=2))

    assert [t.price for t in trades] == pytest.approx([0.2, 0.4])
    assert {t.condition_id for t in trades} == {LIVE_CONDITION_ID}


def test_idle_socket_warns_once_per_stale_window(caplog):
    """Recv timeouts are not failures, but a long silence logs a warning."""
    sock = FakeSocket(
        [TimeoutError("idle"), TimeoutError("idle"), TimeoutError("idle"), LIVE_FRAME]
    )
    # clock() advances 1 per call; stale_after=2 makes the third idle tick stale.
    client, opened, sleeps = build_client([sock], stale_after=2.0, max_reconnects=0)

    with caplog.at_level(logging.WARNING, logger="src.data.rtds"):
        trades = list(client.stream(max_trades=1))

    assert len(trades) == 1
    assert len(opened) == 1  # idling never triggered a reconnect
    assert sleeps == []
    assert sum("no RTDS frame" in r.message for r in caplog.records) == 1
