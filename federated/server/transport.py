"""In-process transport between the Flower server and simulated hospital clients.

``InProcessClientProxy`` implements Flower's ``ClientProxy`` interface (the same
extension point Flower's own simulation engine uses). Every instruction and
reply is encoded with Flower's protobuf serde into wire bytes and decoded on the
other side, i.e. exactly the messages Flower's legacy gRPC transport carries,
without opening sockets or requiring Ray. The byte counts are recorded so the
communication cost of each round is measurable.
"""
from __future__ import annotations

import threading
from typing import Callable
from dataclasses import dataclass, field

from flwr.client import Client
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
    serde,
)
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy


@dataclass
class TrafficLog:
    """Thread-safe byte counters and client-side failures per round."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    bytes_to_clients: dict[int, int] = field(default_factory=dict)
    bytes_from_clients: dict[int, int] = field(default_factory=dict)
    failures: dict[int, list[tuple[str, str]]] = field(default_factory=dict)
    on_event: Callable[[dict], None] | None = None

    def event(self, kind: str, cid: str, rnd: int) -> None:
        if self.on_event:
            self.on_event({"type": kind, "cid": cid, "round": rnd})

    def add(self, rnd: int, down: int = 0, up: int = 0) -> None:
        with self.lock:
            self.bytes_to_clients[rnd] = self.bytes_to_clients.get(rnd, 0) + down
            self.bytes_from_clients[rnd] = self.bytes_from_clients.get(rnd, 0) + up

    def fail(self, rnd: int, cid: str, error: str) -> None:
        with self.lock:
            self.failures.setdefault(rnd, []).append((cid, error))


class InProcessClientProxy(ClientProxy):
    def __init__(self, cid: str, client: Client, traffic: TrafficLog) -> None:
        super().__init__(cid)
        self._client = client
        self._traffic = traffic

    def _round(self, group_id: int | None) -> int:
        return int(group_id or 0)

    def get_properties(self, ins: GetPropertiesIns, timeout: float | None,
                       group_id: int | None) -> GetPropertiesRes:
        wire = serde.get_properties_ins_to_proto(ins).SerializeToString()
        ins2 = serde.get_properties_ins_from_proto(ServerMessage.GetPropertiesIns.FromString(wire))
        res = self._client.get_properties(ins2)
        out = serde.get_properties_res_to_proto(res).SerializeToString()
        return serde.get_properties_res_from_proto(ClientMessage.GetPropertiesRes.FromString(out))

    def get_parameters(self, ins: GetParametersIns, timeout: float | None,
                       group_id: int | None) -> GetParametersRes:
        wire = serde.get_parameters_ins_to_proto(ins).SerializeToString()
        ins2 = serde.get_parameters_ins_from_proto(ServerMessage.GetParametersIns.FromString(wire))
        res = self._client.get_parameters(ins2)
        out = serde.get_parameters_res_to_proto(res).SerializeToString()
        self._traffic.add(self._round(group_id), down=len(wire), up=len(out))
        return serde.get_parameters_res_from_proto(ClientMessage.GetParametersRes.FromString(out))

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        rnd = self._round(group_id)
        self._traffic.event("client_training", self.cid, rnd)
        wire = serde.fit_ins_to_proto(ins).SerializeToString()
        try:
            ins2 = serde.fit_ins_from_proto(ServerMessage.FitIns.FromString(wire))
            res = self._client.fit(ins2)
            out = serde.fit_res_to_proto(res).SerializeToString()
        except Exception as exc:  # recorded with the client id, then re-raised to Flower
            self._traffic.fail(rnd, self.cid, f"{type(exc).__name__}: {exc}")
            raise
        self._traffic.add(rnd, down=len(wire), up=len(out))
        self._traffic.event("client_submitted", self.cid, rnd)
        return serde.fit_res_from_proto(ClientMessage.FitRes.FromString(out))

    def evaluate(self, ins: EvaluateIns, timeout: float | None,
                 group_id: int | None) -> EvaluateRes:
        rnd = self._round(group_id)
        wire = serde.evaluate_ins_to_proto(ins).SerializeToString()
        ins2 = serde.evaluate_ins_from_proto(ServerMessage.EvaluateIns.FromString(wire))
        res = self._client.evaluate(ins2)
        out = serde.evaluate_res_to_proto(res).SerializeToString()
        self._traffic.add(rnd, down=len(wire), up=len(out))
        return serde.evaluate_res_from_proto(ClientMessage.EvaluateRes.FromString(out))

    def reconnect(self, ins: ReconnectIns, timeout: float | None,
                  group_id: int | None) -> DisconnectRes:
        return DisconnectRes(reason="in-process client; nothing to disconnect")
