"""
Black box tests for the stdio MCP server.

Talks raw JSON-RPC over the process pipes so we can assert that every single
byte on stdout is a valid JSON-RPC 2.0 message.

Run:  pytest -q task1_mcp_server
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).with_name("server.py")


class StdioClient:
    def __init__(self, env_extra=None):
        env = {**os.environ, **(env_extra or {})}
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self.next_id = 1
        self.raw_stdout: list[str] = []

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        rid = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            line = self.proc.stdout.readline()
            assert line, "server closed stdout"
            self.raw_stdout.append(line)
            msg = json.loads(line)  # fails loudly on any non JSON output
            if msg.get("id") == rid:
                return msg

    def initialize(self):
        resp = self.request(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def close(self):
        self.proc.stdin.close()
        rest, err = self.proc.communicate(timeout=10)
        self.raw_stdout.extend(l + "\n" for l in rest.splitlines() if l)
        return err


@pytest.fixture
def client():
    c = StdioClient(env_extra={"MCP_STDOUT_CANARY": "1", "LOG_LEVEL": "DEBUG"})
    c.initialize()
    yield c
    c.close()


def call(c, name, arguments):
    return c.request("tools/call", {"name": name, "arguments": arguments})


def test_tools_list_has_strict_schemas(client):
    tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    schema = tools["trigger_refund"]["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}


def test_get_customer_ok(client):
    r = call(client, "get_customer_record", {"customer_id": "CUST-00042"})
    assert r["result"]["isError"] is False
    assert r["result"]["structuredContent"]["name"] == "Alan Turing"


def test_unknown_customer_is_tool_error_not_protocol_error(client):
    r = call(client, "get_customer_record", {"customer_id": "CUST-99999"})
    assert "error" not in r
    assert r["result"]["isError"] is True


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"customer_id": "CUST-1234"},
        {"customer_id": "cust-12345"},
        {"customer_id": "CUST-12345 extra"},
        {"customer_id": 12345},
        {"customer_id": "CUST-12345", "unexpected": True},
    ],
)
def test_get_customer_invalid(client, args):
    r = call(client, "get_customer_record", args)
    assert r["error"]["code"] == -32602
    assert r["error"]["data"]["errors"]


def test_refund_ok(client):
    r = call(client, "trigger_refund", {"customer_id": "CUST-00001", "amount": 19.99, "reason": "Damaged on arrival"})
    assert r["result"]["isError"] is False
    assert r["result"]["structuredContent"]["status"] == "queued"


@pytest.mark.parametrize(
    "args",
    [
        {"customer_id": "CUST-00001", "amount": 0, "reason": "Damaged on arrival"},
        {"customer_id": "CUST-00001", "amount": -5, "reason": "Damaged on arrival"},
        {"customer_id": "CUST-00001", "amount": "10", "reason": "Damaged on arrival"},
        {"customer_id": "CUST-00001", "amount": 10.123, "reason": "Damaged on arrival"},
        {"customer_id": "CUST-00001", "amount": True, "reason": "Damaged on arrival"},
        {"customer_id": "CUST-00001", "amount": 10, "reason": "too short"},
        {"customer_id": "CUST-00001", "amount": 10, "reason": "      short     "},
        {"customer_id": "CUST-00001", "amount": 10},
        {"customer_id": "BAD", "amount": 10, "reason": "Damaged on arrival"},
    ],
)
def test_refund_invalid(client, args):
    r = call(client, "trigger_refund", args)
    assert r["error"]["code"] == -32602, r


def test_refund_over_balance_is_tool_error(client):
    r = call(client, "trigger_refund", {"customer_id": "CUST-00042", "amount": 5000, "reason": "Way too much money"})
    assert r["result"]["isError"] is True


def test_unknown_tool(client):
    r = call(client, "drop_database", {})
    assert r["error"]["code"] == -32602


def test_unknown_method(client):
    # The SDK rejects methods outside the MCP request union while parsing,
    # which surfaces as -32602. Either code is a correct JSON-RPC rejection.
    r = client.request("does/not/exist")
    assert r["error"]["code"] in (-32601, -32602)


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr():
    c = StdioClient(env_extra={"MCP_STDOUT_CANARY": "1", "LOG_LEVEL": "DEBUG"})
    c.initialize()
    call(c, "get_customer_record", {"customer_id": "BAD"})
    call(c, "trigger_refund", {"customer_id": "CUST-00001", "amount": 1, "reason": "Testing refunds"})
    stderr = c.close()

    assert c.raw_stdout, "expected protocol output"
    for line in c.raw_stdout:
        msg = json.loads(line)
        assert msg["jsonrpc"] == "2.0"
        assert ("result" in msg) or ("error" in msg) or ("method" in msg)
    assert "canary: stray print" in stderr
    assert "refund queued" in stderr
