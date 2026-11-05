"""
Task 1: Customer support MCP server over stdio.

Tools
  get_customer_record(customer_id)
  trigger_refund(customer_id, amount, reason)

Design notes
  * stdout belongs to the JSON-RPC transport and nothing else. At startup we
    duplicate fd 1 into a private fd for the transport, then point fd 1 and
    sys.stdout at stderr. Any stray print() or C level write ends up on
    stderr instead of corrupting the protocol stream.
  * Input is validated with strict Pydantic models. Bad input is a protocol
    error, so we raise JSON-RPC -32602 (Invalid params) with structured detail.
  * Business failures (unknown customer, refund over balance) are tool
    execution errors, so per the MCP spec they come back as a normal result
    with isError=true and the model can read and react to them.
  * Anything unexpected becomes -32603 with a generic message. Details go to
    the stderr log only.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from decimal import Decimal
from typing import Any

# ---------------------------------------------------------------------------
# 1. Lock down stdout BEFORE importing anything that might write to it.
# ---------------------------------------------------------------------------
_PROTOCOL_FD = os.dup(sys.stdout.fileno())      # private copy of real stdout
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # fd 1 now goes to stderr
sys.stdout = sys.stderr                           # print() now goes to stderr

logging.basicConfig(
    stream=sys.stderr,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("customer-mcp")

import anyio  # noqa: E402
import mcp.types as types  # noqa: E402
from mcp.server.lowlevel import NotificationOptions, Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402
from pydantic import (  # noqa: E402
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

# ---------------------------------------------------------------------------
# 2. Schemas
# ---------------------------------------------------------------------------
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"


class _Strict(BaseModel):
    # forbid unknown fields, no silent type coercion ("12" is not a number)
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class GetCustomerRecordInput(_Strict):
    customer_id: str = Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer id in the form CUST-XXXXX, five digits",
        examples=["CUST-00042"],
    )


class TriggerRefundInput(_Strict):
    customer_id: str = Field(pattern=CUSTOMER_ID_PATTERN, examples=["CUST-00042"])
    amount: float = Field(
        gt=0,
        le=1_000_000,
        allow_inf_nan=False,
        description="Positive refund amount, max two decimal places",
    )
    reason: str = Field(min_length=10, max_length=500)

    @field_validator("amount")
    @classmethod
    def _two_decimals(cls, v: float) -> float:
        if Decimal(str(v)).as_tuple().exponent < -2:
            raise ValueError("amount must have at most 2 decimal places")
        return v


TOOLS: dict[str, tuple[type[_Strict], str]] = {
    "get_customer_record": (GetCustomerRecordInput, "Look up a customer record by id."),
    "trigger_refund": (TriggerRefundInput, "Issue a refund to a customer."),
}

# ---------------------------------------------------------------------------
# 3. Fake data layer
# ---------------------------------------------------------------------------
CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Ada Lovelace", "tier": "gold", "balance": 250.00},
    "CUST-00042": {"name": "Alan Turing", "tier": "silver", "balance": 80.50},
}
REFUNDS: list[dict[str, Any]] = []

# ---------------------------------------------------------------------------
# 4. Server
# ---------------------------------------------------------------------------
server: Server = Server("customer-support-mcp", version="1.0.0")


def _rpc_error(code: int, message: str, data: Any = None) -> McpError:
    return McpError(types.ErrorData(code=code, message=message, data=data))


def _validation_detail(err: ValidationError) -> list[dict[str, Any]]:
    # Keep it useful but never echo raw input back
    return [
        {"field": ".".join(str(p) for p in e["loc"]) or "<root>", "error": e["msg"], "type": e["type"]}
        for e in err.errors(include_url=False, include_input=False)
    ]


def _text(payload: Any, is_error: bool = False) -> types.CallToolResult:
    import json

    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload if isinstance(payload, dict) else None,
        isError=is_error,
    )


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=name, description=desc, inputSchema=model.model_json_schema())
        for name, (model, desc) in TOOLS.items()
    ]


async def _get_customer_record(args: GetCustomerRecordInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _text({"error": "customer_not_found", "customer_id": args.customer_id}, is_error=True)
    return _text({"customer_id": args.customer_id, **record})


async def _trigger_refund(args: TriggerRefundInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _text({"error": "customer_not_found", "customer_id": args.customer_id}, is_error=True)
    if args.amount > record["balance"]:
        return _text(
            {"error": "amount_exceeds_balance", "balance": record["balance"]}, is_error=True
        )
    record["balance"] = round(record["balance"] - args.amount, 2)
    refund = {
        "refund_id": f"RF-{len(REFUNDS) + 1:06d}",
        "customer_id": args.customer_id,
        "amount": args.amount,
        "reason": args.reason,
        "status": "queued",
    }
    REFUNDS.append(refund)
    log.info("refund queued id=%s customer=%s amount=%.2f", refund["refund_id"], args.customer_id, args.amount)
    return _text(refund)


HANDLERS = {"get_customer_record": _get_customer_record, "trigger_refund": _trigger_refund}


async def handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    """Registered directly so McpError reaches the wire as a real JSON-RPC error.

    The SDK's @server.call_tool() decorator wraps every exception into an
    isError result, which would hide invalid params behind a 200 style result.
    """
    name = req.params.name
    raw_args = req.params.arguments
    log.debug("tools/call name=%s", name)

    if name not in TOOLS:
        raise _rpc_error(types.INVALID_PARAMS, f"Unknown tool: {name}")
    if raw_args is not None and not isinstance(raw_args, dict):
        raise _rpc_error(types.INVALID_PARAMS, "arguments must be an object")

    model, _ = TOOLS[name]
    try:
        args = model.model_validate(raw_args or {})
    except ValidationError as err:
        log.info("validation failed tool=%s errors=%d", name, err.error_count())
        raise _rpc_error(types.INVALID_PARAMS, "Invalid params", {"tool": name, "errors": _validation_detail(err)})

    try:
        result = await HANDLERS[name](args)
    except McpError:
        raise
    except Exception:  # never leak internals to the client
        log.exception("tool %s crashed", name)
        raise _rpc_error(types.INTERNAL_ERROR, "Internal error")
    return types.ServerResult(result)


server.request_handlers[types.CallToolRequest] = handle_call_tool


async def main() -> None:
    if os.environ.get("MCP_STDOUT_CANARY") == "1":
        # Proves the guard: this would corrupt the stream without it
        print("canary: stray print from server code")

    protocol_out = anyio.wrap_file(io.TextIOWrapper(os.fdopen(_PROTOCOL_FD, "wb"), encoding="utf-8", line_buffering=True))
    protocol_in = anyio.wrap_file(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))

    log.info("starting customer-support-mcp on stdio")
    async with stdio_server(stdin=protocol_in, stdout=protocol_out) as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(notification_options=NotificationOptions()),
        )


if __name__ == "__main__":
    anyio.run(main)
