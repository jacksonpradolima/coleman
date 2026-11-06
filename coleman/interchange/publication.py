"""Explicit optional publication through the existing local immutable writer."""

from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.interchange.codec import decode_request, decode_response
from coleman.interchange.contracts import InterchangeRequest, InterchangeResponse


class InterchangeConflictError(ValueError):
    """Signal changed complete bytes for one immutable logical request/attempt.

    Notes
    -----
    The previously published artifact remains unchanged. This is a local
    consistency conflict, without a global deduplication/authenticity claim.
    """


def _publish(path: Path, content: str, writer: ArtifactWriter) -> None:
    """Translate the existing writer's immutable-content conflict explicitly."""
    try:
        writer.write_immutable(path, content)
    except ValueError as error:
        raise InterchangeConflictError("Conflicting local interchange request/attempt") from error


def _directory(request: InterchangeRequest, writer: ArtifactWriter) -> Path:
    """Use hashed host logical-attempt identity rather than caller-controlled paths."""
    return writer.run_root / "interchange" / request.attempt_identity.removeprefix("sha256:")


def publish_request(request: InterchangeRequest, writer: ArtifactWriter) -> Path:
    """Publish one complete bounded request or reject a conflicting logical attempt.

    Parameters
    ----------
    request : InterchangeRequest
        Revalidated explicit call. Same logical task/operation/exchange/attempt
        maps to one path even if input, implementation or clocks change.
    writer : ArtifactWriter
        Caller-created trusted local writer; no writer is constructed implicitly.

    Returns
    -------
    pathlib.Path
        Complete ``request.json``. Same canonical bytes are idempotent.

    Raises
    ------
    InterchangeConflictError
        Different complete bytes already exist for this logical attempt.
    ValueError, TypeError
        Invalid/oversized request.
    OSError
        Existing writer's local publication errors propagate.

    Notes
    -----
    Creates only the explicit writer's request directories/files. Reuses its
    fsynced temporary-file/hard-link publication and trusted-parent/platform
    limits. A request alone is never a completed output or execution receipt.
    """
    request = decode_request(InterchangeRequest.model_validate(request).to_json())
    path = _directory(request, writer) / "request.json"
    _publish(path, request.to_json(), writer)
    return path


def publish_response(request: InterchangeRequest, response: InterchangeResponse, writer: ArtifactWriter) -> Path:
    """Publish an atomic complete reply after retaining its exact producing call.

    Parameters
    ----------
    request : InterchangeRequest
        Caller-retained actual call, not merely the reply's echoed logical ID.
    response : InterchangeResponse
        Complete typed reply. Its mathematical input/attempt/implementation/state
        and production clocks must match this exact request.
    writer : ArtifactWriter
        Explicit trusted local output root; no transport/store is mandatory.

    Returns
    -------
    pathlib.Path
        Complete immutable ``response.json``. The single complete file is the
        output publication boundary; temporary files and request-only attempts
        are never accepted as completed output.

    Raises
    ------
    InterchangeConflictError
        Different complete bytes exist at either immutable request/reply path.
        Identical retry content succeeds.
    ValueError, TypeError
        Invalid/oversized reply or producing-request mismatch.
    OSError
        Existing writer's write/link/fsync failures propagate.

    Notes
    -----
    Publishes the request first, so changed content for a logical attempt cannot
    silently create a second reply under another content-addressed filename.
    Consumer receipt/freshness remains ``accept_response`` at actual delivery;
    a producer cannot fabricate that receipt. No algorithm, external effect,
    pickle, clock or multi-file transaction is invoked. A crash can leave only
    a request; complete reply bytes embed and bind their full original request.
    """
    request = decode_request(InterchangeRequest.model_validate(request).to_json())
    response = decode_response(InterchangeResponse.model_validate(response).to_json())
    if response.request.to_json() != request.to_json():
        raise ValueError("Published reply differs from the retained producing request")
    publish_request(request, writer)
    path = _directory(request, writer) / "response.json"
    _publish(path, response.to_json(), writer)
    return path
