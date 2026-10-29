# External decision API

SCS calls a configured HTTP service to select one of its seven query playbooks.
The service owns model installation, inference, warmup, and memory use. SCS owns
its playbook choices and validates every answer. No decision model runs in SCS.

Set `decision_model` to the service's exact model identity and configure
`decision_base_url` in `~/.scs/config.toml`:

```toml
decision_model = "your-provider/decision-model@revision"
decision_base_url = "http://127.0.0.1:10000/v1"
```

Omit `decision_model` to disable external decisions. The existing `disabled`
marker also disables them. SCS accepts no built-in model aliases and pins no
decision checkpoint. `routing.provider` reports `external` for service decisions
and `deterministic` for disabled routing or classifier fallback. The separate
`routing.model` field carries the decision's model identity.

SCS allows loopback hosts by default. An exact non-loopback hostname must be
listed in `decision_trusted_hosts`. Requests carry no authorization header.
Only use a host you trust with query goals and anchors.

## Model readiness

SCS URL-encodes the configured identity in the `model` query parameter:

```text
GET {decision_base_url}/decisions/ready?model=your-provider%2Fdecision-model%40revision
```

HTTP 200 must return exactly:

```json
{
  "status": "ok",
  "model": "your-provider/decision-model@revision"
}
```

The response must confirm that the requested model is available and warmed.
Return a non-200 status if it is unavailable or unsupported. A generic health
response or a different model identity cannot satisfy this check. Services can
retain an unqualified readiness endpoint for older consumers; SCS always sends
the configured model.

SCS requires model readiness before publishing its socket and polls it while the
daemon runs. An unavailable or incompatible service prevents configured startup.
A later outage makes SCS unready. SCS waits for bounded recovery and shuts down
if the service does not return.

## Choice request

SCS sends `POST {decision_base_url}/decisions` with
`Content-Type: application/json`:

| Field | Meaning |
| --- | --- |
| `model` | The configured model identity. |
| `state` | A bounded object containing the goal and optional caller anchors. It contains no retrieved source, graph records, or embeddings. |
| `question.key` | `playbook`. |
| `question.instructions` | The routing instruction supplied by SCS. |
| `question.criteria` | A map of eligible SCS playbook labels to descriptions. |

The service must answer within SCS's per-mode classifier budget: 150 ms for
`fast`, 500 ms for `balanced`, and 1 second for `thorough`. SCS can also apply
a lower configured `decision_timeout_seconds` ceiling. A slow or malformed
in-flight answer degrades that query to deterministic discovery.

## Choice response

The HTTP 200 response must contain exactly these fields:

| Field | Meaning |
| --- | --- |
| `model` | The same configured identity. |
| `choice` | One of the supplied playbook labels. |
| `confidence` | A finite number from 0 to 1. |
| `probabilities` | One finite 0-to-1 probability for every supplied label, summing to 1 within 0.01. |

SCS rejects a different model identity, unknown label, missing or extra
probability, nonfinite number, or malformed response. Model output can select
only one fixed SCS playbook; it cannot name tools, files, or execution steps.

## Migration

Replace the former model-specific selector with the service's exact model ID.
Deploy support for model-qualified readiness before updating the SCS client and
configuration. Endpoints that ignore `model` and return an old compatibility
identity fail readiness for a different configured identity.
