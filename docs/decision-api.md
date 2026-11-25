# External decision API

This page defines the HTTP contract for configured external decision services.
Those services own model installation, inference, warmup, and memory use. SCS owns
its playbook choices and validates every answer. The contracts below apply when
an external decision provider is selected.

The bundled profile instead runs local Laya inference through SCS's shared native
worker. It uses no decision HTTP endpoint or external readiness dependency.
Complete verified local caches warm before daemon readiness. Cold downloads run
through first-use model work, including the durable indexing job.
See [provider configuration](configuration.md) for selection and
[runtime ownership](architecture.md#model-providers) for preparation and cleanup.

Set `decision_model` to the service's exact model identity and configure
`decision_base_url` in `~/.scs/config.toml`:

```toml
decision_model = "your-provider/decision-model@revision"
decision_base_url = "http://127.0.0.1:10000/v1"
```

In the external profile, omit `decision_model` to disable external decisions.
The existing `disabled` marker also disables them. External model identities are
never translated to bundled checkpoints. `routing.provider` reports `external`
for service decisions, `local` for bundled inference, and `deterministic` for
disabled routing or classifier fallback. The separate `routing.model` field
carries the decision's actual model identity.

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
`fast`, 1.5 s for `balanced`, and 2.5 s for `thorough`. SCS can also apply a
lower configured `decision_timeout_seconds` ceiling (default 1.5 s,
configurable up to 5.0 s). A slow or malformed in-flight answer degrades that
query to deterministic discovery.

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

Before upgrading SCS, deploy model-qualified readiness at the decision service.
When upgrading an existing client, install the new SCS version before writing
the new model ID. Keep the old selector or use `disabled` during installation.
The installer uses the old client to stop its daemon, and that client rejects
new configuration values. After installation, replace the selector with the
service's exact model identity and start or restart SCS.

Endpoints that ignore `model` and return an old compatibility identity fail
readiness for a different configured identity.
