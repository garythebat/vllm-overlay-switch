# Testing

What the unit tests cover and what was measured on hardware. The summary is in the
[README](../README.md#status-and-results).

## Unit tests

```sh
python3 -m unittest discover -s tests -v
```

159 tests, none of which needs a GPU.

| Area | Tests | How |
|---|---|---|
| Switch sequence: ordering, rollback when a rank fails, forced recovery, access control | `tests/test_logic.py` | mock engine |
| HTTP routes | `tests/test_api.py` | FastAPI's test client and a mock engine |
| safetensors reader | `tests/test_stfile.py` | hand-built files |
| Worker extension: column selection per rank, in-place overwrite, read-back, padding left untouched, rejection of unsupported input | `tests/test_worker.py` | real torch tensors (torch 2.13.0 CPU, the version in the engine image), real files, three simulated ranks, mocks of the engine modules |
| Gate: admission, queueing, failed or interrupted switch, allowed and rejected routes, API key under the prefix | `tests/test_gate.py`, `tests/test_api.py` | mock engine and a mock server that records which variant was active while each request ran; also through FastAPI as the plugin installs it |
| Variant selection by model name; `reasoning_effort` translation | `tests/test_gate.py`, `tests/test_effort.py`, `tests/test_api.py` | the values the engine rejects were taken from a running server: `minimal`, `medium` and unknown words in either position, integers at the top level, integers outside 1 to 100 |
| `tools/acceptance.py` | `tests/test_acceptance.py` | mock server with known behaviour: pass, each failure verdict, drift between samples, an older plugin without the digest, inconclusive |
| `tools/gate_acceptance.py` | `tests/test_gate_acceptance.py` | mock server that can fail in each way the tool is meant to detect |
| Packaging | `tests/test_package.py` | entry point name, class names referenced in the documentation |

Mutation check: 54 faults introduced on purpose in the gate, the effort translation and
the acceptance tool were each caught by a test.

Other checks:

- **Fit with the engine.** The code was reviewed against the engine source at the pinned
  commits. The references are in [design.md](design.md).
- **Installer** (`mod/run.sh`). Dry-run against a directory laid out like the image's
  package directory. It also runs inside the image on every node at each launch. The
  image registers no plugins other than vLLM's own two.

## Hardware results

Measured on the test setup: three nodes of the DGX Spark family (two NVIDIA DGX Spark,
one Gigabyte AI TOP ATOM), tensor parallel 3, DeepSeek V4.1 Flash 1017 with the Keys
overlay, on the engine build listed under
[Compatibility](../README.md#compatibility). Unless noted, the numbers are from the lean
configuration (262,144-token context, `--kv-cache-memory-bytes 500m`).

### Manual switch

| Check | Result |
|---|---|
| Timing | Preparation 3.9 s. A switch 1.4 to 2.0 s end to end, of which the pause before the copy is 0.03 s. After every switch each node read all its tensors back as the requested variant |
| Switching back restores the model | Exact. The digest of all state derived from the switched weights, read on a freshly started server before anything was prepared, was the same on every rank after switching to the overlay and back. Observed on two separate launches |
| The model uses the switched weights | Six independent passes per variant: 5 of 8 prompts distinguished the two variants, with every margin of one variant on one side and every margin of the other on the other |
| The switched state is the overlay model | The overlay was also loaded as a model of its own and sampled with `--read-only`. Wherever base and switched differed, it was on the switched side with nearly the same medians, and at none of 34 positions did it differ from the switched state |
| Memory | Six consecutive switches left free memory unchanged |

### Two endpoints (the gate)

| Check | Result |
|---|---|
| Routing | `tools/gate_acceptance.py` passed. All 156 responses carried the header of the requested variant. Reference: six passes per endpoint, in which 3 of 8 prompts distinguished the variants by at least 3 logits. Then 60 mixed requests: 36 alternating, with a switch before every request, and two concurrent bursts of 12. Half of the overlay requests were made by model name on the standard path and half under the prefix. All 60 reached a reference position and all 60 matched their own variant |
| Cost of a change of variant | A response took 0.69 s on average without a switch and 2.22 s when a switch was needed first (36 responses, 2.02 to 2.66 s). The gate counted 36 switches for the 36 alternating requests. Concurrent requests for both variants are strictly serialized: in the two bursts of 12, a response took 0.6 to 26.2 s, 11.2 s on average |
| Prefix cache | vLLM's own prefix-cache counters around one 604-token prompt sent with one client salt: repeated on base it reused 256 tokens, the overlay's first request reused 0, the overlay again 256, the overlay by model name 256, and base after two switches 256. A switch cleared nothing |
| `reasoning_effort` translation | 42 of 42 requests returned 200: seven values, at the top level and in `chat_template_kwargs`, by either model name and under the prefix. The response header, the log line per distinct translation and the fields in the model list were as documented |
| Client disconnecting mid-stream | Tested by hand with a small probe, six rounds each way: after the client left, a request for the other variant was served in 2.1 to 3.0 s, by its own variant every time (12 of 12) |
| State afterwards | 60 switches into the launch, every tensor read back as base on every rank, and the digests of all derived state were those from startup |
| Memory during a switch | Free memory sampled five times a second on each node during six switches: it dropped by at most 0.07 GiB and recovered |

### Large configuration

The same server with a 1,048,576-token context and `--kv-cache-memory-bytes 3g`, which
leaves each node a few GiB free instead of six to eight.

| Check | Result |
|---|---|
| Acceptance test | Passed: 156 responses each with the header of its own variant, 60 mixed responses all matching their own variant, separate prefix caches, 42 of 42 `reasoning_effort` requests returned 200 |
| Timing | 0.61 s per response on average, 2.24 s when a switch was needed first |
| State afterwards | Every tensor read back as base, with the startup digests, 60 switches into the launch |
| Memory | The first two switches reduced free memory by 0.03 GiB. Over four minutes of switches and bursts the lowest reading on the tightest node was 1.89 GiB, against 1.7 GiB after its first requests and before any switch |

### Repeated with an `abort` pause

The results above were measured while the gate paused the engine in `wait` mode before
a switch. It now pauses in `abort` mode (see
[Limitations](../README.md#limitations)). The two-endpoint tests were repeated on that
version, in the large configuration and with `--max-num-seqs 4` (the earlier runs used
2).

| Check | Result |
|---|---|
| Acceptance test | Passed: 156 responses each with the header of its own variant, 60 mixed responses all matching their own variant, separate prefix caches (256, 0, 256, 256 and 256 tokens reused, as before), 42 of 42 `reasoning_effort` requests returned 200 |
| Timing | 0.66 s per response on average, 2.3 s when a switch was needed first. The gate counted 36 switches for the 36 alternating requests |
| Client disconnecting mid-stream | Six rounds each way: after the client left, a request for the other variant was served in 2.4 to 2.8 s, by its own variant every time (12 of 12) |
| Four concurrent requests for one variant | vLLM reported 4 requests running and 0 waiting. On base: 1,348 tokens in 14.9 s, 90 tokens/s in total. On the overlay, with a switch first: 1,272 tokens in 16.9 s, 75 tokens/s in total. Every response carried the header of the requested variant |
| State afterwards | 60 switches into the launch, every tensor read back as base on every rank, and the digests of all derived state were those from startup, which were also those of the earlier launches |
| Memory | The first two switches reduced free memory by 0.03 to 0.04 GiB. The lowest reading on the tightest node was 1.29 GiB, during the first switch, against 1.32 GiB after its first requests. Over five minutes of switches, bursts and concurrent requests it stayed between 1.63 and 1.89 GiB |

## Not yet tested

- The manual switch under load (requests in flight, several clients).
- The API-key check under the prefix on a real server (unit-tested against a mock that
  replicates vLLM's rule).
- Tensor-parallel sizes other than 3.
- Locating the checkpoint by its served name (the test setup set
  `OVERLAY_SWITCH_BASE_DIR`).
- With two endpoints: a long generation on one variant while the other is requested (the
  wait is by design, its length was not measured), long cached conversations on both
  variants at once (the test prompts are short), a client that disconnects while queued,
  and a server started with `--root-path`.
