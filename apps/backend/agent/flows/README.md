# Text runtime flows

Flows execute a route that the runtime has already selected. Keep this package as
an orchestration layer: call domain services, apply state deltas, finalize turns,
and bridge provider-specific adapters when a route needs one.

Boundary guidelines:

- Prompt shape belongs in specialist or skill rendering modules, not in flow
  string rewriting.
- App-owned lifecycle state belongs in services under the relevant domain
  package.
- Tool schemas and tool execution shims belong in `agent/tools`.
- Core runtime dispatch stays in `agent/runtime` until route-handler ownership is
  moved behind a flow registry.
