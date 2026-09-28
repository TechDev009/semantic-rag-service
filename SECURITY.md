# Security Policy

## Scope

This repository demonstrates a grounded retrieval service. Report vulnerabilities
in this service, its API validation, storage isolation, provider adapters, or
prompt/citation handling.

## Reporting

Please do not open a public issue for a suspected vulnerability. Once this
repository is published, use its GitHub **Security** tab and **Report a
vulnerability** form. If private reporting is not enabled yet, contact the
repository maintainer through their GitHub profile before sharing details. Include
a description, reproduction steps, impact, and proposed mitigation. Remove
credentials and personal data from reports.

## Deployment guidance

The demo intentionally has no authentication or authorization. Before production:

- put the API behind authenticated, authorized service-to-service access;
- enforce collection/tenant ACLs at every route and in retrieval;
- use TLS, secret management, provider egress controls, rate limits, and request
  timeouts;
- protect PostgreSQL and backups, and redact document text, prompts, and API keys
  from logs;
- review provider data retention and disable training/retention where available;
- keep embedding model/dimension migrations explicit and reproducible;
- test prompt injection, data exfiltration, malformed model output, citation
  fabrication, and cross-tenant retrieval in staging.

The service treats retrieved documents as untrusted data and rejects generated
citations that are not retrieved chunk IDs, but no model-based defense is
perfect. Operators remain responsible for defense in depth.
