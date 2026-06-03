# Security Policy

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Instead, please report them privately via:

- **Email**: security@nexus-llm.dev
- **GitHub Security Advisory**: Use the [Security Advisories](https://github.com/nexus-llm/Nexus-LLM/security/advisories) page

We aim to acknowledge vulnerability reports within 24 hours and provide an initial assessment within 72 hours.

## Supported Versions

| Version | Supported          | Status       |
| ------- | ------------------ | ------------ |
| 2.1.x   | :white_check_mark: | Active       |
| 2.0.x   | :white_check_mark: | Maintenance  |
| 1.5.x   | :warning:          | Critical fixes only |
| < 1.5   | :x:                | End of life  |

## Security Features

Nexus-LLM includes several built-in security features:

### Input/Output Filtering
- **Toxicity detection**: Classifies harmful content across 6 categories
- **PII redaction**: Automatically detects and redacts personally identifiable information
- **Prompt injection detection**: Identifies and blocks injection attempts

### API Security
- **API key authentication**: Secure key generation and hashed storage
- **JWT tokens**: Time-limited access tokens with refresh mechanism
- **Rate limiting**: Per-user request and token quotas
- **CORS**: Configurable cross-origin resource sharing

### Model Security
- **Sandboxed tool execution**: Agent tools run in isolated environments
- **Output validation**: Safety filters check model responses before delivery
- **Conversation isolation**: Sessions are independent and cannot access each other

## Security Best Practices

### For Deployment

1. **Always use HTTPS** in production. Never serve the API over plain HTTP.
2. **Rotate API keys** regularly. Set up automatic rotation every 90 days.
3. **Use strong JWT secrets** — at least 32 characters of random data.
4. **Enable rate limiting** to prevent abuse and denial-of-service attacks.
5. **Restrict CORS origins** to your actual frontend domains.
6. **Keep dependencies updated** — subscribe to security advisories.
7. **Run as non-root** — the server should never run with root privileges.
8. **Use network isolation** — place the server behind a firewall or load balancer.

### For Fine-Tuning

1. **Audit training data** for harmful content before fine-tuning.
2. **Test for jailbreaks** after fine-tuning using red-teaming techniques.
3. **Apply safety alignment** (DPO/RLHF) after domain-specific fine-tuning.

### For RAG Pipelines

1. **Validate documents** before ingestion — malicious content in documents can influence outputs.
2. **Set similarity thresholds** to prevent retrieval of irrelevant or harmful content.
3. **Monitor retrieval patterns** for signs of data exfiltration attempts.

## Disclosure Policy

When a vulnerability is reported:

1. We confirm the vulnerability and determine its scope.
2. We develop a fix and test it thoroughly.
3. We release a patch version as soon as possible.
4. We publish a security advisory with details after the patch is available.
5. We credit the reporter (unless they prefer to remain anonymous).

## Security Contact

- **Email**: security@nexus-llm.dev
- **PGP Key**: Available at https://nexus-llm.dev/security.asc
- **Response time**: 24-72 hours for initial acknowledgment
