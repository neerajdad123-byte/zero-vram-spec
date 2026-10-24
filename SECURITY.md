# Security Policy

## Supported versions

| version | supported |
|---|---|
| 0.2.x | yes |
| < 0.2 | no |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report them privately through GitHub: go to the repository's **Security** tab and choose **Report a vulnerability**. Include:

- what the problem is and where it is (file, command or endpoint)
- steps to reproduce it
- the impact you expect

You should get a reply within 7 days. Once a fix is ready it will be released, and you will be credited in the advisory unless you prefer not to be.

## Scope notes

- `structspec serve` binds to `127.0.0.1` by default and has no authentication. Exposing it on a network (`--host 0.0.0.0`) lets anyone who can reach the port run generations on your machine. Put it behind your own auth if you do that.
- Structspec loads GGUF models and llama.cpp libraries you point it to. Only use model files and llama.cpp builds from sources you trust.
