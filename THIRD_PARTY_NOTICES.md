# Third-party software and models

On-Prem RAG Assistant connects to software and models that are licensed
independently from this project's MIT License.

| Component | Role | License information |
|---|---|---|
| Ollama | Local model runtime | [MIT License](https://github.com/ollama/ollama/blob/main/LICENSE) |
| Qwen2.5 7B Instruct | Default answer model | [Apache 2.0 model repository](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Nomic Embed Text | Default embedding model | [Apache 2.0 model repository](https://huggingface.co/nomic-ai/nomic-embed-text-v1) |

The setup scripts download models through Ollama; this repository does not
contain or redistribute model weights. Users who select other models are
responsible for reviewing their licenses and acceptable-use terms.
