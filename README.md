# WikiSearchBackend
This is wiki search backend. To run it, you need:
- [qdrant](https://github.com/qdrant/qdrant/releases/download/v1.20.1/qdrant-x86_64-unknown-linux-musl.tar.gz)
- [bge-small-en-v1.5-q8_0.gguf](https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf)
- [simplewiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2)

Start qdrant and llamaserver. After that, run index and embed once:
```
uv run miniwiki index simplewiki-latest-pages-articles.xml.bz2 --lang english
uv run miniwiki embed --intro-only
```

This process will take a while. For bge-small running on cpu (Ryzen 5 3500), it takes about 2 hours

After indexing, try searching using this command:
```
uv run miniwiki search "cats"
```

Alternatively, you can serve an HTTP api:
```
uv run miniwiki serve
```
