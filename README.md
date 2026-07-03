# Hexo Posts Embedding

[![npm-image]][npm-url]
[![lic-image]](LICENSE)

Posts embedding for Hexo.

## Installation

![size-image]
[![dm-image]][npm-url]
[![dt-image]][npm-url]

```bash
npm install hexo-posts-embedding
hexo clean
```

## Configuration

By default, model files are cached in the Hexo site directory:

```yaml
posts_embedding:
  cache_dir: .cache/hexo-posts-embedding
```

Set `cache_dir` to an absolute path to share the cache across projects, or set it to `false` to use the default Transformers.js cache directory.

## License

Released under the MIT License

[npm-image]: https://img.shields.io/npm/v/hexo-posts-embedding?style=for-the-badge
[lic-image]: https://img.shields.io/npm/l/hexo-posts-embedding?style=for-the-badge

[size-image]: https://img.shields.io/github/languages/code-size/next-theme/hexo-posts-embedding?style=for-the-badge
[dm-image]: https://img.shields.io/npm/dm/hexo-posts-embedding?style=for-the-badge
[dt-image]: https://img.shields.io/npm/dt/hexo-posts-embedding?style=for-the-badge

[npm-url]: https://www.npmjs.com/package/hexo-posts-embedding
