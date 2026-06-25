const { pipeline } = require('@huggingface/transformers');
const { HierarchicalNSW } = require('hnswlib-node');

class BiMap {
  constructor(entries = []) {
    this._forward = new Map(entries);
    this._backward = new Map(entries.map(([k, v]) => [v, k]));
  }
  set(key, value) {
    // 删除已有的正向/反向映射，保证一对一
    if (this._forward.has(key)) {
      const oldVal = this._forward.get(key);
      this._backward.delete(oldVal);
    }
    if (this._backward.has(value)) {
      const oldKey = this._backward.get(value);
      this._forward.delete(oldKey);
    }
    this._forward.set(key, value);
    this._backward.set(value, key);
  }
  get(key) { return this._forward.get(key); }
  getKey(value) { return this._backward.get(value); }
  has(key) { return this._forward.has(key); }
  hasValue(value) { return this._backward.has(value); }
  delete(key) {
    if (!this._forward.has(key)) return false;
    const value = this._forward.get(key);
    this._forward.delete(key);
    this._backward.delete(value);
    return true;
  }
  get length() { return this._forward.size; }
}

const numDimensions = 384; // the length of data point vector that will be indexed.
const maxElements = 1024; // the maximum number of data points.
const modelName = 'Xenova/all-MiniLM-L6-v2';

// declaring and intializing index.
const index = new HierarchicalNSW('l2', numDimensions);
index.initIndex(maxElements);

let extractor;
const labelMapping = new BiMap();

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = units.shift();
  while (value >= 1024 && units.length) {
    value /= 1024;
    unit = units.shift();
  }
  return `${value.toFixed(value >= 10 || unit === 'B' ? 0 : 1)} ${unit}`;
}

function createProgressLogger(log) {
  const lastLogged = new Map();
  let activeDownload;

  return data => {
    if (!data) return;

    const key = data.file
      ? `${data.name || modelName}/${data.file}`
      : activeDownload;

    if (data.status === 'initiate') {
      log.info(`Loading embedding model file: ${key}`);
      return;
    }

    if (data.status === 'download') {
      activeDownload = key;
      return;
    }

    if (data.status === 'done') {
      if (activeDownload === key) activeDownload = null;
      return;
    }

    if (!key) return;
    if (typeof data.progress !== 'number') return;
    if (data.status && data.status !== 'progress') return;

    const progress = Math.floor(data.progress);
    const previous = lastLogged.get(key) || 0;
    if (progress < 100 && progress - previous < 5) return;

    lastLogged.set(key, progress);

    const width = 20;
    const filled = Math.round((Math.min(progress, 100) / 100) * width);
    const bar = `${'#'.repeat(filled)}${'-'.repeat(width - filled)}`;
    const size = data.total
      ? ` (${formatBytes(data.loaded)} / ${formatBytes(data.total)})`
      : '';

    log.info(`Downloading ${key} [${bar}] ${progress}%${size}`);
  };
}

hexo.extend.filter.register('after_init', async function() {
  const log = this.log || hexo.log;
  log.info(`Loading embedding model: ${modelName}`);
  extractor = await pipeline('feature-extraction', modelName, {
    progress_callback: createProgressLogger(log)
  });
  log.info(`Embedding model ready: ${modelName}`);
});

hexo.extend.filter.register('before_post_render', async function(data) {
  const embeddings = await extractor([data._content], { pooling: 'mean', normalize: true });
  data.embedding_vector = embeddings.tolist()[0];
  // Create a new id if data.path doesn't exist in labelMapping
  // Else use the existing id
  let id;
  if (!labelMapping.hasValue(data.path)) {
    id = labelMapping.length;
    labelMapping.set(id, data.path);
  } else {
    id = labelMapping.getKey(data.path);
  }
  index.addPoint(data.embedding_vector, id);
  return data;
});

hexo.extend.helper.register('related_posts', function(post) {
  const result = [];
  if (!post.embedding_vector) {
    post.related_posts = result;
    return result;
  }
  const numNeighbors = 5;
  const query = post.embedding_vector;
  const { neighbors } = index.searchKnn(query, numNeighbors);
  // Skip the first result as it is the query itself
  for (let i = 1; i < neighbors.length; i++) {
    const neighbor = neighbors[i];
    result.push(labelMapping.get(neighbor));
  }
  post.related_posts = result;
  return result;
});
