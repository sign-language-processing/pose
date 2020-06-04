import {Config} from '@stencil/core';
import nodePolyfills from 'rollup-plugin-node-polyfills';

export const config: Config = {
  namespace: 'viewer',
  plugins: [
    nodePolyfills(),
  ],
  outputTargets: [
    {
      type: 'dist',
      esmLoaderPath: '../loader'
    },
    {
      type: 'docs-readme'
    },
    {
      type: 'www',
      copy: [
        {src: 'data', dest: 'data'},
        {src: '../../sample-data', dest: 'sample-data'}
      ],
      serviceWorker: null // disable service workers
    }
  ]
};
