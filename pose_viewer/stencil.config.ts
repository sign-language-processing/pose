import {Config} from '@stencil/core';
import nodePolyfills from 'rollup-plugin-node-polyfills';

export const config: Config = {
  namespace: 'pose-viewer',
  buildEs5: 'prod',
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
        {src: '../../sample-data', dest: 'sample-data'}
      ],
      serviceWorker: null // disable service workers
    }
  ]
};
