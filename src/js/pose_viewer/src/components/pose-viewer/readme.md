# my-component



<!-- Auto Generated Below -->


## Properties

| Property       | Attribute       | Description | Type                                 | Default     |
| -------------- | --------------- | ----------- | ------------------------------------ | ----------- |
| `aspectRatio`  | `aspect-ratio`  |             | `number`                             | `null`      |
| `autoplay`     | `autoplay`      |             | `boolean`                            | `true`      |
| `background`   | `background`    |             | `string`                             | `null`      |
| `currentTime`  | `current-time`  |             | `number`                             | `NaN`       |
| `duration`     | `duration`      |             | `number`                             | `NaN`       |
| `ended`        | `ended`         |             | `boolean`                            | `false`     |
| `height`       | `height`        |             | `string`                             | `null`      |
| `loop`         | `loop`          |             | `boolean`                            | `false`     |
| `padding`      | `padding`       |             | `string`                             | `null`      |
| `paused`       | `paused`        |             | `boolean`                            | `true`      |
| `playbackRate` | `playback-rate` |             | `number`                             | `1`         |
| `readyState`   | `ready-state`   |             | `number`                             | `0`         |
| `renderer`     | `renderer`      |             | `"canvas" \| "interactive" \| "svg"` | `'canvas'`  |
| `src`          | `src`           |             | `Buffer \| string`                   | `undefined` |
| `thickness`    | `thickness`     |             | `number`                             | `null`      |
| `width`        | `width`         |             | `string`                             | `null`      |


## Events

| Event             | Description | Type                |
| ----------------- | ----------- | ------------------- |
| `canplaythrough$` |             | `CustomEvent<void>` |
| `ended$`          |             | `CustomEvent<void>` |
| `firstRender$`    |             | `CustomEvent<void>` |
| `loadeddata$`     |             | `CustomEvent<void>` |
| `loadedmetadata$` |             | `CustomEvent<void>` |
| `loadstart$`      |             | `CustomEvent<void>` |
| `pause$`          |             | `CustomEvent<void>` |
| `play$`           |             | `CustomEvent<void>` |
| `render$`         |             | `CustomEvent<void>` |


## Methods

### `getPose() => Promise<PoseModel>`



#### Returns

Type: `Promise<PoseModel>`



### `nextFrame() => Promise<void>`



#### Returns

Type: `Promise<void>`



### `pause() => Promise<void>`



#### Returns

Type: `Promise<void>`



### `play() => Promise<void>`



#### Returns

Type: `Promise<void>`



### `syncMedia(media: HTMLMediaElement) => Promise<void>`



#### Parameters

| Name    | Type               | Description |
| ------- | ------------------ | ----------- |
| `media` | `HTMLMediaElement` |             |

#### Returns

Type: `Promise<void>`




----------------------------------------------

*Built with [StencilJS](https://stenciljs.com/)*
