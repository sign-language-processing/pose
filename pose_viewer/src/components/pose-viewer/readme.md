# my-component



<!-- Auto Generated Below -->


## Properties

| Property       | Attribute       | Description | Type      | Default     |
| -------------- | --------------- | ----------- | --------- | ----------- |
| `aspectRatio`  | `aspect-ratio`  |             | `number`  | `null`      |
| `autoplay`     | `autoplay`      |             | `boolean` | `true`      |
| `background`   | `background`    |             | `string`  | `null`      |
| `currentTime`  | `current-time`  |             | `number`  | `NaN`       |
| `duration`     | `duration`      |             | `number`  | `NaN`       |
| `ended`        | `ended`         |             | `boolean` | `false`     |
| `height`       | `height`        |             | `string`  | `null`      |
| `loop`         | `loop`          |             | `boolean` | `false`     |
| `padding`      | `padding`       |             | `string`  | `null`      |
| `paused`       | `paused`        |             | `boolean` | `true`      |
| `playbackRate` | `playback-rate` |             | `number`  | `1`         |
| `readyState`   | `ready-state`   |             | `number`  | `0`         |
| `src`          | `src`           |             | `string`  | `undefined` |
| `svg`          | `svg`           |             | `boolean` | `false`     |
| `thickness`    | `thickness`     |             | `number`  | `null`      |
| `width`        | `width`         |             | `string`  | `null`      |


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


## Methods

### `getPose() => Promise<any>`



#### Returns

Type: `Promise<any>`



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



#### Returns

Type: `Promise<void>`




----------------------------------------------

*Built with [StencilJS](https://stenciljs.com/)*
