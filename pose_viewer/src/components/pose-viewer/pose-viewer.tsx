// @ts-ignore
import {Component, Element, Event, EventEmitter, h, Host, Method, Prop, Watch} from '@stencil/core';
import {Pose, PoseModel} from "pose-format";
import {PoseRenderer} from "./renderers/pose-renderer";
import {SVGPoseRenderer} from "./renderers/svg.pose-renderer";
import {CanvasPoseRenderer} from "./renderers/canvas.pose-renderer";

declare type ResizeObserver = any;
declare var ResizeObserver: ResizeObserver;


@Component({
  tag: 'pose-viewer',
  styleUrl: 'pose-viewer.css',
  shadow: true
})
export class PoseViewer {
  @Element() element: HTMLElement;
  private resizeObserver: ResizeObserver;

  @Prop() src: string; // Source URL for .pose file
  @Prop() svg: boolean = false; // Render in an SVG instead of a canvas

  // Dimensions
  @Prop() width: string = null;
  @Prop() height: string = null;

  @Prop() background: string = null;

  elWidth: number;
  elHeight: number;
  elPadding: { width: number, height: number };

  // MediaElement-like properties
  @Prop({mutable: true}) loop: boolean = false;
  @Prop() autoplay: boolean = true;
  @Prop({mutable: true}) playbackRate: number = 1;

  @Prop({mutable: true}) currentTime: number = NaN; // This affects re-rendering
  @Prop({mutable: true}) duration: number = NaN;
  @Prop({mutable: true}) ended: boolean = false;
  @Prop({mutable: true}) paused: boolean = true;
  @Prop({mutable: true}) readyState: number = 0;

  // MediaElement-like events
  @Event() canplaythrough$: EventEmitter<void>;
  @Event() ended$: EventEmitter<void>;
  @Event() loadeddata$: EventEmitter<void>;
  @Event() loadedmetadata$: EventEmitter<void>;
  @Event() loadstart$: EventEmitter<void>;
  @Event() pause$: EventEmitter<void>;
  @Event() play$: EventEmitter<void>;
  // @Event() ratechange$: EventEmitter<void>;
  // @Event() seeked$: EventEmitter<void>;
  // @Event() seeking$: EventEmitter<void>;
  // @Event() timeupdate$: EventEmitter<void>;

  renderer!: PoseRenderer;

  media: HTMLMediaElement;
  pose: PoseModel;

  private loopInterval: any;

  componentWillLoad() {
    this.renderer = this.svg ? new SVGPoseRenderer(this) : new CanvasPoseRenderer(this);

    return this.srcChange();
  }

  componentDidLoad() {
    this.resizeObserver = new ResizeObserver(this.setDimensions.bind(this));
    this.resizeObserver.observe(this.element);
  }

  @Watch('src')
  async srcChange() {
    // Clear previous pose
    this.clearInterval();
    this.setDimensions();
    delete this.pose;
    if (!this.src) {
      return;
    }
    // Load new pose
    this.loadstart$.emit();
    this.pose = await Pose.fromRemote(this.src);

    this.setDimensions();

    // Loaded done events
    this.loadedmetadata$.emit();
    this.loadeddata$.emit();
    this.canplaythrough$.emit();

    this.duration = (this.pose.body.frames.length - 1) / this.pose.body.fps;
    this.currentTime = 0;

    if (this.autoplay) {
      this.play();
    }
  }

  setDimensions() {
    this.elPadding = {width: 0, height: 0};
    if (!this.pose) {
      this.elWidth = 0;
      this.elHeight = 0;
      return;
    }

    // When nothing is marked, use pose dimensions
    if (!this.width && !this.height) {
      this.elWidth = this.pose.header.width;
      this.elHeight = this.pose.header.height;
      return;
    }

    const rect = this.element.getBoundingClientRect();
    const parseSize = (size, by) => size.endsWith("px") ? Number(size.slice(0, -2)) : (size.endsWith("%") ? by * size.slice(0, -1) / 100 : Number(size));

    // When both are marked,
    if (this.width && this.height) {
      this.elWidth = parseSize(this.width, rect.width);
      this.elHeight = parseSize(this.height, rect.height);

      const elAR = this.elWidth / this.elHeight;
      const poseAR = this.pose.header.width / this.pose.header.height;
      if (poseAR > elAR) {
        this.elPadding.height = (poseAR - elAR) * this.elHeight / 2;
      } else {
        this.elPadding.width = (1 / elAR - 1 / poseAR) * this.elWidth / 2;
      }

    } else if (this.width) {
      this.elWidth = parseSize(this.width, rect.width);
      this.elHeight = (this.pose.header.height / this.pose.header.width) * this.elWidth;
    } else if (this.height) {
      this.elHeight = parseSize(this.height, rect.height);
      this.elWidth = (this.pose.header.width / this.pose.header.height) * this.elHeight;
    }
  }

  @Method()
  async syncMedia(media: HTMLMediaElement) {
    this.media = media;

    this.media.addEventListener('pause', this.pause.bind(this));
    this.media.addEventListener('play', this.play.bind(this));
    const syncTime = () => this.currentTime = this.frameTime(this.media.currentTime);
    this.media.addEventListener('seek', syncTime);
    this.media.addEventListener('timeupdate', syncTime); // To always keep synced

    // Others
    const updateRate = () => this.playbackRate = this.media.playbackRate;
    this.media.addEventListener('ratechange', updateRate);
    updateRate();

    // Start the pose according to the video
    this.clearInterval();
    if (this.media.paused) {
      this.pause();
    } else {
      this.play();
    }
  }

  @Method()
  async nextFrame() {
    const newTime = this.currentTime + 1 / this.pose.body.fps
    if (newTime > this.duration) {
      if (this.loop) {
        this.currentTime = newTime % this.duration;
      } else {
        this.ended$.emit();
        this.ended = true;
      }
    } else {
      this.currentTime = newTime;
    }
  }

  frameTime(time: number) {
    if (!this.pose) {
      return 0;
    }
    return Math.floor(time * this.pose.body.fps) / this.pose.body.fps;
  }

  play() {
    if (!this.paused) {
      this.clearInterval();
    }

    this.paused = false;
    this.play$.emit();

    // Reset clip if exceeded duration
    if (this.currentTime > this.duration) {
      this.currentTime = 0;
    }

    const intervalTime = 1000 / (this.pose.body.fps * this.playbackRate);
    if (this.media) {
      this.loopInterval = setInterval(() => this.currentTime = this.frameTime(this.media.currentTime), intervalTime);
    } else {
      // Add the time passed in an interval.
      let lastTime = Date.now() / 1000;
      this.loopInterval = setInterval(() => {
        const now = Date.now() / 1000;
        this.currentTime += (now - lastTime) * this.playbackRate;
        lastTime = now;
        if (this.currentTime > this.duration) {
          if (this.loop) {
            this.currentTime = this.currentTime % this.duration;
          } else {
            this.ended$.emit();
            this.ended = true;

            this.clearInterval();
          }
        }
      }, intervalTime);
    }
  }

  pause() {
    this.paused = true;
    this.pause$.emit();
    this.clearInterval();
  }

  clearInterval() {
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
  }

  disconnectedCallback() {
    this.clearInterval();
  }

  render() {
    if (!this.pose || isNaN(this.currentTime) || !this.renderer) {
      return "";
    }

    const currentTime = this.currentTime > this.duration ? this.duration : this.currentTime;

    const frameId = Math.floor(currentTime * this.pose.body.fps);
    const frame = this.pose.body.frames[frameId];

    return (
      <Host>
        {this.renderer.render(frame)}
      </Host>
    );
  }
}


