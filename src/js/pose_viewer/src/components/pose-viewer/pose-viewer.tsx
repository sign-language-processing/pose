// @ts-ignore
import {Component, Element, Event, EventEmitter, h, Host, Method, Prop, State, Watch} from '@stencil/core';
import type {PoseModel} from "pose-format/dist/types";
import {Pose} from "pose-format";
import type {Buffer} from "buffer";
// import {Pose, PoseModel} from "../../../../pose_format/dist";
import {PoseRenderer} from "./renderers/pose-renderer";
import {SVGPoseRenderer} from "./renderers/svg.pose-renderer";
import {CanvasPoseRenderer} from "./renderers/canvas.pose-renderer";
import {InteractivePoseRenderer} from "./renderers/interactive.pose-renderer";

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
  private fetchAbortController: AbortController;

  private lastSrc: string | Buffer;
  @Prop() src: string | Buffer; // Source URL for .pose file or path to a local file or Buffer
  @Prop() renderer: 'canvas' | 'svg' | 'interactive' = 'canvas'; // Render in an SVG instead of a canvas

  // Dimensions
  @Prop() width: string = null;
  @Prop() height: string = null;
  @Prop() aspectRatio: number = null;
  @Prop() padding: string = null;
  @Prop() thickness: number = null;

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
  @Event() firstRender$: EventEmitter<void>;
  @Event() render$: EventEmitter<void>;
  // @Event() ratechange$: EventEmitter<void>;
  // @Event() seeked$: EventEmitter<void>;
  // @Event() seeking$: EventEmitter<void>;
  // @Event() timeupdate$: EventEmitter<void>;

  hasRendered = false;

  rendererInstance!: PoseRenderer;

  media: HTMLMediaElement;
  pose: PoseModel;

  @State() error: Error;

  private loopInterval: any;

  componentWillLoad() {
    switch (this.renderer) {
      case 'canvas':
        this.rendererInstance = new CanvasPoseRenderer(this);
        break;
      case 'svg':
        this.rendererInstance = new SVGPoseRenderer(this);
        break;
      case 'interactive':
        this.rendererInstance = new InteractivePoseRenderer(this);
        break;
      default:
        throw new Error('Invalid renderer');
    }

    return this.srcChange();
  }

  componentDidLoad() {
    this.resizeObserver = new ResizeObserver(this.setDimensions.bind(this));
    this.resizeObserver.observe(this.element);
  }

  private async loadPose() {
    // Abort previous request if it exists
    if (this.fetchAbortController) {
      this.fetchAbortController.abort();
    }

    if(typeof this.src === 'string') {
      const isRemoteUrl = this.src.startsWith('http') || this.src.startsWith('//');
      const isBrowserEnvironment = typeof window !== 'undefined';

      if (isRemoteUrl || isBrowserEnvironment) {
        // Remote URL or Browser environment
        this.fetchAbortController = new AbortController();
        this.pose = await Pose.fromRemote(this.src, this.fetchAbortController);
      } else {
        // Local environment
        this.pose = await Pose.fromLocal(this.src);
      }
    } else {
      this.pose = Pose.from(this.src);
    }
  }

  private initPose() {
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

  @Watch('src')
  async srcChange() {
    // Can occur from both an attribute change AND componentWillLoad event
    if (this.src === this.lastSrc) {
      return;
    }
    this.lastSrc = this.src;

    // Clear previous pose
    this.clearInterval();
    this.setDimensions();
    delete this.pose;
    this.currentTime = NaN;
    this.duration = NaN;
    this.hasRendered = false;

    if (!this.src) {
      return;
    }
    // Load new pose
    this.ended = false;
    this.loadstart$.emit();

    this.error = null;
    try {
      await this.loadPose();
      this.initPose();
      this.error = null;
    } catch (e) {
      console.error('PoseViewer error', e);
      this.error = e;
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
    } else if (this.width) {
      this.elWidth = parseSize(this.width, rect.width);
      this.elHeight = this.aspectRatio ? this.elWidth * this.aspectRatio :
        (this.pose.header.height / this.pose.header.width) * this.elWidth;
    } else if (this.height) {
      this.elHeight = parseSize(this.height, rect.height);
      this.elWidth = this.aspectRatio ? this.elHeight / this.aspectRatio :
        (this.pose.header.width / this.pose.header.height) * this.elHeight;
    }

    // General padding
    if (this.padding) {
      this.elPadding.width += parseSize(this.padding, this.elWidth);
      this.elPadding.height += parseSize(this.padding, this.elHeight);
    }

    // Aspect ratio padding
    const ratioWidth = this.elWidth - this.elPadding.width * 2;
    const ratioHeight = this.elHeight - this.elPadding.height * 2;
    const elAR = ratioWidth / ratioHeight;
    const poseAR = this.pose.header.width / this.pose.header.height;
    if (poseAR > elAR) {
      this.elPadding.height += (poseAR - elAR) * ratioHeight / 2;
    } else {
      this.elPadding.width += (elAR - poseAR) * ratioWidth / 2;
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
  async getPose() {
    return this.pose;
  }


  @Method()
  async nextFrame() {
    const newTime = this.currentTime + 1 / this.pose.body.fps;
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

  @Method()
  async play() {
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

  @Method()
  async pause() {
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
    if (this.error) {
      return this.error.name !== "AbortError" ? this.error.message : "";
    }

    if (!this.pose || isNaN(this.currentTime) || !this.rendererInstance) {
      return "";
    }

    const currentTime = this.currentTime > this.duration ? this.duration : this.currentTime;

    const frameId = Math.floor(currentTime * this.pose.body.fps);
    const frame = this.pose.body.frames[frameId];
    console.log(frame);

    const render = this.rendererInstance.render(frame);
    if (!this.hasRendered) {
      requestAnimationFrame(() => {
        this.hasRendered = true;
        this.firstRender$.emit();
      });
    }
    requestAnimationFrame(() => this.render$.emit());

    return (<Host>{render}</Host>);
  }
}


