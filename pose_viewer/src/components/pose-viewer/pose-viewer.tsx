import {Component, Element, Event, EventEmitter, h, Method, Prop, State, Watch} from '@stencil/core';

import {Pose, PoseLimb, PoseModel, PosePointModel, RGBColor} from "pose-format";


@Component({
  tag: 'pose-viewer',
  styleUrl: 'pose-viewer.css',
  shadow: true
})
export class PoseViewer {
  @Element() element: HTMLElement;
  private resizeObserver: ResizeObserver;

  @Prop() src: string; // Source URL for .pose file

  // Dimensions
  @Prop() width: string = null;
  @Prop() height: string = null;
  @Prop() aspectRatio: string = null;

  elWidth: number;
  elHeight: number;

  // MediaElement-like properties
  @Prop({mutable: true}) loop: boolean = false;
  @Prop() autoplay: boolean = true;
  @Prop({mutable: true}) playbackRate: number = 1;

  @State() currentTime: number = NaN; // This affects re-rendering
  duration: number = NaN;
  ended: boolean = false;
  paused: boolean = true;
  readyState: number = 0;

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


  media: HTMLMediaElement;
  pose: PoseModel;

  private loopInterval: any;

  componentWillLoad() {
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

  // Render functions
  x(v: number) {
    return v * this.elWidth / this.pose.header.width;
  }

  y(v: number) {
    return v * this.elHeight / this.pose.header.height;
  }

  isJointValid(joint: PosePointModel) {
    return joint.C > 0;
  }

  renderJoints(joints: PosePointModel[], colors: RGBColor[]) {
    return joints
      .filter(this.isJointValid.bind(this))
      .map((joint, i) => {
        const {R, G, B} = colors[i % colors.length];
        return (<circle
          cx={this.x(joint.X)}
          cy={this.y(joint.Y)}
          r={4}
          class="joint draggable"
          style={{
            fill: `rgb(${R}, ${G}, ${B})`,
            opacity: String(joint.C)
          }}
          data-id={i}>
        </circle>);
      });
  }

  renderLimbs(limbs: PoseLimb[], joints: PosePointModel[], colors: RGBColor[]) {
    return limbs.map(({from, to}) => {
      const a = joints[from];
      const b = joints[to];
      if (!this.isJointValid(a) || !this.isJointValid(b)) {
        return "";
      }

      const c1 = colors[from % colors.length];
      const c2 = colors[to % colors.length];
      const {R, G, B} = {
        R: (c1.R + c2.R) / 2,
        G: (c1.G + c2.G) / 2,
        B: (c1.B + c2.B) / 2,
      };

      return (<line
        x1={this.x(joints[from].X)}
        y1={this.y(joints[from].Y)}
        x2={this.x(joints[to].X)}
        y2={this.y(joints[to].Y)}
        style={{
          stroke: `rgb(${R}, ${G}, ${B})`,
          opacity: String((joints[from].C + joints[to].C) / 2)
        }}>
      </line>);
    });
  }


  render() {
    if (!this.pose || isNaN(this.currentTime)) {
      return "";
    }

    const currentTime = this.currentTime > this.duration ? this.duration : this.currentTime;

    const frameId = Math.floor(currentTime * this.pose.body.fps);
    const frame = this.pose.body.frames[frameId];

    return (
      <svg xmlns="http://www.w3.org/2000/svg" width={this.elWidth} height={this.elHeight}>
        <g>
          {frame.people.map(person => this.pose.header.components.map(component => {
            const joints = person[component.name];
            return [
              this.renderLimbs(component.limbs, joints, component.colors),
              this.renderJoints(joints, component.colors),
            ]
          }))}
        </g>
      </svg>
    )
  }
}


